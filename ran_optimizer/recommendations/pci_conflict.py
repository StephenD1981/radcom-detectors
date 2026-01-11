"""
PCI Collision Detector - Detects cells with same PCI and overlapping coverage areas.

Physical Cell ID (PCI) collisions occur when cells on the same frequency band have:
1. The same PCI value (or same PCI mod 3/mod 30 for reference signal interference)
2. Overlapping coverage areas (hull overlap > threshold)

This causes mobile devices to confuse the two cells during cell search and initial access.

Note: This module detects PCI COLLISIONS (coverage overlap). For PCI CONFUSION detection
(neighbor relation ambiguity during handover), see pci_planner.py which uses HO data.

3GPP Reference: TS 36.211 (LTE), TS 38.211 (NR) for PCI and reference signal design.
"""

import geopandas as gpd
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from shapely.strtree import STRtree

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


# PCI range constants per 3GPP specifications
LTE_PCI_MAX = 503   # LTE: 504 PCIs (0-503), TS 36.211
NR_PCI_MAX = 1007   # NR: 1008 PCIs (0-1007), TS 38.211

# Technology prefixes for band identification
TECH_PREFIXES = {
    'L': 'LTE',   # L700, L800, L1800, L2100, L2600
    'N': 'NR',    # N78, N258, N260
    'U': 'UMTS',  # U900, U2100
    'G': 'GSM',   # G900, G1800
}


def _get_tech_from_band(band: str) -> str:
    """Extract technology from band string (e.g., 'L800' -> 'LTE', 'N78' -> 'NR')."""
    if not band or not isinstance(band, str):
        return 'UNKNOWN'
    band_upper = band.upper().strip()
    if not band_upper:
        return 'UNKNOWN'
    prefix = band_upper[0]
    return TECH_PREFIXES.get(prefix, 'UNKNOWN')


def _get_pci_max_for_band(band: str) -> int:
    """Get maximum valid PCI value for a band's technology."""
    tech = _get_tech_from_band(band)
    if tech == 'NR':
        return NR_PCI_MAX
    return LTE_PCI_MAX  # Default to LTE for LTE/UMTS/GSM/unknown


@dataclass
class PCIConflictParams:
    """Parameters for PCI conflict detection."""
    overlap_threshold: float = 0.10  # 10% minimum overlap
    min_overlap_area_km2: float = 0.0005  # 500m² minimum
    severity_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'critical': {'overlap_pct': 50, 'distance_km': 2},
        'high': {'overlap_pct': 30, 'distance_km': 3},
        'medium': {'overlap_pct': 20, 'distance_km': 5}
    })
    # Same-site filtering
    filter_same_site: bool = True  # Filter out same-site cell pairs (expected overlap)
    site_id_column: str = 'site_id'  # Column name for site identifier
    # Mod 3/30 interference detection (3GPP TS 36.211)
    check_mod3_conflicts: bool = True  # Check PCI mod 3 (PSS interference)
    check_mod30_conflicts: bool = False  # Check PCI mod 30 (RS interference) - optional
    mod3_overlap_threshold: float = 0.30  # Higher threshold for mod conflicts (30%)
    mod30_overlap_threshold: float = 0.30  # Higher threshold for mod conflicts (30%)

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'PCIConflictParams':
        """Load parameters from config file or use defaults."""
        if config_path is None:
            return cls()

        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            return cls(
                overlap_threshold=config.get('overlap_threshold', 0.10),
                min_overlap_area_km2=config.get('min_overlap_area_km2', 0.0005),
                severity_thresholds=config.get('severity_thresholds', cls().severity_thresholds),
                filter_same_site=config.get('filter_same_site', True),
                site_id_column=config.get('site_id_column', 'site_id'),
                check_mod3_conflicts=config.get('check_mod3_conflicts', True),
                check_mod30_conflicts=config.get('check_mod30_conflicts', False),
                mod3_overlap_threshold=config.get('mod3_overlap_threshold', 0.30),
                mod30_overlap_threshold=config.get('mod30_overlap_threshold', 0.30),
            )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return cls()


class PCIConflictDetector:
    """Detects PCI collisions using per-band convex hull overlap analysis."""

    def __init__(
        self,
        params: Optional[PCIConflictParams] = None,
        target_crs: Optional[str] = None
    ):
        """
        Initialize the PCI Conflict Detector.

        Args:
            params: Detection parameters (uses defaults if None)
            target_crs: Target CRS for area/distance calculations (auto-detected if None)
        """
        self.params = params or PCIConflictParams()
        self.target_crs = target_crs
        self.input_crs = None  # Will be set from input GeoDataFrame

        logger.info(
            "PCI Conflict detector initialized",
            overlap_threshold=self.params.overlap_threshold,
            min_overlap_area_km2=self.params.min_overlap_area_km2,
            filter_same_site=self.params.filter_same_site,
            check_mod3=self.params.check_mod3_conflicts,
            check_mod30=self.params.check_mod30_conflicts,
        )

    def detect(self, hulls_gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """
        Detect PCI conflicts across all bands.

        Args:
            hulls_gdf: GeoDataFrame with columns: cell_name, geometry, band, pci
                       Optional: site_id (for same-site filtering)

        Returns:
            List of conflict dictionaries with cell pairs and overlap details
        """
        logger.info("Starting PCI conflict detection")

        # Validate required columns
        required_cols = ['cell_name', 'geometry', 'band', 'pci']
        missing = [c for c in required_cols if c not in hulls_gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Store input CRS for later use
        self.input_crs = hulls_gdf.crs if hulls_gdf.crs else 'EPSG:4326'

        # Filter out cells with missing band or PCI
        valid_hulls = hulls_gdf[
            hulls_gdf['band'].notna() & hulls_gdf['pci'].notna()
        ].copy()

        # Validate PCI range per band technology
        valid_hulls = self._validate_pci_range(valid_hulls)

        logger.info(f"Analyzing {len(valid_hulls)} cells with band and PCI info")

        conflicts = []

        # Process each band separately
        for band in valid_hulls['band'].unique():
            logger.info(f"Checking PCI conflicts for band {band}")
            band_conflicts = self._detect_conflicts_per_band(valid_hulls, band)
            conflicts.extend(band_conflicts)

        logger.info(f"Detected {len(conflicts)} PCI conflicts")
        return conflicts

    def _validate_pci_range(self, hulls_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate PCI values are within valid range for each band's technology."""
        invalid_pci_cells = []

        for band in hulls_gdf['band'].unique():
            pci_max = _get_pci_max_for_band(band)
            band_mask = hulls_gdf['band'] == band
            band_hulls = hulls_gdf[band_mask]

            # Check for invalid PCI values
            invalid_mask = (band_hulls['pci'] < 0) | (band_hulls['pci'] > pci_max)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                invalid_cells = band_hulls.loc[invalid_mask, 'cell_name'].head(5).tolist()
                invalid_pcis = band_hulls.loc[invalid_mask, 'pci'].unique().tolist()
                invalid_pci_cells.extend(band_hulls.loc[invalid_mask].index.tolist())
                logger.warning(
                    f"Found {invalid_count} cells with invalid PCI for band {band} "
                    f"(valid range: 0-{pci_max})",
                    sample_cells=invalid_cells,
                    invalid_pcis=invalid_pcis[:5],
                )

        # Remove cells with invalid PCI
        if invalid_pci_cells:
            hulls_gdf = hulls_gdf.drop(index=invalid_pci_cells)
            logger.info(f"Removed {len(invalid_pci_cells)} cells with invalid PCI values")

        return hulls_gdf

    def _detect_conflicts_per_band(
        self, hulls_gdf: gpd.GeoDataFrame, band: str
    ) -> List[Dict[str, Any]]:
        """
        Detect PCI conflicts within a single frequency band.

        Detects:
        1. Exact PCI collisions (same PCI, overlapping coverage)
        2. PCI mod 3 conflicts (PSS interference) if enabled
        3. PCI mod 30 conflicts (RS interference) if enabled

        Args:
            hulls_gdf: GeoDataFrame with all cells
            band: Frequency band to analyze (e.g., 'L800', 'L1800')

        Returns:
            List of conflicts for this band
        """
        # Filter to this band only
        band_hulls = hulls_gdf[hulls_gdf['band'] == band].copy()

        if len(band_hulls) < 2:
            logger.info(f"Band {band}: Only {len(band_hulls)} cell(s), skipping")
            return []

        # Check if site_id column exists for same-site filtering
        has_site_id = self.params.site_id_column in band_hulls.columns

        if self.params.filter_same_site and not has_site_id:
            logger.warning(
                f"Same-site filtering enabled but '{self.params.site_id_column}' column not found. "
                "Same-site pairs will not be filtered."
            )

        conflicts = []

        # 1. Check exact PCI collisions
        pci_conflicts = self._check_pci_group_conflicts(
            band_hulls, band, has_site_id,
            conflict_type='exact',
            group_by_func=lambda pci: int(pci),
            overlap_threshold=self.params.overlap_threshold
        )
        conflicts.extend(pci_conflicts)

        # 2. Check mod 3 conflicts (PSS interference) if enabled
        if self.params.check_mod3_conflicts:
            mod3_conflicts = self._check_pci_group_conflicts(
                band_hulls, band, has_site_id,
                conflict_type='mod3',
                group_by_func=lambda pci: int(pci) % 3,
                overlap_threshold=self.params.mod3_overlap_threshold,
                exclude_exact_pci=True  # Don't double-count exact matches
            )
            conflicts.extend(mod3_conflicts)

        # 3. Check mod 30 conflicts (RS interference) if enabled
        if self.params.check_mod30_conflicts:
            mod30_conflicts = self._check_pci_group_conflicts(
                band_hulls, band, has_site_id,
                conflict_type='mod30',
                group_by_func=lambda pci: int(pci) % 30,
                overlap_threshold=self.params.mod30_overlap_threshold,
                exclude_exact_pci=True  # Don't double-count exact matches
            )
            conflicts.extend(mod30_conflicts)

        return conflicts

    def _check_pci_group_conflicts(
        self,
        band_hulls: gpd.GeoDataFrame,
        band: str,
        has_site_id: bool,
        conflict_type: str,
        group_by_func,
        overlap_threshold: float,
        exclude_exact_pci: bool = False
    ) -> List[Dict[str, Any]]:
        """Check for conflicts within PCI groups (exact, mod3, or mod30)."""
        conflicts = []

        # Add grouping column
        band_hulls = band_hulls.copy()
        band_hulls['_pci_group'] = band_hulls['pci'].apply(group_by_func)

        # Group cells by PCI group value
        pci_groups = band_hulls.groupby('_pci_group')

        for group_val, group in pci_groups:
            if len(group) < 2:
                continue

            # Use spatial indexing
            cells = group.reset_index(drop=True)
            geoms = cells['geometry'].tolist()
            spatial_index = STRtree(geoms)
            checked_pairs = set()

            for idx, cell1 in cells.iterrows():
                geom1 = cell1['geometry']
                intersecting_indices = spatial_index.query(geom1, predicate='intersects')

                for other_idx in intersecting_indices:
                    if idx == other_idx:
                        continue

                    pair = tuple(sorted([idx, other_idx]))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    cell2 = cells.iloc[other_idx]

                    # Skip same-site pairs if filtering enabled
                    if self.params.filter_same_site and has_site_id:
                        site1 = cell1.get(self.params.site_id_column)
                        site2 = cell2.get(self.params.site_id_column)
                        if site1 and site2 and site1 == site2:
                            continue

                    # For mod conflicts, skip if exact PCI match (already counted)
                    if exclude_exact_pci and cell1['pci'] == cell2['pci']:
                        continue

                    # Calculate overlap with appropriate threshold
                    conflict_info = self._check_overlap(
                        cell1, cell2, band,
                        pci1=int(cell1['pci']),
                        pci2=int(cell2['pci']),
                        conflict_type=conflict_type,
                        overlap_threshold=overlap_threshold
                    )
                    if conflict_info:
                        conflicts.append(conflict_info)

        return conflicts

    def _check_overlap(
        self,
        cell1: pd.Series,
        cell2: pd.Series,
        band: str,
        pci1: int,
        pci2: int,
        conflict_type: str = 'exact',
        overlap_threshold: float = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if two cells have overlapping coverage.

        Args:
            cell1: First cell (GeoSeries row)
            cell2: Second cell (GeoSeries row)
            band: Frequency band
            pci1: PCI of first cell
            pci2: PCI of second cell
            conflict_type: 'exact', 'mod3', or 'mod30'
            overlap_threshold: Override threshold for this check

        Returns:
            Conflict dictionary if overlap exceeds threshold, None otherwise
        """
        geom1 = cell1['geometry']
        geom2 = cell2['geometry']

        # Validate geometries
        if geom1.is_empty or geom2.is_empty:
            return None
        if not geom1.is_valid:
            geom1 = geom1.buffer(0)
        if not geom2.is_valid:
            geom2 = geom2.buffer(0)

        # Check if geometries intersect (fast check in original CRS)
        if not geom1.intersects(geom2):
            return None

        # Use input CRS from GeoDataFrame (not hardcoded)
        input_crs = self.input_crs or 'EPSG:4326'

        # Create temporary GeoDataFrames for CRS transformation
        gdf1 = gpd.GeoDataFrame([cell1], geometry='geometry', crs=input_crs)
        gdf2 = gpd.GeoDataFrame([cell2], geometry='geometry', crs=input_crs)

        # Determine target CRS for accurate area/distance calculations
        target_crs = self.target_crs
        if target_crs is None:
            # Estimate UTM zone from midpoint of the two cells
            midpoint_x = (geom1.centroid.x + geom2.centroid.x) / 2
            midpoint_y = (geom1.centroid.y + geom2.centroid.y) / 2
            utm_zone = int((midpoint_x + 180) / 6) + 1
            epsg_code = 32600 + utm_zone if midpoint_y >= 0 else 32700 + utm_zone
            target_crs = f'EPSG:{epsg_code}'

        # Transform to projected CRS for accurate area/distance calculations
        gdf1_proj = gdf1.to_crs(target_crs)
        gdf2_proj = gdf2.to_crs(target_crs)

        geom1_proj = gdf1_proj.iloc[0]['geometry']
        geom2_proj = gdf2_proj.iloc[0]['geometry']

        # Calculate overlap area in projected CRS
        overlap_geom_proj = geom1_proj.intersection(geom2_proj)
        overlap_area_m2 = overlap_geom_proj.area  # m^2 in projected CRS
        overlap_area_km2 = overlap_area_m2 / 1_000_000  # Convert to km^2

        # Calculate overlap percentage relative to smaller cell
        area1_m2 = geom1_proj.area
        area2_m2 = geom2_proj.area
        smaller_area_m2 = min(area1_m2, area2_m2)
        overlap_pct = (overlap_area_m2 / smaller_area_m2) * 100 if smaller_area_m2 > 0 else 0

        # Use provided threshold or default
        threshold = overlap_threshold if overlap_threshold is not None else self.params.overlap_threshold

        # Check if overlap exceeds threshold
        if overlap_pct < (threshold * 100) and overlap_area_km2 < self.params.min_overlap_area_km2:
            return None

        # Calculate distance between cell centroids
        centroid1_proj = geom1_proj.centroid
        centroid2_proj = geom2_proj.centroid
        distance_m = centroid1_proj.distance(centroid2_proj)
        distance_km = distance_m / 1000

        # Generate appropriate log message and detector type
        if conflict_type == 'exact':
            detector_type = 'PCI_COLLISION'
            conflict_desc = f"PCI {pci1}"
            logger.info(
                f"PCI collision found: {cell1['cell_name']} <-> {cell2['cell_name']} "
                f"(PCI {pci1}, Band {band}, {overlap_pct:.1f}% overlap)"
            )
        elif conflict_type == 'mod3':
            detector_type = 'PCI_MOD3_CONFLICT'
            conflict_desc = f"PCI mod 3 = {pci1 % 3} (PCIs: {pci1}, {pci2})"
            logger.info(
                f"PCI mod 3 conflict found: {cell1['cell_name']} (PCI {pci1}) <-> "
                f"{cell2['cell_name']} (PCI {pci2}), Band {band}, {overlap_pct:.1f}% overlap"
            )
        else:  # mod30
            detector_type = 'PCI_MOD30_CONFLICT'
            conflict_desc = f"PCI mod 30 = {pci1 % 30} (PCIs: {pci1}, {pci2})"
            logger.info(
                f"PCI mod 30 conflict found: {cell1['cell_name']} (PCI {pci1}) <-> "
                f"{cell2['cell_name']} (PCI {pci2}), Band {band}, {overlap_pct:.1f}% overlap"
            )

        # Determine severity (mod conflicts are generally less severe than exact)
        severity = self._calculate_severity(overlap_pct, distance_km, conflict_type)

        return {
            'detector': detector_type,
            'conflict_type': conflict_type,
            'band': band,
            'pci1': pci1,
            'pci2': pci2,
            'cell1_name': cell1['cell_name'],
            'cell2_name': cell2['cell_name'],
            'overlap_percentage': round(overlap_pct, 2),
            'overlap_area_km2': round(overlap_area_km2, 6),
            'distance_km': round(distance_km, 2),
            'severity': severity,
            'recommendation': self._generate_recommendation(
                cell1['cell_name'],
                cell2['cell_name'],
                pci1, pci2,
                conflict_type,
                overlap_pct
            )
        }

    def _calculate_severity(
        self, overlap_pct: float, distance_km: float, conflict_type: str = 'exact'
    ) -> str:
        """
        Calculate conflict severity based on overlap, distance, and conflict type.

        Args:
            overlap_pct: Overlap percentage
            distance_km: Distance between cells in km
            conflict_type: 'exact', 'mod3', or 'mod30'

        Returns:
            Severity level: 'critical', 'high', 'medium', 'low'
        """
        thresholds = self.params.severity_thresholds
        critical = thresholds.get('critical', {'overlap_pct': 50, 'distance_km': 2})
        high = thresholds.get('high', {'overlap_pct': 30, 'distance_km': 3})
        medium = thresholds.get('medium', {'overlap_pct': 20, 'distance_km': 5})

        # Mod conflicts are generally less severe than exact PCI collisions
        # Downgrade severity by one level for mod 3, two levels for mod 30
        base_severity = 'low'
        if overlap_pct > critical['overlap_pct'] and distance_km < critical['distance_km']:
            base_severity = 'critical'
        elif overlap_pct > high['overlap_pct'] and distance_km < high['distance_km']:
            base_severity = 'high'
        elif overlap_pct > medium['overlap_pct'] or distance_km < medium['distance_km']:
            base_severity = 'medium'

        # Downgrade for mod conflicts
        if conflict_type == 'mod3':
            severity_order = ['low', 'medium', 'high', 'critical']
            idx = severity_order.index(base_severity)
            return severity_order[max(0, idx - 1)]  # Downgrade by 1
        elif conflict_type == 'mod30':
            severity_order = ['low', 'medium', 'high', 'critical']
            idx = severity_order.index(base_severity)
            return severity_order[max(0, idx - 2)]  # Downgrade by 2

        return base_severity

    def _generate_recommendation(
        self, cell1: str, cell2: str, pci1: int, pci2: int,
        conflict_type: str, overlap_pct: float
    ) -> str:
        """Generate recommendation text for resolving the conflict."""
        if conflict_type == 'exact':
            return (
                f"PCI COLLISION: Change PCI for either {cell1} or {cell2} "
                f"(both use PCI {pci1}). Coverage overlap is {overlap_pct:.1f}%, "
                f"causing cell identity confusion during initial access. "
                f"Assign a PCI not used by neighboring cells."
            )
        elif conflict_type == 'mod3':
            return (
                f"PSS INTERFERENCE: {cell1} (PCI {pci1}) and {cell2} (PCI {pci2}) "
                f"have same PCI mod 3 value ({pci1 % 3}), causing Primary Sync Signal interference "
                f"with {overlap_pct:.1f}% coverage overlap. "
                f"Consider changing one PCI to have different mod 3 value for better cell search."
            )
        else:  # mod30
            return (
                f"RS INTERFERENCE: {cell1} (PCI {pci1}) and {cell2} (PCI {pci2}) "
                f"have same PCI mod 30 value ({pci1 % 30}), causing Reference Signal interference "
                f"with {overlap_pct:.1f}% coverage overlap. "
                f"Consider changing one PCI for better RSRP measurement accuracy."
            )


def detect_pci_conflicts(
    hulls_gdf: gpd.GeoDataFrame,
    params: Optional[PCIConflictParams] = None,
) -> pd.DataFrame:
    """
    Convenience function to detect PCI conflicts.

    Args:
        hulls_gdf: GeoDataFrame with cell hulls (cell_name, geometry, band, pci)
        params: Optional detection parameters

    Returns:
        DataFrame with PCI conflict results
    """
    detector = PCIConflictDetector(params)
    conflicts = detector.detect(hulls_gdf)

    if not conflicts:
        return pd.DataFrame()

    return pd.DataFrame(conflicts)
