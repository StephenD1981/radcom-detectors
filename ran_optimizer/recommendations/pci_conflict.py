"""
PCI Conflict Detector - Detects cells with same PCI and overlapping coverage areas.

Physical Cell ID (PCI) conflicts occur when cells on the same frequency band have:
1. The same PCI value
2. Overlapping coverage areas (hull overlap > threshold)

This causes mobile devices to confuse the two cells, leading to handover failures.
"""

import geopandas as gpd
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from shapely.strtree import STRtree

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


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
                severity_thresholds=config.get('severity_thresholds', cls().severity_thresholds)
            )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return cls()


class PCIConflictDetector:
    """Detects PCI conflicts using per-band convex hull overlap analysis."""

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

        logger.info(
            "PCI Conflict detector initialized",
            overlap_threshold=self.params.overlap_threshold,
            min_overlap_area_km2=self.params.min_overlap_area_km2,
        )

    def detect(self, hulls_gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
        """
        Detect PCI conflicts across all bands.

        Args:
            hulls_gdf: GeoDataFrame with columns: cell_name, geometry, band, pci

        Returns:
            List of conflict dictionaries with cell pairs and overlap details
        """
        logger.info("Starting PCI conflict detection")

        # Validate required columns
        required_cols = ['cell_name', 'geometry', 'band', 'pci']
        missing = [c for c in required_cols if c not in hulls_gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter out cells with missing band or PCI
        valid_hulls = hulls_gdf[
            hulls_gdf['band'].notna() & hulls_gdf['pci'].notna()
        ].copy()

        logger.info(f"Analyzing {len(valid_hulls)} cells with band and PCI info")

        conflicts = []

        # Process each band separately
        for band in valid_hulls['band'].unique():
            logger.info(f"Checking PCI conflicts for band {band}")
            band_conflicts = self._detect_conflicts_per_band(valid_hulls, band)
            conflicts.extend(band_conflicts)

        logger.info(f"Detected {len(conflicts)} PCI conflicts")
        return conflicts

    def _detect_conflicts_per_band(
        self, hulls_gdf: gpd.GeoDataFrame, band: str
    ) -> List[Dict[str, Any]]:
        """
        Detect PCI conflicts within a single frequency band.

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

        conflicts = []

        # Group cells by PCI value
        pci_groups = band_hulls.groupby('pci')

        for pci, group in pci_groups:
            if len(group) < 2:
                # Only 1 cell with this PCI on this band - no conflict possible
                continue

            logger.info(f"Band {band}, PCI {pci}: Checking {len(group)} cells")

            # Use spatial indexing to avoid O(n^2) nested loops
            cells = group.reset_index(drop=True)

            # Build spatial index for this PCI group
            geoms = cells['geometry'].tolist()
            spatial_index = STRtree(geoms)

            # Track already-checked pairs to avoid duplicates
            checked_pairs = set()

            # Query spatial index for intersecting geometries
            for idx, cell1 in cells.iterrows():
                geom1 = cell1['geometry']

                # Find all geometries that intersect with this cell
                intersecting_indices = spatial_index.query(geom1, predicate='intersects')

                for other_idx in intersecting_indices:
                    # Skip self-intersection
                    if idx == other_idx:
                        continue

                    # Skip if we've already checked this pair
                    pair = tuple(sorted([idx, other_idx]))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    cell2 = cells.iloc[other_idx]

                    # Calculate overlap
                    conflict_info = self._check_overlap(cell1, cell2, band, pci)
                    if conflict_info:
                        conflicts.append(conflict_info)

        return conflicts

    def _check_overlap(
        self,
        cell1: pd.Series,
        cell2: pd.Series,
        band: str,
        pci: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check if two cells with same PCI have overlapping coverage.

        Args:
            cell1: First cell (GeoSeries row)
            cell2: Second cell (GeoSeries row)
            band: Frequency band
            pci: PCI value

        Returns:
            Conflict dictionary if overlap exceeds threshold, None otherwise
        """
        geom1 = cell1['geometry']
        geom2 = cell2['geometry']

        # Check if geometries intersect (fast check in original CRS)
        if not geom1.intersects(geom2):
            return None

        # Create temporary GeoDataFrames for CRS transformation
        gdf1 = gpd.GeoDataFrame([cell1], geometry='geometry', crs='EPSG:4326')
        gdf2 = gpd.GeoDataFrame([cell2], geometry='geometry', crs='EPSG:4326')

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

        # Check if overlap exceeds threshold
        if overlap_pct < (self.params.overlap_threshold * 100) and \
           overlap_area_km2 < self.params.min_overlap_area_km2:
            return None

        # Calculate distance between cell centroids
        centroid1_proj = geom1_proj.centroid
        centroid2_proj = geom2_proj.centroid
        distance_m = centroid1_proj.distance(centroid2_proj)
        distance_km = distance_m / 1000

        logger.info(
            f"PCI conflict found: {cell1['cell_name']} <-> {cell2['cell_name']} "
            f"(PCI {pci}, Band {band}, {overlap_pct:.1f}% overlap)"
        )

        return {
            'detector': 'PCI_CONFLICT',
            'band': band,
            'pci': int(pci),
            'cell1_name': cell1['cell_name'],
            'cell2_name': cell2['cell_name'],
            'overlap_percentage': round(overlap_pct, 2),
            'overlap_area_km2': round(overlap_area_km2, 6),
            'distance_km': round(distance_km, 2),
            'severity': self._calculate_severity(overlap_pct, distance_km),
            'recommendation': self._generate_recommendation(
                cell1['cell_name'],
                cell2['cell_name'],
                pci,
                overlap_pct
            )
        }

    def _calculate_severity(self, overlap_pct: float, distance_km: float) -> str:
        """
        Calculate conflict severity based on overlap and distance.

        Args:
            overlap_pct: Overlap percentage
            distance_km: Distance between cells in km

        Returns:
            Severity level: 'critical', 'high', 'medium', 'low'
        """
        thresholds = self.params.severity_thresholds
        critical = thresholds['critical']
        high = thresholds['high']
        medium = thresholds['medium']

        if overlap_pct > critical['overlap_pct'] and distance_km < critical['distance_km']:
            return 'critical'
        elif overlap_pct > high['overlap_pct'] or distance_km < high['distance_km']:
            return 'high'
        elif overlap_pct > medium['overlap_pct'] or distance_km < medium['distance_km']:
            return 'medium'
        else:
            return 'low'

    def _generate_recommendation(
        self, cell1: str, cell2: str, pci: int, overlap_pct: float
    ) -> str:
        """Generate recommendation text for resolving the conflict."""
        return (
            f"Change PCI for either {cell1} or {cell2} (currently both use PCI {pci}). "
            f"Coverage overlap is {overlap_pct:.1f}%, causing handover confusion. "
            f"Recommended: Assign different PCI values from neighboring cells."
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
