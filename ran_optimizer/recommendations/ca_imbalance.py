"""
CA Imbalance Detector - Detects carrier aggregation imbalances between coverage and capacity bands.

In LTE carrier aggregation, the coverage band (e.g., L800) provides the anchor while the
capacity band (e.g., L1800) provides additional capacity. For optimal CA performance,
each site/sector with the coverage band should also have adequate capacity band coverage.
Sites with coverage band but insufficient capacity band suffer from capacity limitations.

IMPORTANT: Coverage Hulls Data Source
The convex hulls used in this detector should be generated from ACTUAL UE (User Equipment)
measurement points collected via triangulated trace data, NOT geometric approximations
around cell sites. This makes them accurate representations of real-world coverage
based on customer device measurements.
"""

import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CAPairConfig:
    """Configuration for a carrier aggregation band pair."""
    name: str  # e.g., 'L800-L1800'
    coverage_band: str  # e.g., 'L800'
    capacity_band: str  # e.g., 'L1800'
    coverage_threshold: float = 0.70  # 70% minimum coverage ratio


@dataclass
class CAImbalanceParams:
    """Parameters for CA imbalance detection."""
    ca_pairs: List[CAPairConfig]  # Must be provided via config
    cell_name_pattern: str  # Must be provided via config (e.g., r'(\w+)[A-Z]+(\d)')
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.30,
        'high': 0.50,
        'medium': 0.60,
        'warning': 0.70,
        'low': 1.00
    })
    environment_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'urban': {'coverage_threshold': 0.85},
        'suburban': {'coverage_threshold': 0.70},
        'rural': {'coverage_threshold': 0.60}
    })
    use_environment_thresholds: bool = False
    # Data quality thresholds
    min_sample_count: int = 100  # Minimum UE measurement points for hull confidence
    sample_count_column: str = 'sample_count'  # Column name for sample count in hulls_gdf
    # Temporal alignment thresholds
    timestamp_column: str = 'data_timestamp'  # Column name for hull data timestamp
    max_data_age_days: int = 30  # Maximum age of hull data in days
    max_temporal_gap_days: int = 7  # Maximum gap between coverage and capacity hull timestamps

    @classmethod
    def from_config(cls, config_path: str) -> 'CAImbalanceParams':
        """Load parameters from config file.

        Args:
            config_path: Path to JSON config file (required)

        Returns:
            CAImbalanceParams instance

        Raises:
            ValueError: If config_path is None or required fields are missing
        """
        if config_path is None:
            raise ValueError("config_path is required - CA imbalance detection requires network-specific configuration")

        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Require ca_pairs in config
        if 'ca_pairs' not in config or not config['ca_pairs']:
            raise ValueError("ca_pairs must be specified in config")

        # Require cell_name_pattern in config
        if 'cell_name_pattern' not in config:
            raise ValueError("cell_name_pattern must be specified in config")

        ca_pairs = []
        for pair_config in config['ca_pairs']:
            ca_pairs.append(CAPairConfig(
                name=pair_config['name'],
                coverage_band=pair_config['coverage_band'],
                capacity_band=pair_config['capacity_band'],
                coverage_threshold=pair_config.get('coverage_threshold', 0.70)
            ))

        # Get default severity/environment thresholds
        default_severity = {
            'critical': 0.30, 'high': 0.50, 'medium': 0.60, 'warning': 0.70, 'low': 1.00
        }
        default_env = {
            'urban': {'coverage_threshold': 0.85},
            'suburban': {'coverage_threshold': 0.70},
            'rural': {'coverage_threshold': 0.60}
        }

        return cls(
            ca_pairs=ca_pairs,
            cell_name_pattern=config['cell_name_pattern'],
            severity_thresholds=config.get('severity_thresholds', default_severity),
            environment_thresholds=config.get('environment_thresholds', default_env),
            use_environment_thresholds=config.get('use_environment_thresholds', False),
            min_sample_count=config.get('min_sample_count', 100),
            sample_count_column=config.get('sample_count_column', 'sample_count'),
            timestamp_column=config.get('timestamp_column', 'data_timestamp'),
            max_data_age_days=config.get('max_data_age_days', 30),
            max_temporal_gap_days=config.get('max_temporal_gap_days', 7),
        )


class CAImbalanceDetector:
    """Detects CA imbalance using per-cell/per-site analysis."""

    def __init__(
        self,
        params: CAImbalanceParams,
        target_crs: Optional[str] = None
    ):
        """
        Initialize the CA Imbalance Detector.

        Args:
            params: Detection parameters (required - use CAImbalanceParams.from_config())
            target_crs: Target CRS for area calculations (auto-detected from input if None)
        """
        self.params = params
        self.target_crs = target_crs

        logger.info(
            "CA Imbalance detector initialized",
            ca_pairs=[p.name for p in self.params.ca_pairs],
            use_environment_thresholds=self.params.use_environment_thresholds,
        )

    def detect(
        self,
        hulls_gdf: gpd.GeoDataFrame,
        site_environments: Optional[Dict[str, str]] = None,
        coverage_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect CA imbalance across all configured CA pairs.

        For each CA pair, compares coverage and capacity band areas to identify imbalances.
        Flags as CA imbalance if capacity coverage area < threshold of coverage area.

        Args:
            hulls_gdf: GeoDataFrame with columns: cell_name, geometry, band
            site_environments: Optional dict mapping site_id to environment
            coverage_threshold: Optional override for coverage threshold

        Returns:
            List of imbalance issue dictionaries for all CA pairs
        """
        # Validate input
        hulls_gdf = self._validate_input(hulls_gdf)

        logger.info(f"Starting CA imbalance detection for {len(self.params.ca_pairs)} CA pair(s)")

        all_issues = []

        # Iterate through each configured CA pair
        for ca_pair in self.params.ca_pairs:
            threshold = coverage_threshold if coverage_threshold is not None else ca_pair.coverage_threshold

            logger.info(
                f"Checking CA pair: {ca_pair.name} "
                f"({ca_pair.coverage_band} -> {ca_pair.capacity_band}, threshold={threshold*100}%)"
            )

            # Detect issues for this specific CA pair
            pair_issues = self._detect_for_ca_pair(
                hulls_gdf,
                ca_pair,
                threshold,
                site_environments
            )

            all_issues.extend(pair_issues)

        logger.info(f"Total CA imbalance issues across all pairs: {len(all_issues)}")
        return all_issues

    def _validate_input(self, hulls_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate input GeoDataFrame."""
        initial_count = len(hulls_gdf)

        # Check required columns
        required_columns = ['cell_name', 'geometry', 'band']
        missing_columns = [col for col in required_columns if col not in hulls_gdf.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for null geometries
        null_geom_count = hulls_gdf['geometry'].isna().sum()
        if null_geom_count > 0:
            logger.warning(f"Found {null_geom_count} null geometries. These will be skipped.")
            hulls_gdf = hulls_gdf[hulls_gdf['geometry'].notna()].copy()

        if len(hulls_gdf) == 0:
            raise ValueError("GeoDataFrame is empty after filtering null geometries")

        # Check for invalid geometry types (vectorized)
        valid_geom_types = {'Polygon', 'MultiPolygon'}
        invalid_geom_mask = ~hulls_gdf['geometry'].geom_type.isin(valid_geom_types)
        invalid_geom_count = invalid_geom_mask.sum()

        if invalid_geom_count > 0:
            logger.warning(f"Found {invalid_geom_count} invalid geometry types. These will be skipped.")
            hulls_gdf = hulls_gdf[~invalid_geom_mask].copy()

        if len(hulls_gdf) == 0:
            raise ValueError("GeoDataFrame is empty after filtering invalid geometry types")

        # Check for sample count (data confidence) if column exists
        sample_col = self.params.sample_count_column
        if sample_col in hulls_gdf.columns:
            min_samples = self.params.min_sample_count
            low_confidence_mask = hulls_gdf[sample_col] < min_samples
            low_confidence_count = low_confidence_mask.sum()

            if low_confidence_count > 0:
                low_conf_cells = hulls_gdf.loc[low_confidence_mask, 'cell_name'].head(5).tolist()
                logger.warning(
                    f"Found {low_confidence_count} hulls with <{min_samples} UE samples (low confidence)",
                    sample_cells=low_conf_cells,
                )

        # Check temporal alignment if timestamp column exists
        ts_col = self.params.timestamp_column
        if ts_col in hulls_gdf.columns:
            try:
                timestamps = pd.to_datetime(hulls_gdf[ts_col], errors='coerce')
                valid_timestamps = timestamps.dropna()

                if len(valid_timestamps) > 0:
                    now = datetime.now()
                    oldest = valid_timestamps.min()
                    newest = valid_timestamps.max()
                    data_age_days = (now - newest).days
                    data_span_days = (newest - oldest).days

                    # Warn if data is too old
                    if data_age_days > self.params.max_data_age_days:
                        logger.warning(
                            f"Hull data is {data_age_days} days old (max recommended: {self.params.max_data_age_days})",
                            newest_data=newest.isoformat(),
                        )

                    # Warn if data spans too long a period (potential inconsistency)
                    if data_span_days > self.params.max_temporal_gap_days:
                        logger.warning(
                            f"Hull data spans {data_span_days} days (max recommended: {self.params.max_temporal_gap_days}). "
                            "Consider using data from a narrower time window for consistency.",
                            oldest_data=oldest.isoformat(),
                            newest_data=newest.isoformat(),
                        )
            except Exception as e:
                logger.warning(f"Could not parse timestamp column '{ts_col}': {e}")

        valid_count = len(hulls_gdf)
        logger.info(
            f"Input validation: {valid_count}/{initial_count} cells valid",
            filtered_out=initial_count - valid_count,
        )

        return hulls_gdf

    def _detect_for_ca_pair(
        self,
        hulls_gdf: gpd.GeoDataFrame,
        ca_pair: CAPairConfig,
        threshold: float,
        site_environments: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Detect CA imbalance for a specific coverage/capacity band pair."""

        # Filter to this specific CA pair's bands
        valid_cells = hulls_gdf[
            hulls_gdf['band'].isin([ca_pair.coverage_band, ca_pair.capacity_band])
        ].copy()

        logger.info(f"Analyzing {len(valid_cells)} cells ({ca_pair.coverage_band} + {ca_pair.capacity_band})")

        if len(valid_cells) == 0:
            logger.warning(f"No {ca_pair.coverage_band} or {ca_pair.capacity_band} cells found")
            return []

        # Parse cell names to extract site_id and sector
        pattern = self.params.cell_name_pattern
        extracted = valid_cells['cell_name'].str.extract(pattern, expand=True)
        extracted.columns = ['site_id', 'sector']

        # Create site_sector column (e.g., "CK001_1")
        valid_cells['site_sector'] = extracted['site_id'] + '_' + extracted['sector']

        # Log cells where parsing failed before filtering them out
        parse_failed_mask = valid_cells['site_sector'].isna()
        parse_failed_count = parse_failed_mask.sum()

        if parse_failed_count > 0:
            failed_cell_names = valid_cells.loc[parse_failed_mask, 'cell_name'].head(10).tolist()
            logger.warning(
                f"Failed to parse {parse_failed_count} cell names with pattern '{pattern}'",
                sample_failed_cells=failed_cell_names,
            )

        # Filter out cells where parsing failed
        valid_cells = valid_cells[~parse_failed_mask].copy()

        logger.info(f"Successfully parsed {len(valid_cells)} cell names")

        # Determine target CRS for accurate area calculations
        target_crs = self.target_crs
        if target_crs is None:
            # Auto-detect: use input CRS if projected, else estimate UTM zone
            if valid_cells.crs and valid_cells.crs.is_projected:
                target_crs = valid_cells.crs
                logger.info(f"Using input projected CRS: {target_crs}")
            else:
                # Estimate UTM zone from centroid
                centroid = valid_cells.unary_union.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                hemisphere = 'north' if centroid.y >= 0 else 'south'
                epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
                target_crs = f'EPSG:{epsg_code}'
                logger.info(f"Auto-detected UTM zone: {target_crs}")

        # Transform to projected CRS for accurate area calculations
        valid_cells_projected = valid_cells.to_crs(target_crs)

        # Calculate areas in km^2 using projected CRS
        valid_cells_projected['area_km2'] = valid_cells_projected['geometry'].area / 1_000_000

        logger.info(f"Calculated areas using projected CRS ({target_crs})")

        # Group by site_sector to find pairs with both coverage and capacity bands
        site_sector_groups = valid_cells_projected.groupby('site_sector')

        issues = []
        pairs_with_both_bands = 0

        for site_sector, group in site_sector_groups:
            bands_present = set(group['band'].unique())

            # Only check site/sectors that have BOTH coverage AND capacity bands
            has_coverage = ca_pair.coverage_band in bands_present
            has_capacity = ca_pair.capacity_band in bands_present

            if not (has_coverage and has_capacity):
                continue

            pairs_with_both_bands += 1

            # Get cells for each band (may have multiple cells per band in advanced deployments)
            coverage_cells = group[group['band'] == ca_pair.coverage_band]
            capacity_cells = group[group['band'] == ca_pair.capacity_band]

            # Handle multiple cells per band - use the one with largest area (most representative)
            if len(coverage_cells) > 1:
                logger.info(
                    f"Multiple {ca_pair.coverage_band} cells at {site_sector}, using largest by area",
                    cell_count=len(coverage_cells),
                )
                coverage_cell = coverage_cells.loc[coverage_cells['area_km2'].idxmax()]
            else:
                coverage_cell = coverage_cells.iloc[0]

            if len(capacity_cells) > 1:
                logger.info(
                    f"Multiple {ca_pair.capacity_band} cells at {site_sector}, using largest by area",
                    cell_count=len(capacity_cells),
                )
                capacity_cell = capacity_cells.loc[capacity_cells['area_km2'].idxmax()]
            else:
                capacity_cell = capacity_cells.iloc[0]

            # Get geometries and areas
            coverage_geom = coverage_cell['geometry']
            capacity_geom = capacity_cell['geometry']
            coverage_area_km2 = coverage_cell['area_km2']
            capacity_area_km2 = capacity_cell['area_km2']

            # Calculate intersection area - this is what matters for CA capability
            # CA requires UE to be in BOTH coverage and capacity cell footprints
            if coverage_geom.is_valid and capacity_geom.is_valid:
                intersection = coverage_geom.intersection(capacity_geom)
                intersection_area_km2 = intersection.area / 1_000_000 if not intersection.is_empty else 0
            else:
                logger.warning(f"Invalid geometry for {site_sector}, skipping intersection calculation")
                intersection_area_km2 = 0

            # Coverage ratio = intersection / coverage area (what % of coverage band has CA capability)
            coverage_ratio = intersection_area_km2 / coverage_area_km2 if coverage_area_km2 > 0 else 0

            # Determine threshold (environment-aware if enabled)
            site_id = site_sector.split('_')[0]
            effective_threshold = threshold

            if self.params.use_environment_thresholds and site_environments and site_id in site_environments:
                environment = site_environments[site_id]
                if environment in self.params.environment_thresholds:
                    effective_threshold = self.params.environment_thresholds[environment]['coverage_threshold']

            # Check if capacity coverage is insufficient
            if coverage_ratio < effective_threshold:
                issue = {
                    'detector': 'CA_IMBALANCE',
                    'ca_pair': ca_pair.name,
                    'coverage_band': ca_pair.coverage_band,
                    'capacity_band': ca_pair.capacity_band,
                    'issue_type': 'insufficient_capacity_coverage',
                    'severity': self._calculate_severity(coverage_ratio),
                    'site_sector': site_sector,
                    'coverage_cell_name': coverage_cell['cell_name'],
                    'capacity_cell_name': capacity_cell['cell_name'],
                    'coverage_area_km2': round(coverage_area_km2, 2),
                    'capacity_area_km2': round(capacity_area_km2, 2),
                    'intersection_area_km2': round(intersection_area_km2, 2),
                    'coverage_ratio': round(coverage_ratio, 3),
                    'coverage_percentage': round(coverage_ratio * 100, 1),
                    'threshold_percentage': round(effective_threshold * 100, 1),
                    'recommendation': self._generate_recommendation(
                        coverage_cell['cell_name'],
                        capacity_cell['cell_name'],
                        ca_pair.coverage_band,
                        ca_pair.capacity_band,
                        coverage_ratio,
                        effective_threshold
                    )
                }

                # Add environment info if available
                if self.params.use_environment_thresholds and site_environments and site_id in site_environments:
                    issue['environment'] = site_environments[site_id]

                issues.append(issue)

        logger.info(
            f"Found {pairs_with_both_bands} site/sectors with both bands, "
            f"{len(issues)} with insufficient capacity coverage"
        )

        return issues

    def _calculate_severity(self, coverage_ratio: float) -> str:
        """Calculate severity based on coverage ratio."""
        thresholds = self.params.severity_thresholds

        if coverage_ratio < thresholds.get('critical', 0.30):
            return 'critical'
        elif coverage_ratio < thresholds.get('high', 0.50):
            return 'high'
        elif coverage_ratio < thresholds.get('medium', 0.60):
            return 'medium'
        elif coverage_ratio < thresholds.get('warning', 0.70):
            return 'warning'
        else:
            return 'low'

    def _generate_recommendation(
        self,
        coverage_cell: str,
        capacity_cell: str,
        coverage_band: str,
        capacity_band: str,
        coverage_ratio: float,
        threshold: float
    ) -> str:
        """Generate recommendation for improving capacity band coverage overlap."""
        overlap_pct = coverage_ratio * 100
        threshold_pct = threshold * 100
        gap_pct = 100 - overlap_pct

        return (
            f"{capacity_band} cell '{capacity_cell}' overlaps only {overlap_pct:.1f}% of "
            f"{coverage_band} cell '{coverage_cell}' footprint (threshold: {threshold_pct:.0f}%). "
            f"This means {gap_pct:.1f}% of {coverage_band} coverage area cannot use carrier aggregation "
            f"with {capacity_band}. "
            f"Recommendation: Adjust {capacity_band} antenna tilt/azimuth to better align with "
            f"{coverage_band} coverage, or increase {capacity_band} transmit power to extend footprint."
        )


def detect_ca_imbalance(
    hulls_gdf: gpd.GeoDataFrame,
    params: CAImbalanceParams,
    site_environments: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to detect CA imbalance.

    Args:
        hulls_gdf: GeoDataFrame with cell hulls (cell_name, geometry, band)
        params: Detection parameters (required - use CAImbalanceParams.from_config())
        site_environments: Optional dict mapping site_id to environment

    Returns:
        DataFrame with CA imbalance results

    Raises:
        ValueError: If params is not provided (network-specific config required)
    """
    if params is None:
        raise ValueError("params is required - CA imbalance detection requires network-specific configuration")

    detector = CAImbalanceDetector(params)
    issues = detector.detect(hulls_gdf, site_environments)

    if not issues:
        return pd.DataFrame()

    return pd.DataFrame(issues)
