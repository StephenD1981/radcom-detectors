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
import re
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
        'low': 1.00
    })
    environment_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'urban': {'coverage_threshold': 0.85},
        'suburban': {'coverage_threshold': 0.70},
        'rural': {'coverage_threshold': 0.60}
    })
    use_environment_thresholds: bool = False

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
            'critical': 0.30, 'high': 0.50, 'medium': 0.60, 'low': 1.00
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

        # Filter out cells where parsing failed
        valid_cells = valid_cells[valid_cells['site_sector'].notna()].copy()

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

            # Get the coverage and capacity cells
            coverage_cell = group[group['band'] == ca_pair.coverage_band].iloc[0]
            capacity_cell = group[group['band'] == ca_pair.capacity_band].iloc[0]

            # Get accurate areas from projected CRS calculation
            coverage_area_km2 = coverage_cell['area_km2']
            capacity_area_km2 = capacity_cell['area_km2']

            # Calculate coverage ratio (capacity / coverage)
            coverage_ratio = capacity_area_km2 / coverage_area_km2 if coverage_area_km2 > 0 else 0

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

        if coverage_ratio < thresholds['critical']:
            return 'critical'
        elif coverage_ratio < thresholds['high']:
            return 'high'
        elif coverage_ratio < thresholds['medium']:
            return 'medium'
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
        """Generate recommendation for improving capacity band coverage."""
        coverage_pct = coverage_ratio * 100
        threshold_pct = threshold * 100
        gap_pct = threshold_pct - coverage_pct

        return (
            f"{capacity_band} cell '{capacity_cell}' provides only {coverage_pct:.1f}% coverage of its "
            f"{coverage_band} counterpart '{coverage_cell}' (threshold: {threshold_pct:.0f}%). "
            f"This leaves {gap_pct:.1f}% of {coverage_band} coverage without {capacity_band}, "
            f"preventing carrier aggregation in those areas. "
            f"Recommendation: Increase {capacity_band} transmit power, adjust antenna tilt/azimuth, "
            f"or add additional {capacity_band} capacity to match {coverage_band} footprint."
        )


def detect_ca_imbalance(
    hulls_gdf: gpd.GeoDataFrame,
    params: Optional[CAImbalanceParams] = None,
    site_environments: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to detect CA imbalance.

    Args:
        hulls_gdf: GeoDataFrame with cell hulls (cell_name, geometry, band)
        params: Optional detection parameters
        site_environments: Optional dict mapping site_id to environment

    Returns:
        DataFrame with CA imbalance results
    """
    detector = CAImbalanceDetector(params)
    issues = detector.detect(hulls_gdf, site_environments)

    if not issues:
        return pd.DataFrame()

    return pd.DataFrame(issues)
