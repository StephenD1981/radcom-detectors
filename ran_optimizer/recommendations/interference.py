"""
High Interference Detector.

Detects excessive cell overlap causing poor SINR using geohash-based spatial analysis.
This module contains the core detection algorithm with multiple stages:
filtering, clustering, dominance detection, spatial validation, and scoring.
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import alphashape

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.geohash import kring, decode as decode_geohash

logger = get_logger(__name__)

# Geographic constant for approximate area calculation fallback
KM_PER_DEGREE_LAT = 111.32

# Memory safeguard: maximum records per band to prevent OOM
MAX_RECORDS_PER_BAND = 5_000_000

# Band-specific interference thresholds (dB)
# Different frequency bands have different propagation characteristics:
# - Sub-1GHz: Better propagation, wider coverage, more natural overlap
# - Mid-band: Higher path loss, tighter cells
# - C-band/mmWave: Even higher path loss, very tight cells
BAND_INTERFERENCE_THRESHOLDS = {
    # LTE sub-1GHz bands (coverage layer) - wider coverage, more natural overlap
    'L700': {'max_rsrp_diff': 6.0, 'dominance_diff': 6.0},
    'L800': {'max_rsrp_diff': 6.0, 'dominance_diff': 6.0},
    # LTE mid-band (capacity layer) - tighter cells
    'L1800': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    'L2100': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    'L2600': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    # 5G NR high-band (C-band) - even tighter cells
    'L3500': {'max_rsrp_diff': 4.0, 'dominance_diff': 4.0},
    'L3700': {'max_rsrp_diff': 4.0, 'dominance_diff': 4.0},
    # 5G NR sub-6GHz (same as LTE equivalents for NR refarming)
    'N1': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    'N3': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    'N7': {'max_rsrp_diff': 5.0, 'dominance_diff': 5.0},
    'N28': {'max_rsrp_diff': 6.0, 'dominance_diff': 6.0},
    'N78': {'max_rsrp_diff': 4.0, 'dominance_diff': 4.0},
    'N77': {'max_rsrp_diff': 4.0, 'dominance_diff': 4.0},
}


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class InterferenceParams:
    """Parameters for interference detection."""
    # Grid filtering parameters
    min_filtered_cells_per_grid: int = 4
    min_cell_event_count: int = 2
    perc_grid_events: float = 0.05

    # Dominance detection parameters
    dominant_perc_grid_events: float = 0.3
    dominance_diff: float = 5.0  # dB

    # RSRP similarity threshold
    max_rsrp_diff: float = 5.0  # dB

    # SINR threshold for validation (only flag if SINR is degraded)
    sinr_threshold_db: float = 0.0  # Filter clusters with avg SINR above this

    # Spatial clustering parameters
    k: int = 3  # k-ring neighborhood size
    perc_interference: float = 0.33

    # Clustering algorithm configuration
    clustering_algo: str = 'fixed'  # 'fixed' or 'dynamic-sklearn'
    fixed_width: float = 5.0
    dynamic_width: float = 2.5

    # RSRP quantiles for weighting
    rsrp_min_quantile: float = 0.02
    rsrp_max_quantile: float = 0.98

    # Perceived data multiplier
    perceived_multiplier: int = 3

    # Spatial clustering parameters for polygon output
    hdbscan_min_cluster_size: int = 5
    alpha_shape_alpha: Optional[float] = None  # None = auto
    max_alphashape_points: int = 2000

    # Environment-specific threshold overrides
    environment_overrides: Optional[dict] = None

    # Geohash precision validation (Fix #5)
    # Precision 6: ~1.2km x 0.6km, Precision 7: ~153m x 153m, Precision 8: ~38m x 19m
    expected_geohash_precision: Optional[int] = 7
    geohash_precision_warning: bool = True  # Warn if precision doesn't match expected

    # Memory safeguard: maximum records per band (Fix #11 - now configurable)
    max_records_per_band: int = 5_000_000

    # Data freshness validation (Fix #10)
    max_data_age_days: Optional[int] = 7  # Warn if data older than this, None to disable

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'InterferenceParams':
        """Load parameters from config file or use defaults."""
        if config_path is None:
            config_path = "config/interference_params.json"

        try:
            import json
            from pathlib import Path

            path = Path(config_path)
            if not path.exists():
                logger.warning("config_file_not_found", path=config_path, using_defaults=True)
                return cls()

            with open(path, 'r') as f:
                config = json.load(f)

            # Get default parameters section
            params = config.get('default', config)
            clustering = params.get('clustering', {})
            rsrp_quantiles = params.get('rsrp_quantiles', {})

            polygon_params = params.get('polygon_clustering', {})

            # Validation params
            validation_params = params.get('validation', {})

            return cls(
                min_filtered_cells_per_grid=params.get('min_filtered_cells_per_grid', 4),
                min_cell_event_count=params.get('min_cell_event_count', 2),
                perc_grid_events=params.get('perc_grid_events', 0.05),
                dominant_perc_grid_events=params.get('dominant_perc_grid_events', 0.30),
                dominance_diff=params.get('dominance_diff_db', 5.0),
                max_rsrp_diff=params.get('max_rsrp_diff_db', 5.0),
                sinr_threshold_db=params.get('sinr_threshold_db', 0.0),
                k=params.get('k_ring_steps', 3),
                perc_interference=params.get('perc_interference', 0.33),
                clustering_algo=clustering.get('algo', 'fixed'),
                fixed_width=clustering.get('fixed_width_db', 5.0),
                dynamic_width=clustering.get('dynamic_width_db', 2.5),
                rsrp_min_quantile=rsrp_quantiles.get('min', 0.02),
                rsrp_max_quantile=rsrp_quantiles.get('max', 0.98),
                perceived_multiplier=params.get('perceived_multiplier', 3),
                hdbscan_min_cluster_size=polygon_params.get('hdbscan_min_cluster_size', 5),
                alpha_shape_alpha=polygon_params.get('alpha_shape_alpha'),
                max_alphashape_points=polygon_params.get('max_alphashape_points', 2000),
                environment_overrides=config.get('environment_overrides'),
                expected_geohash_precision=validation_params.get('expected_geohash_precision', 7),
                geohash_precision_warning=validation_params.get('geohash_precision_warning', True),
                max_records_per_band=validation_params.get('max_records_per_band', 5_000_000),
                max_data_age_days=validation_params.get('max_data_age_days', 7),
            )
        except Exception as e:
            logger.warning("config_load_failed", error=str(e), using_defaults=True)
            return cls()


# -----------------------------
# Clustering algorithms
# -----------------------------

def cluster_by_anchor_within5(s: pd.Series, width: float = 5.0) -> pd.Series:
    """
    Cluster values using a fixed anchor approach.

    This algorithm works by:
    1. Sort values in ascending order
    2. Take first value as anchor
    3. Add subsequent values to cluster if they're within 'width' of anchor
    4. When a value exceeds width, start a new cluster with that value as new anchor

    Args:
        s: Series of RSRP distance values
        width: Maximum allowed difference from anchor (default 5 dB)

    Returns:
        Series of 1-based cluster IDs aligned to s.index
    """
    vals = s.to_numpy()
    if len(vals) == 0:
        return pd.Series(dtype="int64", index=s.index)

    grp = np.zeros(len(vals), dtype=int)
    anchor = vals[0]
    gid = 0
    for i in range(1, len(vals)):
        if vals[i] - anchor <= width:
            grp[i] = gid
        else:
            gid += 1
            grp[i] = gid
            anchor = vals[i]
    return pd.Series(grp + 1, index=s.index)  # 1-based


def cluster_with_complete_linkage(s: pd.Series, width: float = 5.0) -> pd.Series:
    """
    Cluster values using complete linkage hierarchical clustering.

    This algorithm guarantees that every pair of points within a cluster has
    distance <= 2*width.

    Args:
        s: Series of RSRP distance values
        width: Half of the maximum allowed pairwise distance within a cluster

    Returns:
        Series of 1-based cluster IDs aligned to s.index
    """
    x = pd.to_numeric(s, errors='coerce').dropna().sort_values()
    if x.size == 0:
        return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)
    if x.size == 1:
        out = pd.Series(pd.array([1], dtype="Int64"), index=x.index)
        return out.reindex(s.index)

    X = x.to_numpy().reshape(-1, 1)
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=2 * width,
            linkage='complete',
            metric='euclidean',
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=2 * width,
            linkage='complete',
            affinity='euclidean',
        )

    labels = model.fit_predict(X) + 1  # 1-based labels
    out = pd.Series(pd.array(labels, dtype="Int64"), index=x.index)
    return out.reindex(s.index)


# -----------------------------
# Geohash utilities
# -----------------------------

def count_present_in_kring(gh: str, k: int, present: set) -> int:
    """
    Count how many geohashes from a given set are present within k rings of gh.

    Args:
        gh: Center geohash string
        k: Number of expansion steps (rings)
        present: Set of geohashes to check for presence

    Returns:
        Count of geohashes from 'present' that are within k-ring of gh
    """
    K = kring(gh, k)
    return len(K & present) - 1  # minus 1 to exclude gh itself


# -----------------------------
# Detector
# -----------------------------

class InterferenceDetector:
    """Detects high interference cells using geohash-based spatial analysis."""

    def __init__(self, params: Optional[InterferenceParams] = None):
        """
        Initialize the Interference Detector.

        Args:
            params: Detection parameters (uses defaults if None)
        """
        self.params = params or InterferenceParams()
        # Store original params for idempotent environment overrides (Fix #6)
        self._original_params = InterferenceParams(
            min_filtered_cells_per_grid=self.params.min_filtered_cells_per_grid,
            min_cell_event_count=self.params.min_cell_event_count,
            perc_grid_events=self.params.perc_grid_events,
            dominant_perc_grid_events=self.params.dominant_perc_grid_events,
            dominance_diff=self.params.dominance_diff,
            max_rsrp_diff=self.params.max_rsrp_diff,
            sinr_threshold_db=self.params.sinr_threshold_db,
            k=self.params.k,
            perc_interference=self.params.perc_interference,
            clustering_algo=self.params.clustering_algo,
            fixed_width=self.params.fixed_width,
            dynamic_width=self.params.dynamic_width,
            rsrp_min_quantile=self.params.rsrp_min_quantile,
            rsrp_max_quantile=self.params.rsrp_max_quantile,
            perceived_multiplier=self.params.perceived_multiplier,
            hdbscan_min_cluster_size=self.params.hdbscan_min_cluster_size,
            alpha_shape_alpha=self.params.alpha_shape_alpha,
            max_alphashape_points=self.params.max_alphashape_points,
            environment_overrides=self.params.environment_overrides,
            expected_geohash_precision=self.params.expected_geohash_precision,
            geohash_precision_warning=self.params.geohash_precision_warning,
            max_records_per_band=self.params.max_records_per_band,
            max_data_age_days=self.params.max_data_age_days,
        )

        logger.info(
            "interference_detector_initialized",
            min_filtered_cells=self.params.min_filtered_cells_per_grid,
            max_rsrp_diff=self.params.max_rsrp_diff,
            k=self.params.k,
        )

    def _reset_to_original_params(self) -> None:
        """Reset params to original values before applying new environment overrides."""
        for field in ['min_filtered_cells_per_grid', 'min_cell_event_count', 'perc_grid_events',
                      'dominant_perc_grid_events', 'dominance_diff', 'max_rsrp_diff',
                      'sinr_threshold_db', 'k', 'perc_interference', 'clustering_algo',
                      'fixed_width', 'dynamic_width', 'rsrp_min_quantile', 'rsrp_max_quantile',
                      'perceived_multiplier', 'hdbscan_min_cluster_size', 'alpha_shape_alpha',
                      'max_alphashape_points']:
            setattr(self.params, field, getattr(self._original_params, field))

    def _apply_environment_overrides(self, environment: Optional[str]) -> None:
        """Apply environment-specific parameter overrides (idempotent - Fix #6)."""
        # Always reset to original params first for idempotency
        self._reset_to_original_params()

        if environment is None or self.params.environment_overrides is None:
            return

        overrides = self.params.environment_overrides.get(environment)
        if overrides:
            logger.info(
                "applying_environment_overrides",
                environment=environment,
                overrides=overrides,
            )
            for key, value in overrides.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)

    def detect(
        self,
        df: pd.DataFrame,
        data_type: str = 'measured',
        environment: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Find interference clusters using geohash-based spatial analysis.

        This is the main entry point for interference detection. It orchestrates
        the entire detection pipeline across all frequency bands and returns
        clustered polygon regions.

        Args:
            df: Input DataFrame with required columns (cell_name, avg_rsrp, grid, band)
            data_type: 'perceived' or 'measured'
            environment: Optional environment type ('urban', 'suburban', 'rural')
                        for environment-specific thresholds

        Returns:
            GeoDataFrame with interference cluster polygons containing:
                - cluster_id: Unique cluster identifier
                - band: Frequency band
                - n_grids: Number of interference grids in cluster
                - n_cells: Number of cells involved in cluster
                - cells: List of cell names involved
                - centroid_lat, centroid_lon: Cluster centroid
                - area_km2: Cluster area in square kilometers
                - avg_rsrp: Average RSRP in cluster
                - avg_sinr: Average SINR in cluster (if available)
                - geometry: Alpha shape polygon
        """
        execution_start = time.time()
        logger.info(
            "Interference detector started",
            data_type=data_type,
            input_records=len(df),
            environment=environment,
        )

        # Apply environment-specific overrides if provided
        self._apply_environment_overrides(environment)

        # Validate input (expects canonical column names: grid, avg_rsrp, cell_name, band)
        df = self._validate_input(df, data_type)

        # Check if SINR data is available for validation
        has_sinr = 'avg_sinr' in df.columns
        if has_sinr:
            logger.info("sinr_data_available", action="will_apply_sinr_validation_filter")

        # Log validated input details
        logger.info(
            "input_validated",
            bands=df['band'].nunique(),
            cells=df['cell_name'].nunique(),
        )

        # Process each band and collect interference grids
        all_grids = []
        total_records = 0
        max_records = self.params.max_records_per_band  # Fix #11: use configurable limit

        for band in df['band'].unique():
            band_data_size = len(df[df['band'] == band])
            if band_data_size > max_records:
                logger.error(
                    "band_exceeds_max_records",
                    band=band,
                    records=band_data_size,
                    max_allowed=max_records,
                )
                raise ValueError(
                    f"Band {band} has {band_data_size:,} records, exceeding limit of {max_records:,}"
                )

            grids = self._process_band(df, band, data_type)
            if not grids.empty:
                all_grids.append(grids)
                total_records += len(grids)

                # Check accumulated size
                if total_records > max_records:
                    logger.warning(
                        "accumulated_grids_exceeds_limit",
                        total_records=total_records,
                        max_allowed=max_records,
                    )

        # Handle empty result
        if not all_grids:
            logger.info("no_interference_patterns_found")
            return gpd.GeoDataFrame(
                columns=['cluster_id', 'band', 'n_grids', 'n_cells', 'cells',
                        'centroid_lat', 'centroid_lon', 'area_km2', 'avg_rsrp', 'geometry'],
                crs="EPSG:4326"
            )

        # Combine all band results
        combined_grids = pd.concat(all_grids, ignore_index=True)

        # Create polygon clusters from interference grids
        cluster_gdf = self._create_interference_polygons(combined_grids, has_sinr=has_sinr)

        # Apply SINR filter if data is available
        if has_sinr and len(cluster_gdf) > 0 and 'avg_sinr' in cluster_gdf.columns:
            before_filter = len(cluster_gdf)
            sinr_threshold = self.params.sinr_threshold_db
            cluster_gdf = cluster_gdf[cluster_gdf['avg_sinr'] <= sinr_threshold]
            filtered_count = before_filter - len(cluster_gdf)
            if filtered_count > 0:
                logger.info(
                    "sinr_filter_applied",
                    threshold_db=sinr_threshold,
                    clusters_removed=filtered_count,
                    clusters_remaining=len(cluster_gdf)
                )

        # Log final summary
        execution_elapsed = time.time() - execution_start
        logger.info(
            "Interference detection complete",
            execution_time_s=round(execution_elapsed, 2),
            total_clusters=len(cluster_gdf),
            total_grids=combined_grids['grid'].nunique(),
        )

        return cluster_gdf

    def _validate_input(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate input DataFrame and data_type parameter."""
        if df is None:
            raise ValueError("Input DataFrame cannot be None")

        if df.empty:
            raise ValueError("Input DataFrame is empty")

        required_columns = ['cell_name', 'avg_rsrp', 'grid', 'band']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        valid_data_types = ['perceived', 'measured']
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be one of: {valid_data_types}")

        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            null_report = null_counts[null_counts > 0].to_dict()
            logger.warning("null_values_found", null_counts=null_report, action="dropping_rows")
            df = df.dropna(subset=required_columns)

            if df.empty:
                raise ValueError("All rows contain null values in required columns")

        # Validate geohash precision (Fix #5)
        if self.params.geohash_precision_warning and self.params.expected_geohash_precision:
            self._validate_geohash_precision(df)

        # Validate data freshness (Fix #10)
        if self.params.max_data_age_days is not None:
            self._validate_data_freshness(df)

        return df

    def _validate_geohash_precision(self, df: pd.DataFrame) -> None:
        """Validate geohash precision matches expected value (Fix #5)."""
        sample_grids = df['grid'].dropna().head(100)
        if sample_grids.empty:
            return

        precisions = sample_grids.str.len().unique()
        expected = self.params.expected_geohash_precision

        if len(precisions) > 1:
            logger.warning(
                "mixed_geohash_precision",
                precisions_found=list(precisions),
                expected_precision=expected,
                recommendation="Ensure consistent geohash precision across dataset",
            )
        elif len(precisions) == 1 and precisions[0] != expected:
            actual = int(precisions[0])
            # Provide context on resolution difference
            resolution_info = {
                5: "~4.9km x 4.9km",
                6: "~1.2km x 0.6km",
                7: "~153m x 153m",
                8: "~38m x 19m",
                9: "~4.8m x 4.8m",
            }
            logger.warning(
                "geohash_precision_mismatch",
                expected_precision=expected,
                actual_precision=actual,
                expected_resolution=resolution_info.get(expected, "unknown"),
                actual_resolution=resolution_info.get(actual, "unknown"),
            )

    def _validate_data_freshness(self, df: pd.DataFrame) -> None:
        """Validate data is not too old (Fix #10)."""
        from datetime import datetime, timedelta

        timestamp_cols = ['timestamp', 'date', 'collection_date', 'created_at']
        ts_col = None
        for col in timestamp_cols:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            # No timestamp column found, skip validation
            return

        try:
            dates = pd.to_datetime(df[ts_col], errors='coerce')
            valid_dates = dates.dropna()
            if valid_dates.empty:
                return

            max_date = valid_dates.max()
            min_date = valid_dates.min()
            now = datetime.now()
            age_days = (now - max_date).days

            if age_days > self.params.max_data_age_days:
                logger.warning(
                    "data_freshness_warning",
                    max_data_age_days=self.params.max_data_age_days,
                    actual_age_days=age_days,
                    newest_data=max_date.strftime("%Y-%m-%d"),
                    oldest_data=min_date.strftime("%Y-%m-%d"),
                    recommendation="Consider using more recent data for accurate interference detection",
                )
        except Exception as e:
            logger.debug("data_freshness_check_failed", error=str(e))

    def _process_band(
        self,
        df: pd.DataFrame,
        band: str,
        data_type: str
    ) -> pd.DataFrame:
        """Process a single frequency band for interference detection.

        Returns:
            DataFrame with interference grids for this band (empty if none found)
        """
        cfg = self.params
        band_start = time.time()
        logger.info("processing_band", band=band)

        grid_geo_data_diff = df[df.band == band].copy()

        # STEP 1: Normalize grid event percentages
        if 'perc_grid_events' in grid_geo_data_diff.columns:
            den = grid_geo_data_diff.groupby('grid')['perc_grid_events'].transform('sum')
            grid_geo_data_diff['perc_grid_events'] = (
                grid_geo_data_diff['perc_grid_events'].div(den).fillna(0)
            )

        # STEP 2: Apply initial data quality filters
        if data_type != 'perceived' and 'event_count' in grid_geo_data_diff.columns:
            if 'perc_grid_events' in grid_geo_data_diff.columns:
                grid_geo_data_diff = grid_geo_data_diff[
                    (grid_geo_data_diff.event_count >= cfg.min_cell_event_count) &
                    (grid_geo_data_diff.perc_grid_events >= cfg.perc_grid_events)
                ]

        # STEP 3: Calculate RSRP gap from strongest cell
        grid_rsrp_max = (
            grid_geo_data_diff.groupby(['grid'], as_index=False)
            .agg({'avg_rsrp': 'max'})
        )
        grid_rsrp_max.rename(columns={'avg_rsrp': 'max_rsrp'}, inplace=True)
        grid_geo_data_diff = grid_geo_data_diff.merge(grid_rsrp_max, on='grid', how='inner')

        grid_geo_data_diff['rsrp_dist_max'] = (
            grid_geo_data_diff['max_rsrp'] - grid_geo_data_diff['avg_rsrp']
        )
        grid_geo_data_diff['rsrp_dist_max'] = pd.to_numeric(
            grid_geo_data_diff['rsrp_dist_max'], errors='coerce'
        )

        # STEP 4: Filter cells within max_rsrp_diff of strongest cell
        # Use band-specific threshold if available, otherwise fall back to config default
        band_thresholds = BAND_INTERFERENCE_THRESHOLDS.get(band, {})
        max_rsrp_diff = band_thresholds.get('max_rsrp_diff', cfg.max_rsrp_diff)
        grid_geo_data_diff = grid_geo_data_diff[
            grid_geo_data_diff['rsrp_dist_max'] <= max_rsrp_diff
        ]

        # STEP 5: Cluster cells by RSRP similarity
        grid_geo_data_diff = grid_geo_data_diff.sort_values(['grid', 'rsrp_dist_max'])
        if cfg.clustering_algo == 'fixed':
            grid_geo_data_diff['grid_cluster_no'] = (
                grid_geo_data_diff.groupby('grid', group_keys=False)['rsrp_dist_max']
                .apply(lambda s: cluster_by_anchor_within5(s, width=cfg.fixed_width))
            )
        elif cfg.clustering_algo == 'dynamic-sklearn':
            grid_geo_data_diff['grid_cluster_no'] = (
                grid_geo_data_diff.groupby('grid', group_keys=False)['rsrp_dist_max']
                .apply(lambda s: cluster_with_complete_linkage(s, width=cfg.dynamic_width))
            )

        # Give each cluster a readable name
        labels_num = grid_geo_data_diff['grid_cluster_no'].astype('Int64').fillna(0)
        grid_geo_data_diff['grid_cluster_name'] = (
            grid_geo_data_diff['grid'].astype(str) + '_grp' +
            labels_num.astype(int).astype(str).str.zfill(2)
        )

        # STEP 6: Exclude grids with dominant cells
        dominant_cell_grids = self._find_dominant_grids(df, band, grid_rsrp_max, data_type)
        grid_geo_data_diff = grid_geo_data_diff[
            ~grid_geo_data_diff.grid.isin(dominant_cell_grids)
        ]

        # STEP 7: Filter clusters by minimum cell count
        grid_counts = grid_geo_data_diff.groupby("grid_cluster_name").size().reset_index(name="count")

        if data_type == 'perceived':
            min_cells = int(np.ceil(cfg.min_filtered_cells_per_grid * cfg.perceived_multiplier))
            grid_counts = grid_counts[grid_counts['count'] >= min_cells]
        else:
            grid_counts = grid_counts[grid_counts['count'] >= cfg.min_filtered_cells_per_grid]

        grid_geo_data_diff = grid_geo_data_diff.merge(grid_counts, on="grid_cluster_name", how="inner")

        # STEP 8: Apply k-ring spatial clustering
        grid_geo_data_geo_filtered = self._apply_spatial_clustering(grid_geo_data_diff)

        # STEP 9: Calculate RSRP-based weighting
        grid_geo_data_geo_filtered = self._calculate_rsrp_weighting(grid_geo_data_geo_filtered)

        # Handle empty result
        if len(grid_geo_data_geo_filtered) == 0:
            logger.warning("no_interference_patterns_for_band", band=band)
            return pd.DataFrame()

        # Log band-level results
        band_elapsed = time.time() - band_start
        n_cells = grid_geo_data_geo_filtered['cell_name'].nunique()

        logger.info(
            "band_processing_completed",
            band=band,
            elapsed_time_s=round(band_elapsed, 2),
            interference_cells=n_cells,
        )

        return grid_geo_data_geo_filtered

    def _find_dominant_grids(
        self,
        df: pd.DataFrame,
        band: str,
        grid_rsrp_max: pd.DataFrame,
        data_type: str
    ) -> list:
        """Identify grids with dominant cells."""
        cfg = self.params

        # Use band-specific threshold if available, otherwise fall back to config default
        band_thresholds = BAND_INTERFERENCE_THRESHOLDS.get(band, {})
        dominance_diff = band_thresholds.get('dominance_diff', cfg.dominance_diff)

        grid_geo_data_dominant = df[df.band == band].copy()
        grid_geo_data_dominant = grid_geo_data_dominant.merge(
            grid_rsrp_max[['grid', 'max_rsrp']], on='grid', how='inner'
        )

        # Find second-strongest cell in each grid (using band-filtered data)
        band_data = df[df.band == band]
        second = (
            band_data.groupby('grid')['avg_rsrp']
            .nlargest(2)
            .groupby(level=0)
            .nth(1)
            .reset_index(name='avg_rsrp_second')
        )
        grid_geo_data_dominant = grid_geo_data_dominant.merge(second, on='grid', how='inner')

        # Calculate gap between strongest and 2nd strongest
        grid_geo_data_dominant['rsrp_diff_1_2'] = (
            grid_geo_data_dominant['max_rsrp'] - grid_geo_data_dominant['avg_rsrp_second']
        )

        # Filter grids where strongest cell dominates
        if 'perc_grid_events' in grid_geo_data_dominant.columns:
            if data_type == 'perceived':
                grid_geo_data_dominant = grid_geo_data_dominant[
                    grid_geo_data_dominant['rsrp_diff_1_2'] >= dominance_diff
                ]
            else:
                grid_geo_data_dominant = grid_geo_data_dominant[
                    (grid_geo_data_dominant.perc_grid_events >= cfg.dominant_perc_grid_events) &
                    (grid_geo_data_dominant['rsrp_diff_1_2'] >= dominance_diff)
                ]
        else:
            grid_geo_data_dominant = grid_geo_data_dominant[
                grid_geo_data_dominant['rsrp_diff_1_2'] >= dominance_diff
            ]

        return list(set(grid_geo_data_dominant.grid.to_list()))

    def _apply_spatial_clustering(self, grid_geo_data_diff: pd.DataFrame) -> pd.DataFrame:
        """Apply k-ring spatial clustering to filter isolated false positives."""
        cfg = self.params

        interference_set = set(grid_geo_data_diff['grid'].astype(str))

        # For each grid, count neighbors within k-ring that also have interference
        records = [
            (gh, count_present_in_kring(gh, cfg.k, interference_set))
            for gh in interference_set
        ]
        interferer_k_df = pd.DataFrame(records, columns=['grid', f'interferers_within_{cfg.k}_steps'])

        # Filter: keep only grids with > perc_interference of k-ring neighbors showing interference
        threshold = ((2 * cfg.k) + 1) ** 2 * cfg.perc_interference
        interferer_k_df = interferer_k_df[
            interferer_k_df[f'interferers_within_{cfg.k}_steps'] > threshold
        ]

        grid_geo_data_geo_filtered = grid_geo_data_diff.merge(interferer_k_df, on="grid", how="inner")

        return grid_geo_data_geo_filtered

    def _calculate_rsrp_weighting(self, grid_geo_data_geo_filtered: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSRP-based weighting for interference severity."""
        cfg = self.params

        if len(grid_geo_data_geo_filtered) == 0:
            return grid_geo_data_geo_filtered

        rsrp_min = grid_geo_data_geo_filtered['avg_rsrp'].quantile(cfg.rsrp_min_quantile)
        rsrp_max = grid_geo_data_geo_filtered['avg_rsrp'].quantile(cfg.rsrp_max_quantile)

        if rsrp_max > rsrp_min:
            grid_geo_data_geo_filtered['weight'] = (
                (grid_geo_data_geo_filtered['avg_rsrp'] - rsrp_min) / (rsrp_max - rsrp_min)
            ).clip(0, 1)
        else:
            grid_geo_data_geo_filtered['weight'] = 0.5

        return grid_geo_data_geo_filtered

    def _create_interference_polygons(
        self,
        grids_df: pd.DataFrame,
        has_sinr: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Create clustered polygon regions from interference grids.

        Groups nearby interference grids using HDBSCAN and creates alpha shape
        polygons for each cluster.

        Args:
            grids_df: DataFrame with interference grids (must have 'grid', 'band', 'cell_name', 'avg_rsrp')
            has_sinr: Whether avg_sinr column is available for SINR-based filtering

        Returns:
            GeoDataFrame with cluster polygons and metadata
        """
        cfg = self.params
        all_clusters = []

        # Process each band separately
        for band in grids_df['band'].unique():
            band_grids = grids_df[grids_df['band'] == band].copy()

            if len(band_grids) < cfg.hdbscan_min_cluster_size:
                logger.info("skipping_band_clustering", band=band, grids=len(band_grids), reason="too_few_grids")
                continue

            # Decode geohashes to coordinates
            unique_grids = band_grids['grid'].unique()
            coords_map = {}
            for gh in unique_grids:
                try:
                    lat, lon = decode_geohash(gh)
                    coords_map[gh] = (lat, lon)
                except Exception:
                    continue

            band_grids = band_grids[band_grids['grid'].isin(coords_map.keys())]
            if len(band_grids) < cfg.hdbscan_min_cluster_size:
                continue

            # Vectorized coordinate mapping using DataFrame merge
            coords_df = pd.DataFrame([
                {'grid': gh, 'latitude': lat, 'longitude': lon}
                for gh, (lat, lon) in coords_map.items()
            ])
            # Drop existing lat/lon columns to avoid suffixes after merge
            cols_to_drop = [c for c in ['latitude', 'longitude'] if c in band_grids.columns]
            if cols_to_drop:
                band_grids = band_grids.drop(columns=cols_to_drop)
            band_grids = band_grids.merge(coords_df, on='grid', how='left')

            # Cluster using HDBSCAN on unique coordinates
            unique_coords = coords_df[['latitude', 'longitude']].drop_duplicates()
            coords = unique_coords.values

            # Skip if too few points for clustering
            if len(coords) < cfg.hdbscan_min_cluster_size:
                logger.info("skipping_band_clustering", band=band, unique_coords=len(coords), reason="too_few_coordinates")
                continue

            coords_rad = np.radians(coords)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=cfg.hdbscan_min_cluster_size,
                metric='haversine'
            )
            labels = clusterer.fit_predict(coords_rad)

            # Vectorized label mapping using merge
            unique_coords = unique_coords.copy()
            unique_coords['cluster_id'] = labels
            band_grids = band_grids.merge(
                unique_coords[['latitude', 'longitude', 'cluster_id']],
                on=['latitude', 'longitude'],
                how='left'
            )
            band_grids['cluster_id'] = band_grids['cluster_id'].fillna(-1).astype(int)

            # Filter out noise
            clustered = band_grids[band_grids['cluster_id'] >= 0].copy()

            if clustered.empty:
                logger.info("no_valid_clusters_for_band", band=band)
                continue

            # Create polygons for each cluster
            for cluster_id in clustered['cluster_id'].unique():
                cluster_data = clustered[clustered['cluster_id'] == cluster_id]
                cluster_coords = cluster_data[['longitude', 'latitude']].drop_duplicates().values

                if len(cluster_coords) < 3:
                    continue

                # Create polygon
                polygon = self._create_alpha_shape(cluster_coords)
                if polygon is None or polygon.is_empty:
                    continue

                # Gather cluster metadata
                cells_involved = cluster_data['cell_name'].unique().tolist()
                n_unique_grids = cluster_data['grid'].nunique()

                cluster_meta = {
                    'cluster_id': f"{band}_{cluster_id}",
                    'band': band,
                    'n_grids': n_unique_grids,
                    'n_cells': len(cells_involved),
                    'cells': cells_involved,
                    'centroid_lat': cluster_data['latitude'].mean(),
                    'centroid_lon': cluster_data['longitude'].mean(),
                    'avg_rsrp': cluster_data['avg_rsrp'].mean(),
                    'geometry': polygon
                }

                # Add SINR if available
                if has_sinr and 'avg_sinr' in cluster_data.columns:
                    cluster_meta['avg_sinr'] = cluster_data['avg_sinr'].mean()

                all_clusters.append(cluster_meta)

            n_clusters = clustered['cluster_id'].nunique()
            logger.info("band_clusters_created", band=band, clusters=n_clusters)

        if not all_clusters:
            columns = ['cluster_id', 'band', 'n_grids', 'n_cells', 'cells',
                       'centroid_lat', 'centroid_lon', 'area_km2', 'avg_rsrp', 'geometry']
            if has_sinr:
                columns.insert(-1, 'avg_sinr')
            return gpd.GeoDataFrame(columns=columns, crs="EPSG:4326")

        gdf = gpd.GeoDataFrame(all_clusters, crs="EPSG:4326")

        # Batch calculate areas using single UTM projection for efficiency
        gdf['area_km2'] = self._calculate_areas_batch(gdf)

        gdf = gdf.sort_values(['band', 'n_grids'], ascending=[True, False]).reset_index(drop=True)

        logger.info("total_interference_clusters", count=len(gdf))
        return gdf

    def _grid_based_subsample(self, coords: np.ndarray, max_points: int) -> np.ndarray:
        """
        Subsample coordinates using grid-based sampling (Fix #12).

        This method preserves cluster boundaries better than random sampling by:
        1. Dividing the coordinate space into a grid
        2. Selecting one representative point from each grid cell
        3. Prioritizing boundary points (extreme values in each direction)

        Args:
            coords: Array of (longitude, latitude) coordinates
            max_points: Maximum number of points to return

        Returns:
            Subsampled coordinates array
        """
        if len(coords) <= max_points:
            return coords

        # Calculate grid dimensions to get approximately max_points cells
        n_cells = int(np.sqrt(max_points))

        # Get coordinate bounds
        lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
        lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

        # Handle edge case where all points have same coordinate
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        if lon_range == 0:
            lon_range = 0.0001
        if lat_range == 0:
            lat_range = 0.0001

        # Calculate grid cell size
        cell_width = lon_range / n_cells
        cell_height = lat_range / n_cells

        # Assign each point to a grid cell
        cell_x = ((coords[:, 0] - lon_min) / cell_width).astype(int)
        cell_y = ((coords[:, 1] - lat_min) / cell_height).astype(int)
        cell_x = np.clip(cell_x, 0, n_cells - 1)
        cell_y = np.clip(cell_y, 0, n_cells - 1)
        cell_ids = cell_x * n_cells + cell_y

        # Select one point per cell (first occurrence for determinism)
        unique_cells, first_indices = np.unique(cell_ids, return_index=True)
        selected_coords = coords[first_indices]

        # If we have too few points, add boundary points
        if len(selected_coords) < max_points:
            # Add extreme points (boundary preservation)
            boundary_indices = [
                coords[:, 0].argmin(),  # westmost
                coords[:, 0].argmax(),  # eastmost
                coords[:, 1].argmin(),  # southmost
                coords[:, 1].argmax(),  # northmost
            ]
            boundary_coords = coords[boundary_indices]
            selected_coords = np.vstack([selected_coords, boundary_coords])
            # Remove duplicates
            selected_coords = np.unique(selected_coords, axis=0)

        # If still too many, truncate
        if len(selected_coords) > max_points:
            selected_coords = selected_coords[:max_points]

        return selected_coords

    def _create_alpha_shape(self, coords: np.ndarray) -> Optional[Polygon]:
        """
        Create an alpha shape polygon from coordinates.

        Args:
            coords: Array of (longitude, latitude) coordinates

        Returns:
            Shapely Polygon or None if creation fails
        """
        cfg = self.params

        # Subsample if too many points using grid-based sampling (Fix #12)
        # Grid-based sampling preserves cluster boundaries better than random sampling
        if len(coords) > cfg.max_alphashape_points:
            coords = self._grid_based_subsample(coords, cfg.max_alphashape_points)

        try:
            if cfg.alpha_shape_alpha is None:
                shape = alphashape.alphashape(coords.tolist())
            else:
                shape = alphashape.alphashape(coords.tolist(), float(cfg.alpha_shape_alpha))

            if shape.is_empty or not shape.is_valid:
                # Fallback to convex hull
                points = [Point(lon, lat) for lon, lat in coords]
                shape = gpd.GeoSeries(points).unary_union.convex_hull

            return unary_union([shape]).buffer(0)

        except Exception as e:
            logger.warning("alpha_shape_creation_failed", error=str(e), fallback="convex_hull")
            # Fallback to convex hull
            try:
                points = [Point(lon, lat) for lon, lat in coords]
                hull = gpd.GeoSeries(points).unary_union.convex_hull
                # Handle case where convex_hull returns a point or line
                if hull.geom_type in ('Point', 'MultiPoint', 'LineString'):
                    return hull.buffer(0.001)  # Small buffer to create polygon
                return hull
            except Exception as fallback_err:
                logger.error(
                    "convex_hull_fallback_failed",
                    error=str(fallback_err),
                    n_coords=len(coords)
                )
                return None

    def _calculate_areas_batch(self, gdf: gpd.GeoDataFrame) -> List[float]:
        """
        Calculate areas in km² for all geometries using batch UTM projection.

        This is more efficient than per-polygon projection as it projects
        all geometries at once using a common UTM zone.

        Args:
            gdf: GeoDataFrame with geometry column in EPSG:4326

        Returns:
            List of areas in square kilometers
        """
        if len(gdf) == 0:
            return []

        try:
            # Use centroid of all geometries to determine UTM zone
            all_centroids = gdf.geometry.centroid
            mean_lon = all_centroids.x.mean()
            mean_lat = all_centroids.y.mean()

            utm_zone = int((mean_lon + 180) / 6) + 1
            hemisphere = 'north' if mean_lat >= 0 else 'south'
            epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone

            # Project all geometries at once
            gdf_projected = gdf.to_crs(epsg=epsg_code)
            areas_m2 = gdf_projected.geometry.area

            return [round(a / 1_000_000, 3) for a in areas_m2]

        except Exception as e:
            logger.warning("batch_area_calculation_failed", error=str(e), fallback="per_geometry")
            return [self._calculate_area_km2(geom) for geom in gdf.geometry]

    def _calculate_area_km2(self, geometry) -> float:
        """
        Calculate area in km² for a single geometry using UTM projection.

        Args:
            geometry: Shapely geometry in EPSG:4326

        Returns:
            Area in square kilometers
        """
        if geometry is None or geometry.is_empty:
            return 0.0

        try:
            gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
            centroid = geometry.centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            hemisphere = 'north' if centroid.y >= 0 else 'south'
            epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone

            gdf_projected = gdf.to_crs(epsg=epsg_code)
            area_m2 = gdf_projected.geometry.iloc[0].area
            return round(area_m2 / 1_000_000, 3)

        except Exception as e:
            logger.warning("area_calculation_failed", error=str(e), fallback="approximation")
            # Fallback approximation
            return round(
                geometry.area * KM_PER_DEGREE_LAT * KM_PER_DEGREE_LAT *
                np.cos(np.radians(geometry.centroid.y)),
                3
            )


# -----------------------------
# Root Cause Analysis
# -----------------------------

@dataclass
class RootCauseParams:
    """Parameters for interference root cause analysis."""
    # Tilt analysis thresholds
    low_tilt_threshold_deg: float = 4.0  # Tilts below this are "low"
    high_tilt_threshold_deg: float = 10.0  # Tilts above this are "high"
    tilt_spread_threshold_deg: float = 3.0  # Spread indicating inconsistent tilts

    # Azimuth analysis
    azimuth_convergence_threshold_deg: float = 60.0  # Azimuths within this are "converging"

    # Distance analysis
    close_proximity_km: float = 0.5  # Cells closer than this are "very close"
    normal_proximity_km: float = 1.5  # Cells closer than this are "close"

    # Power analysis (relative)
    high_power_percentile: float = 0.75  # Cells above this percentile are "high power"

    # Recommendation thresholds
    min_tilt_increase_deg: float = 1.0  # Minimum recommended tilt increase
    max_tilt_increase_deg: float = 4.0  # Maximum recommended tilt increase


class InterferenceRootCauseAnalyzer:
    """
    Analyzes interference clusters to determine root causes and generate recommendations.

    This analyzer examines cell configuration data (tilt, azimuth, power, height) for
    cells involved in interference clusters and identifies likely causes:
    - Low tilt causing overshoot
    - High power causing extended coverage
    - Azimuth convergence (multiple cells pointing at same area)
    - Close proximity (cells too close together)

    Example:
        >>> analyzer = InterferenceRootCauseAnalyzer()
        >>> enriched_gdf = analyzer.analyze(interference_gdf, gis_df)
        >>> print(enriched_gdf[['cluster_id', 'root_cause', 'recommendations']])
    """

    def __init__(self, params: Optional[RootCauseParams] = None):
        """Initialize the root cause analyzer."""
        self.params = params or RootCauseParams()
        logger.info("InterferenceRootCauseAnalyzer initialized")

    def analyze(
        self,
        interference_gdf: gpd.GeoDataFrame,
        gis_df: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        Analyze interference clusters and add root cause analysis.

        Args:
            interference_gdf: GeoDataFrame from InterferenceDetector.detect()
                Must have 'cells' column with list of cell names per cluster
            gis_df: Cell configuration DataFrame with columns:
                - cell_name: Cell identifier
                - latitude, longitude: Cell location
                - bearing: Antenna azimuth (degrees)
                - tilt_elc: Electrical tilt (degrees)
                - tilt_mech: Mechanical tilt (degrees)
                - Optional: tx_power, antenna_height

        Returns:
            GeoDataFrame with additional columns:
                - root_cause: Primary identified cause
                - root_cause_details: Dict with analysis metrics
                - recommendations: List of recommended actions
                - priority: Recommendation priority (high/medium/low)
        """
        if interference_gdf.empty:
            logger.info("No interference clusters to analyze")
            return interference_gdf

        # Validate GIS data
        required_cols = ['cell_name', 'latitude', 'longitude']
        missing = [c for c in required_cols if c not in gis_df.columns]
        if missing:
            raise ValueError(f"GIS data missing required columns: {missing}")

        # Check for optional but important columns
        has_tilt = 'tilt_elc' in gis_df.columns or 'tilt_mech' in gis_df.columns
        has_azimuth = 'bearing' in gis_df.columns
        has_power = 'tx_power' in gis_df.columns
        has_height = 'antenna_height' in gis_df.columns  # Fix #8: height-adjusted tilt

        if not has_tilt:
            logger.warning("gis_data_missing_tilt", impact="tilt_analysis_limited")
        if not has_azimuth:
            logger.warning("gis_data_missing_bearing", impact="azimuth_analysis_disabled")
        if not has_height:
            logger.info("gis_data_missing_antenna_height", impact="height_adjusted_tilt_disabled")

        # Index GIS data for fast lookup
        gis_indexed = gis_df.set_index('cell_name')

        # Analyze each cluster
        analyses = []
        for idx, cluster in interference_gdf.iterrows():
            analysis = self._analyze_cluster(cluster, gis_indexed, has_tilt, has_azimuth, has_power, has_height)
            analyses.append(analysis)

        # Add analysis columns to GeoDataFrame
        result = interference_gdf.copy()
        result['root_cause'] = [a['root_cause'] for a in analyses]
        result['root_cause_details'] = [a['details'] for a in analyses]
        result['recommendations'] = [a['recommendations'] for a in analyses]
        result['priority'] = [a['priority'] for a in analyses]

        logger.info(
            "root_cause_analysis_complete",
            clusters_analyzed=len(result),
            high_priority=sum(1 for a in analyses if a['priority'] == 'high'),
            medium_priority=sum(1 for a in analyses if a['priority'] == 'medium'),
        )

        return result

    def _analyze_cluster(
        self,
        cluster: pd.Series,
        gis_indexed: pd.DataFrame,
        has_tilt: bool,
        has_azimuth: bool,
        has_power: bool,
        has_height: bool = False
    ) -> dict:
        """Analyze a single interference cluster."""
        cfg = self.params
        cells = cluster.get('cells', [])

        if not cells:
            return {
                'root_cause': 'unknown',
                'details': {},
                'recommendations': [],
                'priority': 'low'
            }

        # Gather cell configurations
        cell_configs = []
        for cell_name in cells:
            if cell_name in gis_indexed.index:
                cell_data = gis_indexed.loc[cell_name]
                config = {
                    'cell_name': cell_name,
                    'lat': cell_data.get('latitude', 0),
                    'lon': cell_data.get('longitude', 0),
                }
                if has_tilt:
                    tilt_elc = cell_data.get('tilt_elc', 0) or 0
                    tilt_mech = cell_data.get('tilt_mech', 0) or 0
                    config['total_tilt'] = tilt_elc + tilt_mech
                    config['tilt_elc'] = tilt_elc
                    config['tilt_mech'] = tilt_mech
                if has_azimuth:
                    config['azimuth'] = cell_data.get('bearing', 0) or 0
                if has_power:
                    config['tx_power'] = cell_data.get('tx_power', 0) or 0
                if has_height:
                    config['antenna_height'] = cell_data.get('antenna_height', 0) or 0
                cell_configs.append(config)

        if not cell_configs:
            return {
                'root_cause': 'unknown',
                'details': {'error': 'No cell configurations found in GIS data'},
                'recommendations': ['Verify cell names match between data sources'],
                'priority': 'low'
            }

        # Perform analysis
        details = {}
        causes = []
        recommendations = []

        # 1. Tilt Analysis (with height adjustment - Fix #8)
        if has_tilt:
            tilts = [c['total_tilt'] for c in cell_configs if 'total_tilt' in c]
            if tilts:
                avg_tilt = np.mean(tilts)
                min_tilt = np.min(tilts)
                max_tilt = np.max(tilts)
                tilt_spread = max_tilt - min_tilt

                details['avg_tilt_deg'] = round(avg_tilt, 1)
                details['min_tilt_deg'] = round(min_tilt, 1)
                details['max_tilt_deg'] = round(max_tilt, 1)
                details['tilt_spread_deg'] = round(tilt_spread, 1)

                # Height-adjusted tilt analysis (Fix #8)
                # Tall sites with low tilt cause more overshoot than shorter sites
                # Effective range ≈ height / tan(tilt) for small angles
                if has_height:
                    height_adjusted_cells = []
                    for cell in cell_configs:
                        height = cell.get('antenna_height', 0)
                        tilt = cell.get('total_tilt', 0)
                        if height > 0 and tilt > 0:
                            # Calculate effective coverage distance (simplified model)
                            # Range in km ≈ height(m) / (tan(tilt) * 1000)
                            effective_range_km = height / (np.tan(np.radians(tilt)) * 1000)
                            cell['effective_range_km'] = round(effective_range_km, 2)
                            # Flag cells with large effective range (potential overshooters)
                            if effective_range_km > 2.0:  # > 2km range indicates overshoot risk
                                height_adjusted_cells.append(cell)

                    if height_adjusted_cells:
                        details['height_adjusted_overshoot_cells'] = [
                            {
                                'cell': c['cell_name'],
                                'height_m': c.get('antenna_height', 0),
                                'tilt_deg': c.get('total_tilt', 0),
                                'effective_range_km': c.get('effective_range_km', 0)
                            }
                            for c in height_adjusted_cells
                        ]
                        # Add height-aware recommendations
                        for cell in height_adjusted_cells:
                            height = cell.get('antenna_height', 30)  # default 30m
                            current_tilt = cell.get('total_tilt', 0)
                            # Calculate tilt needed for ~1.5km range
                            target_range_km = 1.5
                            suggested_tilt = np.degrees(np.arctan(height / (target_range_km * 1000)))
                            if suggested_tilt > current_tilt + 1:  # Only recommend if > 1 degree increase
                                recommendations.append({
                                    'action': 'increase_tilt',
                                    'cell': cell['cell_name'],
                                    'current_tilt': round(current_tilt, 1),
                                    'suggested_tilt': round(suggested_tilt, 1),
                                    'antenna_height_m': height,
                                    'reason': f'Tall site ({height}m) with low tilt causing ~{cell.get("effective_range_km", 0)}km overshoot'
                                })
                                if 'tall_site_overshoot' not in causes:
                                    causes.append('tall_site_overshoot')

                # Check for low tilt (causing overshoot) - standard analysis
                low_tilt_cells = [c for c in cell_configs if c.get('total_tilt', 99) < cfg.low_tilt_threshold_deg]
                if low_tilt_cells:
                    causes.append('low_tilt')
                    details['low_tilt_cells'] = [c['cell_name'] for c in low_tilt_cells]
                    for cell in low_tilt_cells:
                        # Skip if already handled by height-adjusted analysis
                        if has_height and cell.get('effective_range_km', 0) > 2.0:
                            continue
                        current = cell.get('total_tilt', 0)
                        suggested = min(current + 2, cfg.high_tilt_threshold_deg)
                        recommendations.append({
                            'action': 'increase_tilt',
                            'cell': cell['cell_name'],
                            'current_tilt': round(current, 1),
                            'suggested_tilt': round(suggested, 1),
                            'reason': 'Low tilt causing coverage overshoot'
                        })

                # Check for inconsistent tilts
                if tilt_spread > cfg.tilt_spread_threshold_deg:
                    causes.append('inconsistent_tilt')
                    details['tilt_inconsistency'] = True

        # 2. Azimuth Analysis
        if has_azimuth:
            azimuths = [c['azimuth'] for c in cell_configs if 'azimuth' in c]
            if len(azimuths) >= 2:
                # Check for azimuth convergence (cells pointing at similar direction)
                azimuth_pairs = []
                for i, az1 in enumerate(azimuths):
                    for az2 in azimuths[i+1:]:
                        # Calculate angular difference (handle wraparound)
                        diff = abs(az1 - az2)
                        diff = min(diff, 360 - diff)
                        azimuth_pairs.append(diff)

                if azimuth_pairs:
                    min_az_diff = min(azimuth_pairs)
                    details['min_azimuth_diff_deg'] = round(min_az_diff, 1)

                    if min_az_diff < cfg.azimuth_convergence_threshold_deg:
                        causes.append('azimuth_convergence')
                        # Find converging cells
                        converging = self._find_converging_cells(cell_configs, cfg.azimuth_convergence_threshold_deg)
                        if converging:
                            details['converging_cells'] = converging
                            recommendations.append({
                                'action': 'adjust_azimuth',
                                'cells': converging,
                                'reason': 'Multiple cells with similar azimuth causing overlap'
                            })

        # 3. Proximity Analysis
        if len(cell_configs) >= 2:
            distances = self._calculate_cell_distances(cell_configs)
            if distances:
                min_dist = min(distances)
                avg_dist = np.mean(distances)
                details['min_cell_distance_km'] = round(min_dist, 2)
                details['avg_cell_distance_km'] = round(avg_dist, 2)

                if min_dist < cfg.close_proximity_km:
                    causes.append('very_close_proximity')
                    recommendations.append({
                        'action': 'review_site_design',
                        'reason': f'Cells only {min_dist:.2f} km apart - consider site consolidation or sector reorientation'
                    })
                elif min_dist < cfg.normal_proximity_km:
                    causes.append('close_proximity')

        # 4. Determine primary root cause and priority
        root_cause = self._determine_primary_cause(causes)
        priority = self._determine_priority(causes, details, cluster)

        # Format recommendations as list of strings for easier consumption
        formatted_recs = self._format_recommendations(recommendations)

        return {
            'root_cause': root_cause,
            'details': details,
            'recommendations': formatted_recs,
            'priority': priority
        }

    def _find_converging_cells(self, cell_configs: List[dict], threshold: float) -> List[str]:
        """Find cells with converging azimuths."""
        converging = set()
        for i, c1 in enumerate(cell_configs):
            if 'azimuth' not in c1:
                continue
            for c2 in cell_configs[i+1:]:
                if 'azimuth' not in c2:
                    continue
                diff = abs(c1['azimuth'] - c2['azimuth'])
                diff = min(diff, 360 - diff)
                if diff < threshold:
                    converging.add(c1['cell_name'])
                    converging.add(c2['cell_name'])
        return list(converging)

    def _calculate_cell_distances(self, cell_configs: List[dict]) -> List[float]:
        """Calculate pairwise distances between cells in km."""
        distances = []
        for i, c1 in enumerate(cell_configs):
            for c2 in cell_configs[i+1:]:
                dist = self._haversine_km(c1['lat'], c1['lon'], c2['lat'], c2['lon'])
                distances.append(dist)
        return distances

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in kilometers."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _determine_primary_cause(self, causes: List[str]) -> str:
        """Determine the primary root cause from a list of causes."""
        # Priority order for root causes
        cause_priority = [
            'tall_site_overshoot',  # Fix #8: height-aware overshoot detection
            'low_tilt',
            'very_close_proximity',
            'azimuth_convergence',
            'close_proximity',
            'inconsistent_tilt',
        ]

        for cause in cause_priority:
            if cause in causes:
                return cause

        if causes:
            return causes[0]
        return 'undetermined'

    def _determine_priority(self, causes: List[str], details: dict, cluster: pd.Series) -> str:
        """Determine recommendation priority based on analysis."""
        # High priority indicators (including tall_site_overshoot - Fix #8)
        high_priority_causes = ['low_tilt', 'very_close_proximity', 'tall_site_overshoot']
        if any(c in causes for c in high_priority_causes):
            return 'high'

        # Consider cluster size (n_grids, n_cells)
        n_grids = cluster.get('n_grids', 0)
        n_cells = cluster.get('n_cells', 0)

        if n_grids > 50 or n_cells > 6:
            return 'high'
        elif n_grids > 20 or n_cells > 4:
            return 'medium'

        # Medium priority for other actionable causes
        if 'azimuth_convergence' in causes or 'close_proximity' in causes:
            return 'medium'

        return 'low'

    def _format_recommendations(self, recommendations: List[dict]) -> List[str]:
        """Format recommendations as human-readable strings."""
        formatted = []
        for rec in recommendations:
            action = rec.get('action', '')
            reason = rec.get('reason', '')

            if action == 'increase_tilt':
                cell = rec.get('cell', '')
                current = rec.get('current_tilt', 0)
                suggested = rec.get('suggested_tilt', 0)
                formatted.append(
                    f"Increase tilt on {cell} from {current}° to {suggested}° ({reason})"
                )
            elif action == 'adjust_azimuth':
                cells = rec.get('cells', [])
                formatted.append(
                    f"Review azimuth settings for {', '.join(cells)} ({reason})"
                )
            elif action == 'review_site_design':
                formatted.append(f"Site design review needed: {reason}")
            else:
                formatted.append(reason)

        return formatted


def analyze_interference_root_causes(
    interference_gdf: gpd.GeoDataFrame,
    gis_df: pd.DataFrame,
    params: Optional[RootCauseParams] = None
) -> gpd.GeoDataFrame:
    """
    Convenience function to analyze root causes of interference clusters.

    Args:
        interference_gdf: GeoDataFrame from detect_interference()
        gis_df: Cell configuration DataFrame
        params: Optional analysis parameters

    Returns:
        GeoDataFrame with root cause analysis and recommendations
    """
    analyzer = InterferenceRootCauseAnalyzer(params)
    return analyzer.analyze(interference_gdf, gis_df)


def detect_interference(
    df: pd.DataFrame,
    data_type: str = 'measured',
    params: Optional[InterferenceParams] = None,
    environment: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Convenience function to detect interference clusters.

    Args:
        df: Input DataFrame with cell coverage data
        data_type: 'perceived' or 'measured'
        params: Optional detection parameters
        environment: Optional environment type ('urban', 'suburban', 'rural')

    Returns:
        GeoDataFrame with interference cluster polygons
    """
    detector = InterferenceDetector(params)
    return detector.detect(df, data_type, environment=environment)
