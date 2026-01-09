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
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return cls()

            with open(path, 'r') as f:
                config = json.load(f)

            # Get default parameters section
            params = config.get('default', config)
            clustering = params.get('clustering', {})
            rsrp_quantiles = params.get('rsrp_quantiles', {})

            polygon_params = params.get('polygon_clustering', {})

            return cls(
                min_filtered_cells_per_grid=params.get('min_filtered_cells_per_grid', 4),
                min_cell_event_count=params.get('min_cell_event_count', 2),
                perc_grid_events=params.get('perc_grid_events', 0.05),
                dominant_perc_grid_events=params.get('dominant_perc_grid_events', 0.30),
                dominance_diff=params.get('dominance_diff_db', 5.0),
                max_rsrp_diff=params.get('max_rsrp_diff_db', 5.0),
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
            )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
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

        logger.info(
            "Interference detector initialized",
            min_filtered_cells=self.params.min_filtered_cells_per_grid,
            max_rsrp_diff=self.params.max_rsrp_diff,
            k=self.params.k,
        )

    def detect(
        self,
        df: pd.DataFrame,
        data_type: str = 'measured'
    ) -> gpd.GeoDataFrame:
        """
        Find interference clusters using geohash-based spatial analysis.

        This is the main entry point for interference detection. It orchestrates
        the entire detection pipeline across all frequency bands and returns
        clustered polygon regions.

        Args:
            df: Input DataFrame with required columns (cell_name, avg_rsrp, grid, band)
            data_type: 'perceived' or 'measured'

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
                - geometry: Alpha shape polygon
        """
        execution_start = time.time()
        logger.info(
            "Interference detector started",
            data_type=data_type,
            input_records=len(df),
        )

        # Validate input (expects canonical column names: grid, avg_rsrp, cell_name, band)
        df = self._validate_input(df, data_type)

        # Log validated input details
        logger.info(
            f"Validated input: {df['band'].nunique()} bands, {df['cell_name'].nunique()} cells"
        )

        # Process each band and collect interference grids
        all_grids = []

        for band in df['band'].unique():
            grids = self._process_band(df, band, data_type)
            if not grids.empty:
                all_grids.append(grids)

        # Handle empty result
        if not all_grids:
            logger.info("No interference patterns found")
            return gpd.GeoDataFrame(
                columns=['cluster_id', 'band', 'n_grids', 'n_cells', 'cells',
                        'centroid_lat', 'centroid_lon', 'area_km2', 'avg_rsrp', 'geometry'],
                crs="EPSG:4326"
            )

        # Combine all band results
        combined_grids = pd.concat(all_grids, ignore_index=True)

        # Create polygon clusters from interference grids
        cluster_gdf = self._create_interference_polygons(combined_grids)

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
            logger.warning(f"Found null values: {null_report}. Rows will be dropped.")
            df = df.dropna(subset=required_columns)

            if df.empty:
                raise ValueError("All rows contain null values in required columns")

        return df

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
        logger.info(f"Processing band: {band}")

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
        grid_geo_data_diff = grid_geo_data_diff[
            grid_geo_data_diff['rsrp_dist_max'] <= cfg.max_rsrp_diff
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
            logger.warning(f"No interference patterns found for band {band}")
            return pd.DataFrame()

        # Log band-level results
        band_elapsed = time.time() - band_start
        n_cells = grid_geo_data_geo_filtered['cell_name'].nunique()

        logger.info(
            f"Band {band} completed",
            elapsed_time=round(band_elapsed, 2),
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
                    grid_geo_data_dominant['rsrp_diff_1_2'] >= cfg.dominance_diff
                ]
            else:
                grid_geo_data_dominant = grid_geo_data_dominant[
                    (grid_geo_data_dominant.perc_grid_events >= cfg.dominant_perc_grid_events) &
                    (grid_geo_data_dominant['rsrp_diff_1_2'] >= cfg.dominance_diff)
                ]
        else:
            grid_geo_data_dominant = grid_geo_data_dominant[
                grid_geo_data_dominant['rsrp_diff_1_2'] >= cfg.dominance_diff
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

    def _create_interference_polygons(self, grids_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Create clustered polygon regions from interference grids.

        Groups nearby interference grids using HDBSCAN and creates alpha shape
        polygons for each cluster.

        Args:
            grids_df: DataFrame with interference grids (must have 'grid', 'band', 'cell_name', 'avg_rsrp')

        Returns:
            GeoDataFrame with cluster polygons and metadata
        """
        cfg = self.params
        all_clusters = []

        # Process each band separately
        for band in grids_df['band'].unique():
            band_grids = grids_df[grids_df['band'] == band].copy()

            if len(band_grids) < cfg.hdbscan_min_cluster_size:
                logger.info(f"Band {band}: too few grids ({len(band_grids)}) for clustering")
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
            band_grids = band_grids.merge(coords_df, on='grid', how='left')

            # Cluster using HDBSCAN on unique coordinates
            unique_coords = coords_df[['latitude', 'longitude']].drop_duplicates()
            coords = unique_coords.values

            # Skip if too few points for clustering
            if len(coords) < cfg.hdbscan_min_cluster_size:
                logger.info(f"Band {band}: only {len(coords)} unique coordinates, skipping clustering")
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
                logger.info(f"Band {band}: no valid clusters found")
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

                all_clusters.append({
                    'cluster_id': f"{band}_{cluster_id}",
                    'band': band,
                    'n_grids': n_unique_grids,
                    'n_cells': len(cells_involved),
                    'cells': cells_involved,
                    'centroid_lat': cluster_data['latitude'].mean(),
                    'centroid_lon': cluster_data['longitude'].mean(),
                    'area_km2': self._calculate_area_km2(polygon),
                    'avg_rsrp': cluster_data['avg_rsrp'].mean(),
                    'geometry': polygon
                })

            n_clusters = clustered['cluster_id'].nunique()
            logger.info(f"Band {band}: {n_clusters} interference clusters created")

        if not all_clusters:
            return gpd.GeoDataFrame(
                columns=['cluster_id', 'band', 'n_grids', 'n_cells', 'cells',
                        'centroid_lat', 'centroid_lon', 'area_km2', 'avg_rsrp', 'geometry'],
                crs="EPSG:4326"
            )

        gdf = gpd.GeoDataFrame(all_clusters, crs="EPSG:4326")
        gdf = gdf.sort_values(['band', 'n_grids'], ascending=[True, False]).reset_index(drop=True)

        logger.info(f"Total interference clusters: {len(gdf)}")
        return gdf

    def _create_alpha_shape(self, coords: np.ndarray) -> Optional[Polygon]:
        """
        Create an alpha shape polygon from coordinates.

        Args:
            coords: Array of (longitude, latitude) coordinates

        Returns:
            Shapely Polygon or None if creation fails
        """
        cfg = self.params

        # Subsample if too many points
        if len(coords) > cfg.max_alphashape_points:
            indices = np.random.choice(len(coords), cfg.max_alphashape_points, replace=False)
            coords = coords[indices]

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
            logger.warning(f"Alpha shape creation failed: {e}")
            # Fallback to convex hull
            try:
                points = [Point(lon, lat) for lon, lat in coords]
                hull = gpd.GeoSeries(points).unary_union.convex_hull
                # Handle case where convex_hull returns a point or line
                if hull.geom_type in ('Point', 'MultiPoint', 'LineString'):
                    return hull.buffer(0.001)  # Small buffer to create polygon
                return hull
            except Exception:
                return None

    def _calculate_area_km2(self, geometry) -> float:
        """
        Calculate area in km² for a geometry using UTM projection.

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
            logger.warning(f"Area calculation failed: {e}")
            # Fallback approximation
            return round(
                geometry.area * KM_PER_DEGREE_LAT * KM_PER_DEGREE_LAT *
                np.cos(np.radians(geometry.centroid.y)),
                3
            )


def detect_interference(
    df: pd.DataFrame,
    data_type: str = 'measured',
    params: Optional[InterferenceParams] = None,
) -> gpd.GeoDataFrame:
    """
    Convenience function to detect interference clusters.

    Args:
        df: Input DataFrame with cell coverage data
        data_type: 'perceived' or 'measured'
        params: Optional detection parameters

    Returns:
        GeoDataFrame with interference cluster polygons
    """
    detector = InterferenceDetector(params)
    return detector.detect(df, data_type)
