"""
Coverage gap detection using cell hull clustering and geohash-based gap analysis.

This module identifies significant coverage gaps by:
1. Loading cell convex hulls
2. Clustering cell hulls to group nearby coverage areas
3. Creating convex hull for each cell cluster
4. Finding gap polygons (uncovered areas within cluster hulls)
5. Getting all possible geohashes in gap polygons
6. Applying k-ring density filtering and clustering

Also supports low coverage detection (band-specific):
1. Find single-server regions (only 1 cell provides coverage)
2. Filter by RSRP threshold
3. Apply same k-ring and clustering logic

Based on notebook: tilt-optimisation-low-coverage.ipynb
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, List, Dict
import json

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
import hdbscan
import alphashape
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils import geohash as geohash_utils
from ran_optimizer.core.environment_classifier import EnvironmentClassifier

logger = get_logger(__name__)

# Geographic constants
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0
KM_PER_DEGREE_LAT = 111.32

# Default geohash precision for coverage gap detection
DEFAULT_GEOHASH_PRECISION = 7

# Maximum grid points to process for a single polygon (performance limit)
MAX_POLYGON_GRID_POINTS = 10000


@dataclass
class CoverageGapParams:
    """
    Configuration parameters for coverage gap detection.

    Attributes:
        cell_cluster_eps_km: DBSCAN epsilon for clustering cell hulls (km)
        cell_cluster_min_samples: Minimum cells per cluster
        k_ring_steps: Number of neighbor steps for k-ring density (3 = 7x7 = 49 cells)
        min_missing_neighbors: Minimum missing neighbors in k-ring to be considered a gap
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN on gap geohashes
        alpha_shape_alpha: Alpha parameter for concave hull (None = auto-determine)
        k_nearest_cells: Number of nearest cells to find for each gap cluster
        max_alphashape_points: Maximum points for alphashape (subsample if exceeded)

    Example:
        >>> params = CoverageGapParams(cell_cluster_eps_km=5.0, k_ring_steps=3)
        >>> params = CoverageGapParams.from_config(Path("config/coverage_gaps.json"))
    """
    cell_cluster_eps_km: float = 5.0  # Cluster cells within 5km
    cell_cluster_min_samples: int = 3  # At least 3 cells per cluster
    k_ring_steps: int = 3
    min_missing_neighbors: int = 40  # Out of 49 for k=3
    hdbscan_min_cluster_size: int = 10
    alpha_shape_alpha: Optional[float] = None  # None = auto
    k_nearest_cells: int = 5
    max_alphashape_points: int = 5000  # Subsample for performance

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None, environment: str = "default"):
        """
        Load parameters from JSON configuration file.

        Args:
            config_path: Path to coverage_gaps.json config file
            environment: Environment type for overrides (urban/rural/suburban)

        Returns:
            CoverageGapParams instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid

        Example:
            >>> params = CoverageGapParams.from_config(
            ...     Path("config/coverage_gaps.json"),
            ...     environment="urban"
            ... )
        """
        if config_path is None:
            config_path = Path("config/coverage_gaps.json")

        if not config_path.exists():
            logger.warning(
                "config_file_not_found",
                path=str(config_path),
                using_defaults=True
            )
            return cls()

        try:
            with open(config_path) as f:
                config = json.load(f)

            # Start with base configuration
            base_params = config.get("coverage_gap_detection", {})

            # Apply environment-specific overrides if available
            if environment != "default":
                overrides = config.get("environment_overrides", {}).get(environment, {})
                # Only apply overrides that are valid for CoverageGapParams
                valid_overrides = {k: v for k, v in overrides.items()
                                 if k in ['cell_cluster_eps_km', 'cell_cluster_min_samples',
                                         'k_ring_steps', 'min_missing_neighbors',
                                         'hdbscan_min_cluster_size', 'alpha_shape_alpha',
                                         'k_nearest_cells', 'max_alphashape_points']}
                base_params.update(valid_overrides)
                logger.info(
                    "applied_environment_overrides",
                    environment=environment,
                    overrides=list(valid_overrides.keys())
                )

            return cls(**base_params)

        except Exception as e:
            logger.error(
                "failed_to_load_config",
                path=str(config_path),
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Failed to load coverage gap config: {e}") from e


@dataclass
class LowCoverageParams:
    """
    Configuration parameters for low coverage detection (band-specific).

    Low coverage is defined as single-server areas where:
    - Only 1 cell provides coverage (no overlap from other cells on same band)
    - Average RSRP ≤ threshold (e.g., -115 dBm)

    Attributes:
        rsrp_threshold_dbm: RSRP threshold for low coverage (values ≤ threshold are low coverage)
        k_ring_steps: Number of neighbor steps for k-ring density
        min_missing_neighbors: Minimum missing neighbors in k-ring to be considered low coverage
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN
        alpha_shape_alpha: Alpha parameter for concave hull (None = auto-determine)
        max_alphashape_points: Maximum points for alphashape (subsample if exceeded)

    Example:
        >>> params = LowCoverageParams(rsrp_threshold_dbm=-115, k_ring_steps=3)
        >>> params = LowCoverageParams.from_config(Path("config/coverage_gaps.json"))
    """
    rsrp_threshold_dbm: float = -115
    k_ring_steps: int = 3
    min_missing_neighbors: int = 40
    hdbscan_min_cluster_size: int = 10
    alpha_shape_alpha: Optional[float] = None
    max_alphashape_points: int = 5000

    # Data quality parameters
    rsrp_min_dbm: float = -140
    rsrp_max_dbm: float = -30
    min_sample_count: int = 3  # Minimum samples per geohash (3 = 75th percentile in typical data)

    # Severity scoring weights
    severity_weight_area: float = 0.30
    severity_weight_n_points: float = 0.25
    severity_weight_rsrp: float = 0.25
    severity_weight_serving_cells: float = 0.20

    # Severity normalization
    severity_area_max_km2: float = 5.0
    severity_rsrp_max_db: float = 25.0

    # Severity thresholds
    severity_threshold_critical: float = 0.80
    severity_threshold_high: float = 0.60
    severity_threshold_medium: float = 0.40
    severity_threshold_low: float = 0.20

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None, environment: str = "default"):
        """
        Load parameters from JSON configuration file.

        Args:
            config_path: Path to coverage_gaps.json config file
            environment: Environment type for overrides (urban/rural/suburban)

        Returns:
            LowCoverageParams instance with loaded configuration

        Example:
            >>> params = LowCoverageParams.from_config(
            ...     Path("config/coverage_gaps.json"),
            ...     environment="urban"
            ... )
        """
        if config_path is None:
            config_path = Path("config/coverage_gaps.json")

        if not config_path.exists():
            logger.warning(
                "config_file_not_found",
                path=str(config_path),
                using_defaults=True
            )
            return cls()

        try:
            with open(config_path) as f:
                config = json.load(f)

            # Start with base configuration
            base_params = config.get("low_coverage_detection", {}).copy()

            # Flatten nested config sections
            if 'data_quality' in base_params:
                dq = base_params.pop('data_quality')
                base_params['rsrp_min_dbm'] = dq.get('rsrp_min_dbm', -140)
                base_params['rsrp_max_dbm'] = dq.get('rsrp_max_dbm', -30)
                base_params['min_sample_count'] = dq.get('min_sample_count', 10)

            if 'severity_weights' in base_params:
                sw = base_params.pop('severity_weights')
                base_params['severity_weight_area'] = sw.get('area_km2', 0.30)
                base_params['severity_weight_n_points'] = sw.get('n_points', 0.25)
                base_params['severity_weight_rsrp'] = sw.get('rsrp_severity', 0.25)
                base_params['severity_weight_serving_cells'] = sw.get('serving_cell_count', 0.20)

            if 'severity_normalization' in base_params:
                sn = base_params.pop('severity_normalization')
                base_params['severity_area_max_km2'] = sn.get('area_max_km2', 5.0)
                base_params['severity_rsrp_max_db'] = sn.get('rsrp_severity_max_db', 25.0)

            if 'severity_thresholds' in base_params:
                st = base_params.pop('severity_thresholds')
                base_params['severity_threshold_critical'] = st.get('critical', 0.80)
                base_params['severity_threshold_high'] = st.get('high', 0.60)
                base_params['severity_threshold_medium'] = st.get('medium', 0.40)
                base_params['severity_threshold_low'] = st.get('low', 0.20)

            # Apply environment-specific overrides if available
            if environment != "default":
                overrides = config.get("environment_overrides", {}).get(environment, {})
                # Only apply low coverage related overrides
                low_cov_overrides = {k: v for k, v in overrides.items()
                                    if k in ['rsrp_threshold_dbm', 'k_ring_steps',
                                            'min_missing_neighbors', 'hdbscan_min_cluster_size',
                                            'alpha_shape_alpha', 'max_alphashape_points']}
                base_params.update(low_cov_overrides)
                logger.info(
                    "applied_environment_overrides_low_coverage",
                    environment=environment,
                    overrides=list(low_cov_overrides.keys())
                )

            # Filter to only valid dataclass fields
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered_params = {k: v for k, v in base_params.items() if k in valid_fields}

            return cls(**filtered_params)

        except Exception as e:
            logger.error(
                "failed_to_load_config",
                path=str(config_path),
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Failed to load low coverage config: {e}") from e


class GapDetectorBase:
    """
    Base class with shared methods for gap detection.

    Both CoverageGapDetector and LowCoverageDetector share:
    - Geohash extraction from polygons
    - K-ring density filtering
    - HDBSCAN clustering
    - Alpha shape polygon creation
    """

    def _geohashes_in_polygons(self, polygons: List[Polygon], precision: int = DEFAULT_GEOHASH_PRECISION) -> List[str]:
        """
        Get all possible geohashes within gap polygons.

        Args:
            polygons: List of gap polygons
            precision: Geohash precision (default 7)

        Returns:
            List of geohash strings covering the gap polygons
        """
        all_geohashes = set()

        for polygon in polygons:
            # Get bounding box
            minx, miny, maxx, maxy = polygon.bounds

            # Skip if polygon has no area
            if minx == maxx or miny == maxy:
                logger.warning("skipping_zero_area_polygon", bounds=(minx, miny, maxx, maxy))
                continue

            # Get cell dimensions
            lat_height, lon_width = geohash_utils.get_precision_dimensions(precision)
            # Convert km to degrees (approximate)
            lat_step = lat_height / KM_PER_DEGREE_LAT
            lon_step = lon_width / (KM_PER_DEGREE_LAT * np.cos(np.radians((miny + maxy) / 2)))

            # Ensure positive step values
            if lat_step <= 0 or lon_step <= 0 or not np.isfinite(lat_step) or not np.isfinite(lon_step):
                logger.warning("invalid_step_size", lat_step=lat_step, lon_step=lon_step)
                continue

            # Log debugging info for large polygons
            lat_range = maxy - miny
            lon_range = maxx - minx
            if lat_range / lat_step > MAX_POLYGON_GRID_POINTS or lon_range / lon_step > MAX_POLYGON_GRID_POINTS:
                logger.warning(
                    "polygon_too_large_for_geohash_sampling",
                    lat_range=lat_range,
                    lon_range=lon_range,
                    lat_step=lat_step,
                    lon_step=lon_step,
                    estimated_points=int((lat_range / lat_step) * (lon_range / lon_step))
                )
                continue

            # Generate grid of points within bounding box
            try:
                lats = np.arange(miny, maxy, lat_step)
                lons = np.arange(minx, maxx, lon_step)
            except ValueError as e:
                logger.error(
                    "arange_failed",
                    error=str(e),
                    miny=miny,
                    maxy=maxy,
                    lat_step=lat_step,
                    minx=minx,
                    maxx=maxx,
                    lon_step=lon_step
                )
                continue

            # Vectorized point-in-polygon check using meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            points_flat = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

            # Use shapely's prepared geometry for faster contains checks
            prepared_polygon = prep(polygon)

            # Check all points at once using vectorized approach
            for lon, lat in points_flat:
                if prepared_polygon.contains(Point(lon, lat)):
                    gh = geohash_utils.encode(lat, lon, precision=precision)
                    all_geohashes.add(gh)

        logger.info("geohashes_in_polygons", count=len(all_geohashes))
        return list(all_geohashes)

    def _compute_kring_density(self, gap_geohashes: Set[str], k_ring_steps: int, min_missing_neighbors: int) -> pd.DataFrame:
        """
        Compute k-ring density and filter to dense gaps.

        Args:
            gap_geohashes: Set of gap geohash strings
            k_ring_steps: Number of neighbor steps for k-ring
            min_missing_neighbors: Minimum missing neighbors to be considered dense

        Returns:
            DataFrame with dense gap geohashes and their coordinates
        """
        records = []
        total = len(gap_geohashes)

        logger.info("computing_kring_density", total_geohashes=total, k=k_ring_steps)

        for idx, gh in enumerate(gap_geohashes):
            if idx > 0 and idx % 10000 == 0:
                logger.info("kring_progress", processed=idx, total=total, percent=f"{100*idx/total:.1f}%")

            # Get k-ring neighbors
            kring_set = geohash_utils.kring(gh, k_ring_steps)

            # Count how many are also gaps
            missing_count = len(kring_set & gap_geohashes) - 1  # Exclude self

            # Only keep if dense enough
            if missing_count >= min_missing_neighbors:
                lat, lon = geohash_utils.decode(gh)
                records.append({
                    'grid': gh,
                    'latitude': lat,
                    'longitude': lon,
                    f'missing_within_{k_ring_steps}_steps': missing_count
                })

        df = pd.DataFrame(records)

        if len(df) > 0:
            logger.info(
                "kring_density_computed",
                k=k_ring_steps,
                dense_gaps=len(df),
                mean_missing=df[f'missing_within_{k_ring_steps}_steps'].mean()
            )
        else:
            logger.info("no_dense_gaps_after_kring_filter")

        return df

    def _cluster_hdbscan(self, dense_gaps: pd.DataFrame, min_cluster_size: int) -> pd.DataFrame:
        """
        Cluster dense gaps using HDBSCAN.

        Args:
            dense_gaps: DataFrame with dense gap geohashes
            min_cluster_size: Minimum cluster size for HDBSCAN

        Returns:
            DataFrame with 'cluster_id' column added (noise points removed)
        """
        coords = dense_gaps[['latitude', 'longitude']].values

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean'
        )

        labels = clusterer.fit_predict(coords)

        dense_gaps = dense_gaps.copy()
        dense_gaps['cluster_id'] = labels

        # Remove noise points (label = -1)
        clustered = dense_gaps[dense_gaps['cluster_id'] >= 0].copy()

        n_clusters = len(clustered['cluster_id'].unique())
        n_noise = list(labels).count(-1)

        logger.info(
            "hdbscan_clustering_complete",
            clusters=n_clusters,
            clustered_points=len(clustered),
            noise_points=n_noise
        )

        return clustered

    def _create_cluster_polygons(self, clustered_gaps: pd.DataFrame, alpha_shape_alpha: Optional[float], max_alphashape_points: int) -> gpd.GeoDataFrame:
        """
        Create alpha shape polygons for each gap cluster.

        Args:
            clustered_gaps: DataFrame with clustered gap geohashes
            alpha_shape_alpha: Alpha parameter for concave hull (None = auto)
            max_alphashape_points: Maximum points for alphashape (subsample if exceeded)

        Returns:
            GeoDataFrame with cluster polygons and metadata
        """
        cluster_records = []

        for cluster_id in clustered_gaps['cluster_id'].unique():
            cluster_points = clustered_gaps[clustered_gaps['cluster_id'] == cluster_id]
            coords = cluster_points[['longitude', 'latitude']].values

            # Subsample if too many points for alphashape
            if len(coords) > max_alphashape_points:
                logger.info(
                    "subsampling_for_alphashape",
                    cluster_id=cluster_id,
                    original=len(coords),
                    subsample=max_alphashape_points
                )
                indices = np.random.choice(len(coords), max_alphashape_points, replace=False)
                coords = coords[indices]

            try:
                # Try alpha shape
                if alpha_shape_alpha is None:
                    alpha_shape = alphashape.alphashape(coords.tolist())
                else:
                    alpha_shape = alphashape.alphashape(coords.tolist(), float(alpha_shape_alpha))

                if alpha_shape.is_empty or not alpha_shape.is_valid:
                    # Fallback to convex hull
                    points = [Point(lon, lat) for lon, lat in coords]
                    alpha_shape = gpd.GeoSeries(points).unary_union.convex_hull
                    logger.info("using_convex_hull_fallback", cluster_id=cluster_id)

                alpha_shape = unary_union([alpha_shape]).buffer(0)

            except Exception as e:
                logger.warning("alphashape_failed", cluster_id=cluster_id, error=str(e))
                # Fallback to convex hull
                points = [Point(lon, lat) for lon, lat in coords]
                alpha_shape = gpd.GeoSeries(points).unary_union.convex_hull

            # Calculate area in km² using projected CRS
            area_km2 = self._calculate_area_km2(alpha_shape)

            cluster_records.append({
                'cluster_id': int(cluster_id),
                'n_points': len(cluster_points),
                'centroid_lat': cluster_points['latitude'].mean(),
                'centroid_lon': cluster_points['longitude'].mean(),
                'area_km2': area_km2,
                'geometry': alpha_shape
            })

        gdf = gpd.GeoDataFrame(cluster_records, crs="EPSG:4326")
        logger.info("cluster_polygons_created", clusters=len(gdf))

        return gdf

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
            # Create a GeoDataFrame for projection
            gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

            # Get centroid to determine UTM zone
            centroid = geometry.centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            hemisphere = 'north' if centroid.y >= 0 else 'south'
            epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone

            # Project to UTM and calculate area
            gdf_projected = gdf.to_crs(epsg=epsg_code)
            area_m2 = gdf_projected.geometry.iloc[0].area
            area_km2 = area_m2 / 1_000_000

            return round(area_km2, 3)

        except Exception as e:
            logger.warning("area_calculation_failed", error=str(e))
            # Fallback: approximate using lat/lon (rough estimate)
            return round(geometry.area * KM_PER_DEGREE_LAT * KM_PER_DEGREE_LAT * np.cos(np.radians(geometry.centroid.y)), 3)


class CoverageGapDetector(GapDetectorBase):
    """
    Detects significant coverage gaps using cell hull clustering approach.

    This class implements the notebook approach:
    1. Cluster cell hulls to group nearby coverage areas
    2. Create convex hull for each cell cluster
    3. Find gap polygons (uncovered areas) within each cluster hull
    4. Get all possible geohashes in gap polygons
    5. Apply k-ring density filtering
    6. Cluster gap geohashes with HDBSCAN and create alpha shape polygons

    Example:
        >>> detector = CoverageGapDetector()
        >>> gap_clusters = detector.detect(hulls_gdf)
        >>> print(f"Found {len(gap_clusters)} significant coverage gaps")
    """

    def __init__(self, params: Optional[CoverageGapParams] = None, boundary_shapefile: Optional[str] = None):
        """
        Initialize coverage gap detector.

        Args:
            params: Configuration parameters (uses defaults if not provided)
            boundary_shapefile: Optional path to boundary shapefile to clip polygons
        """
        self.params = params or CoverageGapParams()
        self.boundary_gdf = None

        if boundary_shapefile:
            try:
                self.boundary_gdf = gpd.read_file(boundary_shapefile)
                logger.info("boundary_shapefile_loaded", path=boundary_shapefile, features=len(self.boundary_gdf))
            except Exception as e:
                logger.warning("boundary_shapefile_load_failed", path=boundary_shapefile, error=str(e))

        logger.info(
            "coverage_gap_detector_initialized",
            cell_cluster_eps_km=self.params.cell_cluster_eps_km,
            k_ring_steps=self.params.k_ring_steps,
            min_missing_neighbors=self.params.min_missing_neighbors,
            has_boundary=self.boundary_gdf is not None
        )

    def detect(self, hulls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Detect coverage gap clusters from cell convex hulls.

        Args:
            hulls: GeoDataFrame with cell convex hulls (must have 'geometry' column)

        Returns:
            GeoDataFrame with gap cluster polygons and metadata

        Example:
            >>> hulls_gdf = load_cell_hulls("data/cell_hulls.geojson")
            >>> gap_clusters = detector.detect(hulls_gdf)
            >>> print(gap_clusters[['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon']])
        """
        logger.info("starting_coverage_gap_detection", cell_hulls=len(hulls))

        # Handle empty input
        if len(hulls) == 0:
            logger.warning("empty_hulls_input")
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry'])

        # Step 1: Cluster cell hulls
        clustered_hulls = self._cluster_cell_hulls(hulls)

        if len(clustered_hulls) == 0:
            logger.warning("no_cell_clusters_found")
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry'])

        # Step 2: For each cell cluster, find coverage gaps
        all_gap_geohashes = []

        for cluster_id in clustered_hulls['cell_cluster'].unique():
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_hulls = clustered_hulls[clustered_hulls['cell_cluster'] == cluster_id]
            logger.info("processing_cell_cluster", cluster_id=cluster_id, cells=len(cluster_hulls))

            # Step 3: Create convex hull for this cell cluster
            cluster_hull = self._create_cluster_hull(cluster_hulls)

            # Step 4: Find gap polygons (uncovered areas)
            gap_polygons = self._find_gap_polygons(cluster_hull, cluster_hulls)

            if len(gap_polygons) == 0:
                logger.info("no_gaps_in_cluster", cluster_id=cluster_id)
                continue

            # Step 5: Get all possible geohashes in gap polygons
            gap_geohashes = self._geohashes_in_polygons(gap_polygons)

            if len(gap_geohashes) > 0:
                all_gap_geohashes.extend(gap_geohashes)
                logger.info("gap_geohashes_found", cluster_id=cluster_id, count=len(gap_geohashes))

        if len(all_gap_geohashes) == 0:
            logger.info("no_gap_geohashes_found")
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry'])

        # Step 6: Apply k-ring density filtering
        dense_gaps = self._compute_kring_density(
            set(all_gap_geohashes),
            self.params.k_ring_steps,
            self.params.min_missing_neighbors
        )

        if len(dense_gaps) == 0:
            logger.info("no_dense_gaps_found")
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry'])

        # Step 7: Cluster dense gaps with HDBSCAN
        clustered_gaps = self._cluster_hdbscan(dense_gaps, self.params.hdbscan_min_cluster_size)

        if len(clustered_gaps) == 0:
            logger.info("no_gap_clusters_found")
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry'])

        # Step 8: Create alpha shape polygons
        cluster_polygons = self._create_cluster_polygons(
            clustered_gaps,
            self.params.alpha_shape_alpha,
            self.params.max_alphashape_points
        )

        # Step 9: Clip to boundary if provided
        if self.boundary_gdf is not None and len(cluster_polygons) > 0:
            cluster_polygons = self._clip_to_boundary(cluster_polygons)

        logger.info("coverage_gaps_detected", num_clusters=len(cluster_polygons))
        return cluster_polygons

    def _clip_to_boundary(self, cluster_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clip cluster polygons to the boundary shapefile.

        Args:
            cluster_polygons: GeoDataFrame with cluster polygons

        Returns:
            GeoDataFrame with polygons clipped to boundary
        """
        if self.boundary_gdf is None:
            return cluster_polygons

        # Ensure same CRS
        if cluster_polygons.crs != self.boundary_gdf.crs:
            boundary_aligned = self.boundary_gdf.to_crs(cluster_polygons.crs)
        else:
            boundary_aligned = self.boundary_gdf

        # Create union of all boundary polygons
        boundary_union = boundary_aligned.geometry.unary_union

        # Clip each polygon to the boundary
        clipped_records = []
        for _, row in cluster_polygons.iterrows():
            clipped_geom = row['geometry'].intersection(boundary_union)

            # Skip if clipped polygon is empty or too small
            if clipped_geom.is_empty:
                continue

            # Recalculate area after clipping
            area_km2 = self._calculate_area_km2(clipped_geom)

            # Recalculate centroid
            centroid = clipped_geom.centroid

            clipped_records.append({
                'cluster_id': row['cluster_id'],
                'n_points': row['n_points'],
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x,
                'area_km2': area_km2,
                'geometry': clipped_geom
            })

        if len(clipped_records) == 0:
            logger.info(
                "boundary_clipping_removed_all",
                original_clusters=len(cluster_polygons)
            )
            return gpd.GeoDataFrame(
                columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'area_km2', 'geometry'],
                crs=cluster_polygons.crs
            )

        clipped_gdf = gpd.GeoDataFrame(clipped_records, crs=cluster_polygons.crs)

        logger.info(
            "boundary_clipping_complete",
            original_clusters=len(cluster_polygons),
            clipped_clusters=len(clipped_gdf),
            removed_clusters=len(cluster_polygons) - len(clipped_gdf)
        )

        return clipped_gdf

    def _cluster_cell_hulls(self, hulls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Cluster cell hulls using DBSCAN on centroid locations.

        Args:
            hulls: GeoDataFrame with cell hull geometries

        Returns:
            GeoDataFrame with 'cell_cluster' column added
        """
        # Get centroids using projected CRS for accuracy
        hulls = hulls.copy()

        # Project to UTM for accurate centroid calculation, then transform back
        if hulls.crs and not hulls.crs.is_projected:
            # Estimate UTM zone from data centroid
            bounds = hulls.total_bounds  # [minx, miny, maxx, maxy]
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            utm_zone = int((center_lon + 180) / 6) + 1
            epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone

            # Project, calculate centroid, transform back to WGS84
            hulls_projected = hulls.to_crs(f'EPSG:{epsg_code}')
            centroids_projected = hulls_projected.geometry.centroid
            centroids_gdf = gpd.GeoDataFrame(geometry=centroids_projected, crs=f'EPSG:{epsg_code}')
            centroids = centroids_gdf.to_crs('EPSG:4326').geometry
        else:
            centroids = hulls.geometry.centroid

        hulls['centroid_lat'] = centroids.y
        hulls['centroid_lon'] = centroids.x

        # Cluster using DBSCAN on lat/lon coordinates
        coords = hulls[['centroid_lat', 'centroid_lon']].values

        # Use BallTree with haversine metric for geographic clustering
        # Convert eps from km to radians for haversine
        eps_radians = self.params.cell_cluster_eps_km / EARTH_RADIUS_KM

        clusterer = DBSCAN(
            eps=eps_radians,
            min_samples=self.params.cell_cluster_min_samples,
            metric='haversine'
        )

        # Convert to radians for haversine
        coords_radians = np.radians(coords)
        labels = clusterer.fit_predict(coords_radians)

        hulls['cell_cluster'] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(
            "cell_hulls_clustered",
            n_clusters=n_clusters,
            n_noise=n_noise,
            total_cells=len(hulls)
        )

        return hulls

    def _create_cluster_hull(self, cluster_hulls: gpd.GeoDataFrame) -> Polygon:
        """
        Create convex hull for a cluster of cell hulls.

        Args:
            cluster_hulls: GeoDataFrame with cell hulls in the cluster

        Returns:
            Convex hull polygon encompassing all cell hulls
        """
        # Union all cell hulls and create convex hull
        hull_union = unary_union(cluster_hulls.geometry)
        cluster_hull = hull_union.convex_hull

        area_km2 = cluster_hull.area * KM_PER_DEGREE_LAT * KM_PER_DEGREE_LAT  # Approximate conversion to km²
        logger.info("cluster_hull_created", area_km2=area_km2)

        return cluster_hull

    def _find_gap_polygons(self, cluster_hull: Polygon, cell_hulls: gpd.GeoDataFrame) -> List[Polygon]:
        """
        Find gap polygons (uncovered areas) within cluster hull.

        Args:
            cluster_hull: Convex hull of the cell cluster
            cell_hulls: GeoDataFrame with individual cell hulls

        Returns:
            List of gap polygons (areas with no cell coverage)
        """
        # Union all cell coverage
        coverage_union = unary_union(cell_hulls.geometry)

        # Find gaps: cluster hull minus coverage
        gaps = cluster_hull.difference(coverage_union)

        # Convert to list of polygons
        gap_polygons = []
        if isinstance(gaps, Polygon):
            gap_polygons = [gaps]
        elif isinstance(gaps, MultiPolygon):
            gap_polygons = list(gaps.geoms)

        logger.info("gap_polygons_found", count=len(gap_polygons))
        return gap_polygons

    def detect_per_band(
        self,
        hulls: gpd.GeoDataFrame,
        gis_data: pd.DataFrame,
        bands: Optional[List[str]] = None,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Detect coverage gaps per band.

        For each band, finds areas where cells of that band don't provide coverage,
        even if other bands might cover the area. This complements low coverage
        detection by showing where a band has NO coverage at all.

        Args:
            hulls: GeoDataFrame with cell convex hulls (must have 'geometry' column)
            gis_data: GIS data with cell_name and band columns
            bands: Optional list of specific bands to process (None = all bands)

        Returns:
            Dict mapping band names to GeoDataFrames of gap clusters
            e.g., {'L800': gdf, 'L1800': gdf, 'L2100': gdf}

        Example:
            >>> detector = CoverageGapDetector()
            >>> gaps_by_band = detector.detect_per_band(hulls_gdf, gis_df)
            >>> for band, gaps in gaps_by_band.items():
            ...     print(f"{band}: {len(gaps)} gap clusters")
        """
        logger.info("starting_per_band_coverage_gap_detection", cell_hulls=len(hulls))

        # Step 1: Ensure hulls have band information
        hulls_with_band = self._add_band_info_for_gaps(hulls, gis_data)

        # Step 2: Get available bands
        available_bands = [b for b in hulls_with_band['band'].unique() if pd.notna(b)]
        if bands is not None:
            # Normalise requested bands for comparison
            bands_normalised = [self._normalise_band(b) for b in bands]
            available_bands = [b for b in available_bands if b in bands_normalised]

        logger.info("detecting_no_coverage_per_band", bands=available_bands)

        # Step 3: Process each band
        band_results = {}
        for band in available_bands:
            logger.info("processing_band_no_coverage", band=band)

            # Filter hulls to this band only
            band_hulls = hulls_with_band[hulls_with_band['band'] == band].copy()

            if len(band_hulls) < self.params.cell_cluster_min_samples:
                logger.info(
                    "skipping_band_insufficient_cells",
                    band=band,
                    cells=len(band_hulls),
                    min_required=self.params.cell_cluster_min_samples
                )
                continue

            # Run gap detection for this band
            band_gaps = self._detect_band_gaps(band_hulls, band)

            if len(band_gaps) > 0:
                band_results[band] = band_gaps
                logger.info(
                    "band_no_coverage_detected",
                    band=band,
                    clusters=len(band_gaps),
                    total_area_km2=band_gaps['area_km2'].sum() if 'area_km2' in band_gaps.columns else None
                )

        logger.info(
            "per_band_coverage_gap_detection_complete",
            bands_with_gaps=len(band_results),
            total_clusters=sum(len(gdf) for gdf in band_results.values())
        )
        return band_results

    def _detect_band_gaps(self, band_hulls: gpd.GeoDataFrame, band: str) -> gpd.GeoDataFrame:
        """
        Detect coverage gaps for a single band.

        Uses the same logic as the main detect() method but on band-filtered hulls.

        Args:
            band_hulls: GeoDataFrame with cell hulls for a single band
            band: Band name for logging

        Returns:
            GeoDataFrame with gap cluster polygons for this band
        """
        empty_result = gpd.GeoDataFrame(
            columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'area_km2', 'band', 'geometry']
        )

        if len(band_hulls) == 0:
            return empty_result

        # Step 1: Cluster cell hulls for this band
        clustered_hulls = self._cluster_cell_hulls(band_hulls)

        if len(clustered_hulls) == 0:
            return empty_result

        # Step 2: For each cell cluster, find coverage gaps
        all_gap_geohashes = []

        for cluster_id in clustered_hulls['cell_cluster'].unique():
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_hulls = clustered_hulls[clustered_hulls['cell_cluster'] == cluster_id]

            # Create convex hull for this cell cluster
            cluster_hull = self._create_cluster_hull(cluster_hulls)

            # Find gap polygons (uncovered areas)
            gap_polygons = self._find_gap_polygons(cluster_hull, cluster_hulls)

            if len(gap_polygons) == 0:
                continue

            # Get all possible geohashes in gap polygons
            gap_geohashes = self._geohashes_in_polygons(gap_polygons)

            if len(gap_geohashes) > 0:
                all_gap_geohashes.extend(gap_geohashes)

        if len(all_gap_geohashes) == 0:
            return empty_result

        # Step 3: Apply k-ring density filtering
        dense_gaps = self._compute_kring_density(
            set(all_gap_geohashes),
            self.params.k_ring_steps,
            self.params.min_missing_neighbors
        )

        if len(dense_gaps) == 0:
            return empty_result

        # Step 4: Cluster dense gaps with HDBSCAN
        clustered_gaps = self._cluster_hdbscan(dense_gaps, self.params.hdbscan_min_cluster_size)

        if len(clustered_gaps) == 0:
            return empty_result

        # Step 5: Create alpha shape polygons
        cluster_polygons = self._create_cluster_polygons(
            clustered_gaps,
            self.params.alpha_shape_alpha,
            self.params.max_alphashape_points
        )

        # Step 6: Clip to boundary if provided
        if self.boundary_gdf is not None and len(cluster_polygons) > 0:
            cluster_polygons = self._clip_to_boundary(cluster_polygons)

        # Add band column
        if len(cluster_polygons) > 0:
            cluster_polygons['band'] = band

        return cluster_polygons

    def _add_band_info_for_gaps(
        self,
        hulls: gpd.GeoDataFrame,
        gis_data: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """Add band information to hulls for per-band gap detection."""
        if 'band' in hulls.columns:
            hulls = hulls.copy()
            hulls['band'] = hulls['band'].apply(self._normalise_band)
            return hulls

        if gis_data is None:
            raise ValueError("hulls missing 'band' column and no gis_data provided")

        # Validate required columns
        if 'band' not in gis_data.columns:
            raise ValueError(f"gis_data missing 'band' column. Available: {list(gis_data.columns)}")

        if 'cell_name' not in hulls.columns:
            raise ValueError(f"hulls missing 'cell_name' column. Available: {list(hulls.columns)}")

        if 'cell_name' not in gis_data.columns:
            raise ValueError(f"gis_data missing 'cell_name' column. Available: {list(gis_data.columns)}")

        # Create band mapping
        band_mapping = gis_data[['cell_name', 'band']].drop_duplicates('cell_name').copy()
        band_mapping['cell_name'] = band_mapping['cell_name'].astype(str)

        hulls_copy = hulls.copy()
        hulls_copy['cell_name'] = hulls_copy['cell_name'].astype(str)

        # Join with GIS data to get band
        hulls_with_band = hulls_copy.merge(band_mapping, on='cell_name', how='left')

        # Normalise band values
        hulls_with_band['band'] = hulls_with_band['band'].apply(self._normalise_band)

        missing_band = hulls_with_band['band'].isna().sum()
        if missing_band > 0:
            logger.warning("cells_missing_band_info", count=missing_band, total=len(hulls_with_band))

        return hulls_with_band

    @staticmethod
    def _normalise_band(band_value) -> str:
        """
        Normalise band value to 'L800' string format.

        Handles: 'L800', '800', 800, 800.0 -> 'L800'
        """
        if pd.isna(band_value):
            return None

        band_str = str(band_value).upper().strip()

        # Remove .0 suffix if present
        if band_str.endswith('.0'):
            band_str = band_str[:-2]

        # Add 'L' prefix if not present
        if not band_str.startswith('L'):
            band_str = 'L' + band_str

        return band_str


class LowCoverageDetector(GapDetectorBase):
    """
    Detects low coverage areas on a per-band basis.

    Low coverage is defined as single-server regions where:
    - Only 1 cell provides coverage (no overlap from other cells on same band)
    - Average RSRP ≤ threshold (e.g., -115 dBm)

    This detector processes each band separately and returns results keyed by band.

    Example:
        >>> detector = LowCoverageDetector()
        >>> low_cov_by_band = detector.detect(hulls_gdf, grid_data, gis_data)
        >>> print(f"Found low coverage in {len(low_cov_by_band)} bands")
    """

    def __init__(self, params: Optional[LowCoverageParams] = None, boundary_shapefile: Optional[str] = None):
        """
        Initialize low coverage detector.

        Args:
            params: Configuration parameters (uses defaults if not provided)
            boundary_shapefile: Optional path to boundary shapefile to filter out offshore gaps
        """
        self.params = params or LowCoverageParams()
        self.boundary_gdf = None

        if boundary_shapefile:
            try:
                self.boundary_gdf = gpd.read_file(boundary_shapefile)
                logger.info("boundary_shapefile_loaded", path=boundary_shapefile, features=len(self.boundary_gdf))
            except Exception as e:
                logger.warning("boundary_shapefile_load_failed", path=boundary_shapefile, error=str(e))

        logger.info(
            "low_coverage_detector_initialized",
            rsrp_threshold_dbm=self.params.rsrp_threshold_dbm,
            k_ring_steps=self.params.k_ring_steps,
            min_missing_neighbors=self.params.min_missing_neighbors,
            has_boundary=self.boundary_gdf is not None
        )

    def _build_environment_map(
        self,
        env_df: Optional[pd.DataFrame],
        gis_data: Optional[pd.DataFrame],
    ) -> Dict[str, str]:
        """
        Build cell-to-environment mapping from env_df or auto-generate from gis_data.

        Priority:
        1. Use env_df if provided and valid
        2. Auto-generate from gis_data using EnvironmentClassifier if env_df not provided
        3. Return empty dict if neither available (will use default SUBURBAN threshold)

        Args:
            env_df: Optional pre-computed environment classification with columns:
                    - cell_name: Cell identifier (string)
                    - environment: 'URBAN', 'SUBURBAN', or 'RURAL'
            gis_data: Optional GIS data with cell_name, latitude, longitude columns

        Returns:
            Dict mapping cell_name -> environment (uppercase)
            e.g., {'CORK001': 'URBAN', 'CORK002': 'RURAL'}
        """
        # Case 1: env_df provided - validate and use it
        if env_df is not None and len(env_df) > 0:
            # Validate required columns
            required_cols = ['cell_name', 'environment']
            missing_cols = [c for c in required_cols if c not in env_df.columns]
            if missing_cols:
                logger.warning(
                    "env_df_missing_columns",
                    missing=missing_cols,
                    available=list(env_df.columns),
                    action="will_auto_generate_from_gis_data"
                )
            else:
                # Build map from provided env_df
                cell_env_map = {}
                for _, row in env_df.iterrows():
                    cell_name = str(row['cell_name'])
                    environment = str(row['environment']).upper()
                    if environment in ('URBAN', 'SUBURBAN', 'RURAL'):
                        cell_env_map[cell_name] = environment
                    else:
                        logger.warning(
                            "invalid_environment_value",
                            cell=cell_name,
                            value=row['environment'],
                            defaulting_to="SUBURBAN"
                        )
                        cell_env_map[cell_name] = 'SUBURBAN'

                logger.info(
                    "environment_map_from_env_df",
                    total_cells=len(cell_env_map),
                    urban=sum(1 for v in cell_env_map.values() if v == 'URBAN'),
                    suburban=sum(1 for v in cell_env_map.values() if v == 'SUBURBAN'),
                    rural=sum(1 for v in cell_env_map.values() if v == 'RURAL'),
                )
                return cell_env_map

        # Case 2: Auto-generate from gis_data using EnvironmentClassifier
        if gis_data is not None and len(gis_data) > 0:
            logger.info("auto_generating_environment_classification", cells=len(gis_data))

            try:
                classifier = EnvironmentClassifier(gis_data)
                classified_df = classifier.classify()

                # Build map from classification
                cell_env_map = {}
                for _, row in classified_df.iterrows():
                    cell_name = str(row['cell_name'])
                    environment = row['environment'].upper()
                    cell_env_map[cell_name] = environment

                logger.info(
                    "environment_map_auto_generated",
                    total_cells=len(cell_env_map),
                    urban=sum(1 for v in cell_env_map.values() if v == 'URBAN'),
                    suburban=sum(1 for v in cell_env_map.values() if v == 'SUBURBAN'),
                    rural=sum(1 for v in cell_env_map.values() if v == 'RURAL'),
                )
                return cell_env_map

            except Exception as e:
                logger.warning(
                    "environment_classification_failed",
                    error=str(e),
                    action="using_default_suburban_threshold"
                )
                return {}

        # Case 3: No env_df and no gis_data - return empty map
        logger.info(
            "no_environment_data_available",
            action="using_default_suburban_threshold_for_all_cells"
        )
        return {}

    def detect(
        self,
        hulls: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        gis_data: Optional[pd.DataFrame] = None,
        bands: Optional[List[str]] = None,
        env_df: Optional[pd.DataFrame] = None,
        config_path: Optional[Path] = None,
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Detect low coverage clusters per band with environment-aware thresholds.

        Environment classification determines RSRP thresholds:
        - URBAN: <= -110 dBm (stricter, denser network expected)
        - SUBURBAN: <= -115 dBm (default)
        - RURAL: <= -120 dBm (more lenient, sparser network)

        Args:
            hulls: Cell convex hulls (requires 'band' column or join with gis_data)
            grid_data: Grid measurements with RSRP per band
            gis_data: Optional GIS data to join for band info (also used to auto-generate env_df)
            bands: Optional list of specific bands to process (None = all bands)
            env_df: Optional environment classification per cell. If not provided and gis_data
                    is available, will be auto-generated using intersite distance classification.
                    Expected schema:
                    - cell_name: Cell identifier (string)
                    - environment: 'URBAN', 'SUBURBAN', or 'RURAL' (case-insensitive)
            config_path: Optional path to config file for loading environment-specific params

        Returns:
            Dict mapping band names to GeoDataFrames of low coverage clusters
            e.g., {'L800': gdf, 'L1800': gdf}

        Example:
            >>> # With auto-generated environment classification
            >>> low_cov = detector.detect(hulls_gdf, grid_data, gis_data)
            >>>
            >>> # With pre-computed environment classification
            >>> from ran_optimizer.core.environment_classifier import classify_cell_environments
            >>> env_df = classify_cell_environments(gis_df)
            >>> low_cov = detector.detect(hulls_gdf, grid_data, gis_data, env_df=env_df)
        """
        logger.info("starting_low_coverage_detection", cell_hulls=len(hulls))

        # Build cell environment map for environment-aware thresholds
        cell_env_map = self._build_environment_map(env_df, gis_data)

        # Load environment-specific parameters
        env_params = self._load_environment_params(config_path)

        # Step 1: Ensure hulls have band information
        hulls_with_band = self._add_band_info(hulls, gis_data)

        # Step 2: Normalise grid_data band column
        grid_data = self._normalise_grid_data(grid_data)

        # Step 3: Process each band separately
        available_bands = [b for b in hulls_with_band['band'].unique() if pd.notna(b)]
        if bands is not None:
            # Convert requested bands to normalised format for comparison
            bands_normalised = [self._normalise_band(b) for b in bands]
            available_bands = [b for b in available_bands if b in bands_normalised]

        logger.info("detecting_low_coverage", bands=available_bands)

        # Step 2.5: Apply data quality filtering
        grid_data = self._apply_data_quality_filters(grid_data)

        band_results = {}
        for band in available_bands:
            logger.info("processing_band", band=band)
            band_clusters = self._detect_band_low_coverage(
                hulls_with_band[hulls_with_band['band'] == band],
                grid_data,
                band,
                gis_data=gis_data,
                cell_env_map=cell_env_map,
                env_params=env_params,
            )

            if len(band_clusters) > 0:
                band_results[band] = band_clusters
                logger.info("band_low_coverage_detected", band=band, clusters=len(band_clusters))

        logger.info("low_coverage_detection_complete", bands_with_low_coverage=len(band_results))
        return band_results

    def _load_environment_params(self, config_path: Optional[Path]) -> Dict[str, LowCoverageParams]:
        """
        Load environment-specific LowCoverageParams.

        Returns:
            Dict mapping environment name to LowCoverageParams
        """
        env_params = {}
        for env in ['urban', 'suburban', 'rural']:
            env_params[env] = LowCoverageParams.from_config(config_path, environment=env)
            logger.info(
                f"{env.upper()}_low_coverage_params",
                rsrp_threshold=env_params[env].rsrp_threshold_dbm,
                k_ring_steps=env_params[env].k_ring_steps,
            )
        return env_params

    def _apply_data_quality_filters(self, grid_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters to grid data.

        Filters:
        - RSRP values outside valid range (rsrp_min_dbm to rsrp_max_dbm)
        - Geohashes with fewer than min_sample_count measurements
        """
        original_rows = len(grid_data)
        grid_data = grid_data.copy()

        # Filter invalid RSRP values
        if 'avg_rsrp' in grid_data.columns:
            valid_rsrp = (
                (grid_data['avg_rsrp'] >= self.params.rsrp_min_dbm) &
                (grid_data['avg_rsrp'] <= self.params.rsrp_max_dbm)
            )
            grid_data = grid_data[valid_rsrp]
            filtered_rsrp = original_rows - len(grid_data)
            if filtered_rsrp > 0:
                logger.info(
                    "data_quality_rsrp_filter",
                    removed=filtered_rsrp,
                    remaining=len(grid_data),
                    rsrp_range=f"{self.params.rsrp_min_dbm} to {self.params.rsrp_max_dbm} dBm"
                )

        # Filter geohashes with too few samples
        if 'grid' in grid_data.columns and 'event_count' in grid_data.columns:
            valid_samples = grid_data['event_count'] >= self.params.min_sample_count
            before_filter = len(grid_data)
            grid_data = grid_data[valid_samples]
            filtered_samples = before_filter - len(grid_data)
            if filtered_samples > 0:
                logger.info(
                    "data_quality_sample_count_filter",
                    removed=filtered_samples,
                    remaining=len(grid_data),
                    min_samples=self.params.min_sample_count
                )

        return grid_data

    def _add_band_info(
        self,
        hulls: gpd.GeoDataFrame,
        gis_data: Optional[pd.DataFrame]
    ) -> gpd.GeoDataFrame:
        """Add band information to hulls if not already present."""
        if 'band' in hulls.columns:
            return hulls.copy()

        if gis_data is None:
            raise ValueError("hulls missing 'band' column and no gis_data provided")

        # Validate required columns (canonical schema)
        if 'band' not in gis_data.columns:
            raise ValueError(f"gis_data missing 'band' column. Available columns: {list(gis_data.columns)}")

        if 'cell_name' not in hulls.columns:
            raise ValueError(f"hulls missing 'cell_name' column. Available columns: {list(hulls.columns)}")

        if 'cell_name' not in gis_data.columns:
            raise ValueError(f"gis_data missing 'cell_name' column. Available columns: {list(gis_data.columns)}")

        # Create band mapping
        band_mapping = gis_data[['cell_name', 'band']].drop_duplicates('cell_name').copy()
        band_mapping['cell_name'] = band_mapping['cell_name'].astype(str)

        hulls_copy = hulls.copy()
        hulls_copy['cell_name'] = hulls_copy['cell_name'].astype(str)

        # Join with GIS data to get band
        hulls_with_band = hulls_copy.merge(band_mapping, on='cell_name', how='left')

        # Normalise band values (e.g., 'L800' -> 800, 'L1800' -> 1800)
        hulls_with_band['band'] = hulls_with_band['band'].apply(self._normalise_band)

        missing_band = hulls_with_band['band'].isna().sum()
        if missing_band > 0:
            logger.warning("cells_missing_band_info", count=missing_band, total=len(hulls_with_band))

        return hulls_with_band

    @staticmethod
    def _normalise_band(band_value) -> str:
        """
        Normalise band value to 'L800' string format.

        Handles various formats:
        - 'L800', 'L1800' -> 'L800', 'L1800' (already correct)
        - '800', '1800' -> 'L800', 'L1800' (add 'L' prefix)
        - 800, 1800 -> 'L800', 'L1800' (convert to string with 'L' prefix)
        - 800.0, 1800.0 -> 'L800', 'L1800' (handle float)
        """
        if pd.isna(band_value):
            return None

        # Convert to string and clean up
        band_str = str(band_value).upper().strip()

        # Remove .0 suffix if present (e.g., '800.0' -> '800')
        if band_str.endswith('.0'):
            band_str = band_str[:-2]

        # Add 'L' prefix if not present
        if not band_str.startswith('L'):
            band_str = 'L' + band_str

        return band_str

    def _normalise_grid_data(self, grid_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise grid data band column for consistency.

        - Normalises band column to 'L800' string format
        - Does NOT rename columns - uses canonical names: grid, avg_rsrp, cell_name
        """
        grid_data = grid_data.copy()

        # Find and normalise band column
        band_col = None
        for col in ['band', 'Band', 'frequency_band']:
            if col in grid_data.columns:
                band_col = col
                break

        if band_col:
            # Rename to lowercase 'band' if needed
            if band_col != 'band':
                grid_data = grid_data.rename(columns={band_col: 'band'})

            # Normalise band values to integers
            grid_data['band'] = grid_data['band'].apply(self._normalise_band)

            logger.info(
                "grid_data_normalised",
                bands=grid_data['band'].unique().tolist()[:10],
                rows=len(grid_data)
            )

        return grid_data

    def _detect_band_low_coverage(
        self,
        band_hulls: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        band: str,
        gis_data: Optional[pd.DataFrame] = None,
        cell_env_map: Optional[Dict[str, str]] = None,
        env_params: Optional[Dict[str, LowCoverageParams]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Detect low coverage for a specific band with environment-aware thresholds.

        Low coverage detection is DATA-DRIVEN:
        1. Find ALL geohashes where BEST RSRP (across all serving cells) is below threshold
        2. Apply k-ring density filtering and clustering
        3. Includes both single-server AND multi-server areas with poor coverage

        Args:
            band_hulls: Cell hulls for this band (used for cell info only)
            grid_data: Grid measurements
            band: Band being processed
            gis_data: GIS data for cell info
            cell_env_map: Mapping of cell_name to environment (urban/suburban/rural)
            env_params: Environment-specific LowCoverageParams
        """
        cell_env_map = cell_env_map or {}
        env_params = env_params or {}

        # Filter grid data for this band
        band_grid = grid_data[grid_data['band'] == band] if 'band' in grid_data.columns else grid_data

        if len(band_grid) == 0:
            logger.info("no_grid_data_for_band", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        geohash_col = 'grid' if 'grid' in band_grid.columns else None
        cell_col = 'cell_name' if 'cell_name' in band_grid.columns else None

        if not geohash_col or not cell_col:
            logger.error("missing_required_columns", band=band, has_grid=bool(geohash_col), has_cell=bool(cell_col))
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 1: Get all geohashes and their server counts for logging
        cells_per_geohash = band_grid.groupby(geohash_col)[cell_col].nunique()
        single_server_count = (cells_per_geohash == 1).sum()
        multi_server_count = (cells_per_geohash > 1).sum()

        logger.info(
            "geohash_server_distribution",
            band=band,
            total_geohashes=len(cells_per_geohash),
            single_server=single_server_count,
            multi_server=multi_server_count
        )

        # Step 2: Get ALL geohashes as candidates (not just single-server)
        all_geohashes = set(cells_per_geohash.index)

        # Step 3: Filter by RSRP threshold (environment-aware)
        # For multi-server geohashes, uses BEST (max) RSRP across all serving cells
        low_rsrp_geohashes = self._filter_by_rsrp_environment_aware(
            all_geohashes,
            grid_data,
            band,
            cell_env_map,
            env_params,
        )

        logger.info(
            "low_rsrp_geohashes_found",
            band=band,
            candidates=len(all_geohashes),
            low_rsrp=len(low_rsrp_geohashes)
        )

        if len(low_rsrp_geohashes) == 0:
            logger.info("no_low_rsrp_geohashes", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 4: Apply k-ring density filtering for low coverage
        # For low coverage, we count neighbors with GOOD coverage and invert
        dense_gaps = self._compute_kring_density_low_coverage(
            low_rsrp_geohashes,
            grid_data,
            band,
            self.params.k_ring_steps,
            self.params.min_missing_neighbors
        )

        if len(dense_gaps) == 0:
            logger.info("no_dense_low_coverage", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 5: Cluster with HDBSCAN
        clustered_gaps = self._cluster_hdbscan(dense_gaps, self.params.hdbscan_min_cluster_size)

        if len(clustered_gaps) == 0:
            logger.info("no_low_coverage_clusters", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 6: Create polygons with serving cell info
        cluster_polygons = self._create_cluster_polygons_with_cells(
            clustered_gaps,
            grid_data,
            band,
            self.params.alpha_shape_alpha,
            self.params.max_alphashape_points,
            gis_data=gis_data
        )

        # Step 7: Filter by boundary if provided (remove offshore gaps)
        if self.boundary_gdf is not None and len(cluster_polygons) > 0:
            cluster_polygons = self._filter_by_boundary(cluster_polygons)
            if len(cluster_polygons) == 0:
                logger.info("all_clusters_filtered_by_boundary", band=band)
                return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 8: Add band label
        cluster_polygons['band'] = band

        # Step 9: Calculate severity scores
        cluster_polygons = self._calculate_severity_scores(cluster_polygons, grid_data, band)

        return cluster_polygons

    def _create_cluster_polygons_with_cells(
        self,
        clustered_gaps: pd.DataFrame,
        grid_data: pd.DataFrame,
        band: str,
        alpha_shape_alpha: Optional[float],
        max_alphashape_points: int,
        gis_data: Optional[pd.DataFrame] = None
    ) -> gpd.GeoDataFrame:
        """
        Create alpha shape polygons for each gap cluster with serving cell info.

        Args:
            clustered_gaps: DataFrame with clustered gap geohashes
            grid_data: Grid data to look up serving cells
            band: Band being processed
            alpha_shape_alpha: Alpha parameter for concave hull (None = auto)
            max_alphashape_points: Maximum points for alphashape (subsample if exceeded)
            gis_data: Optional GIS data to map cell IDs to cell names

        Returns:
            GeoDataFrame with cluster polygons, metadata, and serving cell info
        """
        cluster_records = []

        # Use canonical column names
        geohash_col = 'grid' if 'grid' in grid_data.columns else None
        cell_col = 'cell_name' if 'cell_name' in grid_data.columns else None

        # Filter grid data for this band
        band_grid = grid_data[grid_data['band'] == band] if 'band' in grid_data.columns else grid_data

        # Create cell_name mapping from GIS data (cell_name is the canonical identifier)
        cell_id_to_name = {}
        if gis_data is not None and 'cell_name' in gis_data.columns:
            # cell_name is both the ID and the name in canonical schema
            cell_id_to_name = {str(cn): str(cn) for cn in gis_data['cell_name'].dropna().unique()}

        # Create geohash -> cell mapping using vectorized groupby
        geohash_cell_map = {}
        if geohash_col and cell_col:
            # Group by geohash and collect unique cells
            grouped = band_grid.groupby(geohash_col)[cell_col].apply(
                lambda x: set(x.astype(str).unique())
            )
            geohash_cell_map = grouped.to_dict()

        # Create geohash -> event_count mapping for measurement density
        geohash_event_count = {}
        if geohash_col and 'event_count' in band_grid.columns:
            # Sum event counts per geohash (across all cells)
            geohash_event_count = band_grid.groupby(geohash_col)['event_count'].sum().to_dict()

        for cluster_id in clustered_gaps['cluster_id'].unique():
            cluster_points = clustered_gaps[clustered_gaps['cluster_id'] == cluster_id]
            coords = cluster_points[['longitude', 'latitude']].values

            # Find serving cells for this cluster
            serving_cells = set()
            if 'grid' in cluster_points.columns:
                for gh in cluster_points['grid']:
                    if gh in geohash_cell_map:
                        serving_cells.update(geohash_cell_map[gh])

            # Subsample if too many points for alphashape
            if len(coords) > max_alphashape_points:
                logger.info(
                    "subsampling_for_alphashape",
                    cluster_id=cluster_id,
                    original=len(coords),
                    subsample=max_alphashape_points
                )
                indices = np.random.choice(len(coords), max_alphashape_points, replace=False)
                coords = coords[indices]

            try:
                # Try alpha shape
                if alpha_shape_alpha is None:
                    alpha_shape = alphashape.alphashape(coords.tolist())
                else:
                    alpha_shape = alphashape.alphashape(coords.tolist(), float(alpha_shape_alpha))

                if alpha_shape.is_empty or not alpha_shape.is_valid:
                    # Fallback to convex hull
                    points = [Point(lon, lat) for lon, lat in coords]
                    alpha_shape = gpd.GeoSeries(points).unary_union.convex_hull
                    logger.info("using_convex_hull_fallback", cluster_id=cluster_id)

                alpha_shape = unary_union([alpha_shape]).buffer(0)

            except Exception as e:
                logger.warning("alphashape_failed", cluster_id=cluster_id, error=str(e))
                # Fallback to convex hull
                points = [Point(lon, lat) for lon, lat in coords]
                alpha_shape = gpd.GeoSeries(points).unary_union.convex_hull

            # Calculate area in km² using projected CRS
            area_km2 = self._calculate_area_km2(alpha_shape)

            # Get cell names for serving cells
            serving_cell_names = []
            for cell_id in sorted(serving_cells)[:10]:
                cell_name = cell_id_to_name.get(cell_id, '')
                serving_cell_names.append(cell_name)

            # Calculate measurement density for this cluster
            cluster_geohashes = cluster_points['grid'].tolist() if 'grid' in cluster_points.columns else []
            event_counts = [geohash_event_count.get(gh, 0) for gh in cluster_geohashes]
            total_samples = sum(event_counts)
            avg_samples_per_geohash = total_samples / len(cluster_geohashes) if cluster_geohashes else 0

            # Calculate confidence score based on measurement density
            # Higher sample count = higher confidence
            # Thresholds: <5 avg samples = LOW, 5-20 = MEDIUM, >20 = HIGH
            if avg_samples_per_geohash >= 20:
                confidence_score = min(1.0, 0.7 + (avg_samples_per_geohash - 20) / 100)
                confidence_category = 'HIGH'
            elif avg_samples_per_geohash >= 5:
                confidence_score = 0.4 + (avg_samples_per_geohash - 5) / 50
                confidence_category = 'MEDIUM'
            else:
                confidence_score = max(0.1, avg_samples_per_geohash / 12.5)
                confidence_category = 'LOW'

            cluster_records.append({
                'cluster_id': int(cluster_id),
                'n_points': len(cluster_points),
                'centroid_lat': cluster_points['latitude'].mean(),
                'centroid_lon': cluster_points['longitude'].mean(),
                'area_km2': area_km2,
                'serving_cells': ','.join(sorted(serving_cells)[:10]) if serving_cells else '',  # Limit to 10 cells
                'serving_cell_names': ','.join(serving_cell_names) if serving_cell_names else '',
                'n_serving_cells': len(serving_cells),
                'total_samples': int(total_samples),
                'avg_samples_per_geohash': round(avg_samples_per_geohash, 1),
                'confidence_score': round(confidence_score, 3),
                'confidence_category': confidence_category,
                'geometry': alpha_shape
            })

        gdf = gpd.GeoDataFrame(cluster_records, crs="EPSG:4326")
        # Log confidence distribution
        if len(gdf) > 0 and 'confidence_category' in gdf.columns:
            confidence_counts = gdf['confidence_category'].value_counts().to_dict()
            avg_confidence = gdf['confidence_score'].mean()
            logger.info(
                "cluster_polygons_created_with_cells",
                clusters=len(gdf),
                confidence_distribution=confidence_counts,
                avg_confidence=round(avg_confidence, 3)
            )
        else:
            logger.info("cluster_polygons_created_with_cells", clusters=len(gdf))

        return gdf

    def _calculate_severity_scores(
        self,
        cluster_gdf: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        band: str
    ) -> gpd.GeoDataFrame:
        """
        Calculate severity scores for low coverage clusters.

        Severity is based on:
        - Area (larger clusters = more severe)
        - Number of low-coverage points (more points = more severe)
        - RSRP severity (how far below threshold = more severe)
        - Serving cell count (fewer cells = harder to fix = more severe)

        Args:
            cluster_gdf: GeoDataFrame with cluster polygons
            grid_data: Grid measurements for RSRP lookup
            band: Band being analyzed

        Returns:
            GeoDataFrame with severity_score and severity_category columns added
        """
        if len(cluster_gdf) == 0:
            cluster_gdf['severity_score'] = []
            cluster_gdf['severity_category'] = []
            cluster_gdf['avg_rsrp_dbm'] = []
            cluster_gdf['rsrp_deficit_db'] = []
            return cluster_gdf

        cluster_gdf = cluster_gdf.copy()

        # Filter grid data for this band
        band_grid = grid_data[grid_data['band'] == band] if 'band' in grid_data.columns else grid_data

        # Calculate average RSRP per cluster (from geohashes in serving_cells)
        avg_rsrp_per_cluster = []
        for _, row in cluster_gdf.iterrows():
            serving_cells = row.get('serving_cells', '')
            if serving_cells:
                cell_list = serving_cells.split(',')
                # Get RSRP for grids served by these cells with low RSRP
                cell_grids = band_grid[
                    (band_grid['cell_name'].isin(cell_list)) &
                    (band_grid['avg_rsrp'] <= self.params.rsrp_threshold_dbm)
                ]
                if len(cell_grids) > 0:
                    avg_rsrp_per_cluster.append(cell_grids['avg_rsrp'].mean())
                else:
                    avg_rsrp_per_cluster.append(self.params.rsrp_threshold_dbm)
            else:
                avg_rsrp_per_cluster.append(self.params.rsrp_threshold_dbm)

        cluster_gdf['avg_rsrp_dbm'] = avg_rsrp_per_cluster

        # Calculate RSRP deficit (how far below threshold)
        cluster_gdf['rsrp_deficit_db'] = self.params.rsrp_threshold_dbm - cluster_gdf['avg_rsrp_dbm']
        cluster_gdf['rsrp_deficit_db'] = cluster_gdf['rsrp_deficit_db'].clip(lower=0)

        # Normalize components to 0-1 scale
        # 1. Area score (larger = more severe)
        area_score = (cluster_gdf['area_km2'] / self.params.severity_area_max_km2).clip(0, 1)

        # 2. N_points score (use percentile normalization)
        n_points_p95 = cluster_gdf['n_points'].quantile(0.95) if len(cluster_gdf) > 1 else cluster_gdf['n_points'].max()
        n_points_p5 = cluster_gdf['n_points'].quantile(0.05) if len(cluster_gdf) > 1 else 0
        if n_points_p95 > n_points_p5:
            n_points_score = ((cluster_gdf['n_points'] - n_points_p5) / (n_points_p95 - n_points_p5)).clip(0, 1)
        else:
            n_points_score = pd.Series(0.5, index=cluster_gdf.index)

        # 3. RSRP severity score (larger deficit = more severe)
        rsrp_score = (cluster_gdf['rsrp_deficit_db'] / self.params.severity_rsrp_max_db).clip(0, 1)

        # 4. Serving cell score (fewer cells = harder to fix = more severe)
        # Invert: 1 cell = score 1.0, many cells = lower score
        max_serving = cluster_gdf['n_serving_cells'].max() if cluster_gdf['n_serving_cells'].max() > 0 else 1
        serving_score = 1.0 - (cluster_gdf['n_serving_cells'] / max_serving).clip(0, 1)

        # Weighted combination
        severity_score = (
            self.params.severity_weight_area * area_score +
            self.params.severity_weight_n_points * n_points_score +
            self.params.severity_weight_rsrp * rsrp_score +
            self.params.severity_weight_serving_cells * serving_score
        ).clip(0, 1)

        cluster_gdf['severity_score'] = severity_score

        # Categorize severity
        conditions = [
            cluster_gdf['severity_score'] >= self.params.severity_threshold_critical,
            cluster_gdf['severity_score'] >= self.params.severity_threshold_high,
            cluster_gdf['severity_score'] >= self.params.severity_threshold_medium,
            cluster_gdf['severity_score'] >= self.params.severity_threshold_low,
        ]
        choices = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        cluster_gdf['severity_category'] = np.select(conditions, choices, default='MINIMAL')

        logger.info(
            "severity_scores_calculated",
            band=band,
            clusters=len(cluster_gdf),
            avg_severity=cluster_gdf['severity_score'].mean(),
            critical_count=len(cluster_gdf[cluster_gdf['severity_category'] == 'CRITICAL']),
            high_count=len(cluster_gdf[cluster_gdf['severity_category'] == 'HIGH']),
        )

        return cluster_gdf

    def _find_single_server_regions(
        self,
        band_hulls: gpd.GeoDataFrame
    ) -> List[Polygon]:
        """
        Find regions where only one cell provides coverage (no overlap).

        For each cell hull, subtract all overlapping areas from other cells.
        """
        single_server_regions = []

        for idx, hull_row in band_hulls.iterrows():
            cell_name = hull_row.get('cell_name', f'cell_{idx}')
            hull_geom = hull_row['geometry']

            # Find all other hulls that overlap
            other_hulls = band_hulls[band_hulls.index != idx]
            overlapping = other_hulls[other_hulls.intersects(hull_geom)]

            if len(overlapping) == 0:
                # Entire hull is single-server
                single_server_regions.append(hull_geom)
                logger.debug("entire_hull_single_server", cell_name=cell_name)
            else:
                # Subtract overlapping areas
                overlap_union = unary_union(overlapping.geometry)
                single_server = hull_geom.difference(overlap_union)

                # Handle MultiPolygon results
                if isinstance(single_server, Polygon) and single_server.area > 0:
                    single_server_regions.append(single_server)
                elif isinstance(single_server, MultiPolygon):
                    single_server_regions.extend([p for p in single_server.geoms if p.area > 0])
                elif single_server.is_empty:
                    logger.debug("cell_completely_overlapped", cell_name=cell_name)

        logger.info("single_server_regions_found", count=len(single_server_regions))
        return single_server_regions

    def _get_grid_in_single_server_regions(
        self,
        single_server_polygons: List[Polygon],
        grid_data: pd.DataFrame,
        band: str
    ) -> List[str]:
        """
        Get geohashes from grid data that fall within single-server regions.

        Uses spatial join between grid point coordinates and single-server polygons.

        Args:
            single_server_polygons: List of single-server region polygons
            grid_data: Grid measurements with coordinates
            band: Band to filter

        Returns:
            List of geohash strings within single-server regions
        """
        # Filter grid data for this band
        band_grid = grid_data[grid_data['band'] == band] if 'band' in grid_data.columns else grid_data

        if len(band_grid) == 0:
            logger.warning("no_grid_data_for_band", band=band)
            return []

        # Get unique geohashes with their coordinates
        geohash_col = 'grid' if 'grid' in band_grid.columns else None
        if not geohash_col:
            logger.error("missing_grid_column", columns=list(band_grid.columns[:10]))
            return []

        # Get lat/lon columns
        lat_col = 'Latitude' if 'Latitude' in band_grid.columns else 'latitude'
        lon_col = 'Longitude' if 'Longitude' in band_grid.columns else 'longitude'

        if lat_col not in band_grid.columns or lon_col not in band_grid.columns:
            logger.error("missing_coordinate_columns", columns=list(band_grid.columns[:15]))
            return []

        # Get unique geohash -> coordinate mapping (use first occurrence)
        unique_geohashes = band_grid.drop_duplicates(subset=[geohash_col])[[geohash_col, lat_col, lon_col]]

        logger.info(
            "spatial_join_starting",
            band=band,
            unique_geohashes=len(unique_geohashes),
            single_server_polygons=len(single_server_polygons)
        )

        # Create union of all single-server polygons for efficient containment check
        ss_union = unary_union(single_server_polygons)
        prep_ss = prep(ss_union)

        # Check which geohashes fall within single-server regions
        matching_geohashes = []
        for _, row in unique_geohashes.iterrows():
            point = Point(row[lon_col], row[lat_col])
            if prep_ss.contains(point):
                matching_geohashes.append(row[geohash_col])

        logger.info(
            "spatial_join_complete",
            band=band,
            input_geohashes=len(unique_geohashes),
            matching_geohashes=len(matching_geohashes),
            pct_match=f"{100*len(matching_geohashes)/len(unique_geohashes):.1f}%" if len(unique_geohashes) > 0 else "0%"
        )

        return matching_geohashes

    def _compute_kring_density_low_coverage(
        self,
        geohashes: Set[str],
        grid_data: pd.DataFrame,
        band: str,
        k: int,
        min_missing_neighbors: int
    ) -> pd.DataFrame:
        """
        Compute k-ring density for low coverage by counting neighbors with GOOD coverage.

        For low coverage, we want dense clusters of poor coverage. We achieve this by:
        1. Counting neighbors with GOOD coverage (RSRP > threshold) on the same band
        2. Subtracting from total neighbors to get "problematic" neighbors (low or no coverage)
        3. Keeping points where problematic neighbors >= threshold (e.g., 40 out of 49)

        This correctly accounts for both low coverage AND no coverage neighbors.

        Args:
            geohashes: Set of candidate low coverage geohashes
            grid_data: Grid data with RSRP measurements
            band: Band to analyze
            k: K-ring steps (e.g., 3 for 7x7 grid = 49 neighbors)
            min_missing_neighbors: Minimum problematic neighbors required (e.g., 40)

        Returns:
            DataFrame with dense gap geohashes and their coordinates
        """
        # Filter grid data for this band
        band_grid = grid_data[grid_data['band'] == band] if 'band' in grid_data.columns else grid_data

        # Use canonical column names
        if 'avg_rsrp' not in band_grid.columns:
            logger.error("missing_avg_rsrp_column", band=band, columns=list(band_grid.columns[:10]))
            return pd.DataFrame()

        if 'grid' not in band_grid.columns:
            logger.error("missing_grid_column", band=band, columns=list(band_grid.columns[:10]))
            return pd.DataFrame()

        rsrp_col = 'avg_rsrp'
        geohash_col = 'grid'

        # Create set of geohashes with GOOD coverage (RSRP > threshold)
        good_coverage = band_grid[band_grid[rsrp_col] > self.params.rsrp_threshold_dbm]
        good_coverage_geohashes = set(good_coverage[geohash_col].unique())

        logger.info(
            "computing_kring_density_low_coverage",
            k=k,
            total_geohashes=len(geohashes),
            good_coverage_geohashes=len(good_coverage_geohashes),
            band=band
        )

        records = []
        total_neighbors = (2 * k + 1) ** 2  # e.g., 49 for k=3

        for gh in geohashes:
            # Get all k-ring neighbors
            kring_set = geohash_utils.kring(gh, k)

            # Count neighbors with GOOD coverage
            good_neighbors = sum(1 for n in kring_set if n in good_coverage_geohashes)

            # Problematic neighbors = total - good (includes both low coverage AND no coverage)
            problematic_neighbors = total_neighbors - good_neighbors

            # Keep if enough problematic neighbors (dense low/no coverage area)
            if problematic_neighbors >= min_missing_neighbors:
                lat, lon = geohash_utils.decode(gh)
                records.append({
                    'grid': gh,
                    'latitude': lat,
                    'longitude': lon,
                    f'problematic_within_{k}_steps': problematic_neighbors
                })

        df = pd.DataFrame(records)

        if len(df) > 0:
            logger.info(
                "kring_density_low_coverage_complete",
                band=band,
                input_geohashes=len(geohashes),
                dense_gaps=len(df),
                mean_problematic=df[f'problematic_within_{k}_steps'].mean()
            )
        else:
            logger.info("no_dense_gaps_after_kring_filter_low_coverage", band=band)

        return df

    def _filter_by_rsrp(
        self,
        geohashes: Set[str],
        grid_data: pd.DataFrame,
        band: str
    ) -> Set[str]:
        """Filter geohashes by RSRP threshold for specific band."""
        # Validate required columns
        if 'avg_rsrp' not in grid_data.columns:
            logger.error("missing_avg_rsrp_column", columns=list(grid_data.columns[:10]))
            return set()

        if 'grid' not in grid_data.columns:
            logger.error("missing_grid_column", columns=list(grid_data.columns[:10]))
            return set()

        # Filter by band if column exists
        if 'band' in grid_data.columns:
            band_grid = grid_data[grid_data['band'] == band]
        else:
            band_grid = grid_data

        # Filter by RSRP threshold
        low_rsrp = band_grid[band_grid['avg_rsrp'] <= self.params.rsrp_threshold_dbm]
        low_rsrp_geohashes = set(low_rsrp['grid'].unique())

        # Intersect with candidate geohashes
        filtered = geohashes.intersection(low_rsrp_geohashes)
        logger.info("rsrp_filtering_complete", band=band, candidates=len(geohashes), low_rsrp=len(filtered))

        return filtered

    def _filter_by_rsrp_environment_aware(
        self,
        geohashes: Set[str],
        grid_data: pd.DataFrame,
        band: str,
        cell_env_map: Dict[str, str],
        env_params: Dict[str, 'LowCoverageParams'],
    ) -> Set[str]:
        """
        Filter geohashes by environment-specific RSRP thresholds (vectorized).

        For each geohash, determines the serving cell's environment and applies
        the appropriate RSRP threshold:
        - Urban: -110 dBm (stricter)
        - Suburban: -115 dBm (default)
        - Rural: -120 dBm (more lenient)

        Args:
            geohashes: Set of candidate geohashes
            grid_data: Grid measurements with cell_name and avg_rsrp
            band: Band being processed
            cell_env_map: Mapping of cell_name to environment
            env_params: Environment-specific LowCoverageParams

        Returns:
            Set of geohashes that have low RSRP based on their serving cell's environment
        """
        # If no environment info, fall back to default threshold
        if not cell_env_map or not env_params:
            return self._filter_by_rsrp(geohashes, grid_data, band)

        # Validate required columns
        if 'avg_rsrp' not in grid_data.columns or 'grid' not in grid_data.columns:
            logger.error("missing_required_columns", columns=list(grid_data.columns[:10]))
            return set()

        # Filter by band
        if 'band' in grid_data.columns:
            band_grid = grid_data[grid_data['band'] == band].copy()
        else:
            band_grid = grid_data.copy()

        # Filter to candidate geohashes only (early filter for performance)
        band_grid = band_grid[band_grid['grid'].isin(geohashes)]

        if len(band_grid) == 0:
            logger.info("no_candidate_geohashes_in_grid_data", band=band)
            return set()

        # Get RSRP thresholds by environment
        env_thresholds = {
            'urban': env_params.get('urban', self.params).rsrp_threshold_dbm,
            'suburban': env_params.get('suburban', self.params).rsrp_threshold_dbm,
            'rural': env_params.get('rural', self.params).rsrp_threshold_dbm,
        }
        default_threshold = self.params.rsrp_threshold_dbm

        logger.info(
            "environment_aware_rsrp_thresholds",
            urban=env_thresholds['urban'],
            suburban=env_thresholds['suburban'],
            rural=env_thresholds['rural'],
            band=band
        )

        # Vectorized: Map cell_name -> environment -> threshold
        if 'cell_name' in band_grid.columns:
            band_grid['cell_name_str'] = band_grid['cell_name'].astype(str)
            band_grid['environment'] = band_grid['cell_name_str'].map(cell_env_map).fillna('suburban').str.lower()
            band_grid['rsrp_threshold'] = band_grid['environment'].map(env_thresholds).fillna(default_threshold)
        else:
            band_grid['environment'] = 'suburban'
            band_grid['rsrp_threshold'] = default_threshold

        # For multi-server geohashes: use BEST (max) RSRP across all serving cells
        # A geohash has low coverage only if even the best server is below threshold
        # Group by geohash and get max RSRP and corresponding threshold
        geohash_stats = band_grid.groupby('grid').agg({
            'avg_rsrp': 'max',  # Best RSRP from any cell
            'rsrp_threshold': 'min',  # Most lenient threshold (for consistent comparison)
            'environment': 'first'  # Keep one environment for logging
        }).reset_index()

        # Check if BEST RSRP is still below threshold
        geohash_stats['is_low_coverage'] = geohash_stats['avg_rsrp'] <= geohash_stats['rsrp_threshold']

        # Get low-coverage geohashes
        low_cov_df = geohash_stats[geohash_stats['is_low_coverage']]
        filtered = set(low_cov_df['grid'].unique())

        # Count by environment
        env_counts = low_cov_df.groupby('environment')['grid'].nunique().to_dict()

        logger.info(
            "rsrp_filtering_complete",
            band=band,
            candidates=len(geohashes),
            low_rsrp=len(filtered),
            by_environment=env_counts
        )

        return filtered

    def _filter_by_boundary(
        self,
        cluster_polygons: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Clip cluster polygons to the boundary shapefile.

        This clips offshore/out-of-region coverage gaps to keep only the parts within the boundary.

        Args:
            cluster_polygons: GeoDataFrame with cluster polygons

        Returns:
            GeoDataFrame with polygons clipped to boundary
        """
        if self.boundary_gdf is None:
            return cluster_polygons

        # Ensure same CRS
        if cluster_polygons.crs != self.boundary_gdf.crs:
            boundary_aligned = self.boundary_gdf.to_crs(cluster_polygons.crs)
        else:
            boundary_aligned = self.boundary_gdf

        # Create union of all boundary polygons
        boundary_union = boundary_aligned.geometry.unary_union

        # Clip each polygon to the boundary
        clipped_records = []
        for _, row in cluster_polygons.iterrows():
            clipped_geom = row['geometry'].intersection(boundary_union)

            # Skip if clipped polygon is empty
            if clipped_geom.is_empty:
                continue

            # Recalculate area after clipping
            area_km2 = self._calculate_area_km2(clipped_geom)

            # Recalculate centroid
            centroid = clipped_geom.centroid

            clipped_records.append({
                'cluster_id': row['cluster_id'],
                'n_points': row['n_points'],
                'centroid_lat': centroid.y,
                'centroid_lon': centroid.x,
                'area_km2': area_km2,
                'band': row.get('band'),
                'serving_cells': row.get('serving_cells', ''),
                'serving_cell_names': row.get('serving_cell_names', ''),
                'n_serving_cells': row.get('n_serving_cells', 0),
                'geometry': clipped_geom
            })

        if len(clipped_records) == 0:
            logger.info(
                "boundary_clipping_removed_all_low_coverage",
                original_clusters=len(cluster_polygons)
            )
            return gpd.GeoDataFrame(
                columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'area_km2', 'band', 'serving_cells', 'serving_cell_names', 'n_serving_cells', 'geometry'],
                crs=cluster_polygons.crs
            )

        clipped_gdf = gpd.GeoDataFrame(clipped_records, crs=cluster_polygons.crs)

        logger.info(
            "boundary_clipping_complete",
            original_clusters=len(cluster_polygons),
            clipped_clusters=len(clipped_gdf),
            removed_clusters=len(cluster_polygons) - len(clipped_gdf)
        )

        return clipped_gdf


class CoverageGapAnalyzer:
    """
    Analyzes coverage gaps to find nearby serving cells.

    Example:
        >>> analyzer = CoverageGapAnalyzer()
        >>> gap_analysis = analyzer.find_cells_for_gaps(gap_clusters, grid_data)
    """

    def __init__(self, params: Optional[CoverageGapParams] = None):
        """
        Initialize coverage gap analyzer.

        Args:
            params: Configuration parameters (uses defaults if not provided)
        """
        self.params = params or CoverageGapParams()

    def find_cells_for_gaps(
        self,
        gap_clusters: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        gis_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Find nearest serving cells for each gap cluster, filtered by azimuth.

        Only includes cells that are actually pointing toward the gap (within beam coverage).

        Args:
            gap_clusters: GeoDataFrame with gap cluster polygons
            grid_data: DataFrame with grid measurements (must have cell_name, latitude, longitude)
            gis_data: Optional GIS data with cell bearing and beamwidth for azimuth filtering

        Returns:
            DataFrame with gap analysis results including nearby cells

        Example:
            >>> gap_analysis = analyzer.find_cells_for_gaps(gap_clusters, grid_df, gis_df)
            >>> print(gap_analysis[['cluster_id', 'nearby_cell_count', 'cells_pointing_toward_gap']])
        """
        if len(gap_clusters) == 0:
            return pd.DataFrame()

        logger.info("analyzing_gap_clusters", clusters=len(gap_clusters))

        # Build cell info lookup from GIS data for azimuth filtering
        cell_info = {}
        if gis_data is not None:
            required_cols = ['cell_name', 'latitude', 'longitude', 'bearing', 'hbw']
            if all(col in gis_data.columns for col in required_cols):
                for _, row in gis_data.iterrows():
                    cell_info[str(row['cell_name'])] = {
                        'lat': row['latitude'],
                        'lon': row['longitude'],
                        'bearing': row['bearing'],
                        'hbw': row['hbw'] if pd.notna(row['hbw']) else 65  # Default 65° beamwidth
                    }
                logger.info("cell_azimuth_info_loaded", cells=len(cell_info))
            else:
                logger.warning(
                    "gis_data_missing_azimuth_columns",
                    required=required_cols,
                    available=list(gis_data.columns)
                )

        # Build BallTree for grid points
        grid_coords = grid_data[['latitude', 'longitude']].values
        grid_coords_rad = np.radians(grid_coords)
        tree = BallTree(grid_coords_rad, metric='haversine')

        results = []

        for _, cluster in gap_clusters.iterrows():
            cluster_id = cluster['cluster_id']
            centroid_lat = cluster['centroid_lat']
            centroid_lon = cluster['centroid_lon']

            # Query k nearest grid points (get more to filter by azimuth)
            k_query = self.params.k_nearest_cells * 3 if cell_info else self.params.k_nearest_cells
            centroid_rad = np.radians([[centroid_lat, centroid_lon]])
            distances, indices = tree.query(centroid_rad, k=min(k_query, len(grid_data)))

            # Convert distances to meters
            distances_m = distances[0] * EARTH_RADIUS_M

            # Get unique cells using canonical column name
            nearby_rows = grid_data.iloc[indices[0]]
            if 'cell_name' in nearby_rows.columns:
                candidate_cells = nearby_rows['cell_name'].unique().tolist()
            else:
                logger.warning("missing_cell_name_column", columns=list(grid_data.columns[:10]))
                candidate_cells = []

            # Filter by azimuth - only keep cells pointing toward the gap
            cells_pointing_toward = []
            cells_not_pointing = []

            for cell_name in candidate_cells:
                cell_name_str = str(cell_name)
                if cell_name_str in cell_info:
                    info = cell_info[cell_name_str]
                    # Calculate bearing from cell to gap centroid
                    bearing_to_gap = self._calculate_bearing(
                        info['lat'], info['lon'],
                        centroid_lat, centroid_lon
                    )
                    # Check if gap is within cell's beam coverage
                    if self._is_within_beam(info['bearing'], info['hbw'], bearing_to_gap):
                        cells_pointing_toward.append(cell_name_str)
                    else:
                        cells_not_pointing.append(cell_name_str)
                else:
                    # No azimuth info - include by default
                    cells_pointing_toward.append(cell_name_str)

            # Limit to k_nearest_cells
            cells_pointing_toward = cells_pointing_toward[:self.params.k_nearest_cells]

            results.append({
                'cluster_id': cluster_id,
                'nearby_cells': cells_pointing_toward,
                'nearby_cell_count': len(cells_pointing_toward),
                'cells_filtered_by_azimuth': len(cells_not_pointing),
                'avg_distance_to_coverage_m': float(distances_m.mean())
            })

        df = pd.DataFrame(results)

        total_filtered = df['cells_filtered_by_azimuth'].sum() if 'cells_filtered_by_azimuth' in df.columns else 0
        logger.info(
            "gap_analysis_complete",
            analyzed_clusters=len(df),
            cells_filtered_by_azimuth=total_filtered
        )

        return df

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate bearing from point 1 to point 2 in degrees (0-360).

        Args:
            lat1, lon1: Origin point (cell location)
            lat2, lon2: Destination point (gap centroid)

        Returns:
            Bearing in degrees (0 = North, 90 = East, 180 = South, 270 = West)
        """
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lon = np.radians(lon2 - lon1)

        x = np.sin(delta_lon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)

        bearing_rad = np.arctan2(x, y)
        bearing_deg = np.degrees(bearing_rad)

        # Normalize to 0-360
        return (bearing_deg + 360) % 360

    @staticmethod
    def _is_within_beam(cell_bearing: float, beamwidth: float, target_bearing: float) -> bool:
        """
        Check if target bearing is within cell's beam coverage.

        Args:
            cell_bearing: Cell's azimuth in degrees (0-360)
            beamwidth: Horizontal beamwidth in degrees (e.g., 65)
            target_bearing: Bearing from cell to target in degrees

        Returns:
            True if target is within beam coverage (±beamwidth/2 from cell bearing)
        """
        half_beam = beamwidth / 2.0

        # Calculate angular difference (handle wrap-around)
        diff = abs(target_bearing - cell_bearing)
        if diff > 180:
            diff = 360 - diff

        return diff <= half_beam


@dataclass
class LowCoverageRecommendationParams:
    """Parameters for low coverage recommendation generation."""
    # Search radius by environment (km)
    search_radius_urban_km: float = 2.0
    search_radius_suburban_km: float = 5.0
    search_radius_rural_km: float = 10.0

    # Recommendation limits
    max_recommendations_per_cluster: int = 3
    min_expected_rsrp_improvement_db: float = 3.0
    new_site_threshold_db: float = 10.0

    # RF propagation parameters
    hpbw_v_deg: float = 6.5
    sla_v_db: float = 30.0
    path_loss_exponent_urban: float = 4.0
    path_loss_exponent_suburban: float = 3.5
    path_loss_exponent_rural: float = 3.0
    default_antenna_height_m: float = 30.0

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None):
        """Load parameters from config file."""
        if config_path is None:
            config_path = Path("config/coverage_gaps.json")

        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                config = json.load(f)

            rec_config = config.get("recommendation_generation", {})

            # Extract search radius by environment
            search_radius = rec_config.get("candidate_search_radius_km", {})

            # Extract RF propagation params
            rf_config = rec_config.get("rf_propagation", {})
            ple_config = rf_config.get("path_loss_exponent", {})

            return cls(
                search_radius_urban_km=search_radius.get("urban", 2.0),
                search_radius_suburban_km=search_radius.get("suburban", 5.0),
                search_radius_rural_km=search_radius.get("rural", 10.0),
                max_recommendations_per_cluster=rec_config.get("max_recommendations_per_cluster", 3),
                min_expected_rsrp_improvement_db=rec_config.get("min_expected_rsrp_improvement_db", 3.0),
                new_site_threshold_db=rec_config.get("new_site_threshold_db", 10.0),
                hpbw_v_deg=rf_config.get("hpbw_v_deg", 6.5),
                sla_v_db=rf_config.get("sla_v_db", 30.0),
                path_loss_exponent_urban=ple_config.get("urban", 4.0),
                path_loss_exponent_suburban=ple_config.get("suburban", 3.5),
                path_loss_exponent_rural=ple_config.get("rural", 3.0),
                default_antenna_height_m=rf_config.get("default_antenna_height_m", 30.0),
            )
        except Exception as e:
            logger.warning("failed_to_load_recommendation_config", error=str(e))
            return cls()


class LowCoverageRecommender:
    """
    Generates recommendations for resolving low coverage clusters.

    For each low coverage cluster, identifies candidate cells that could
    extend coverage via uptilt and estimates the expected RSRP improvement.

    Example:
        >>> recommender = LowCoverageRecommender()
        >>> recommendations = recommender.generate_recommendations(
        ...     low_coverage_gdf, gis_df, grid_df, environment_df
        ... )
    """

    def __init__(self, params: Optional[LowCoverageRecommendationParams] = None):
        """Initialize recommender with parameters."""
        self.params = params or LowCoverageRecommendationParams()
        logger.info(
            "low_coverage_recommender_initialized",
            max_recommendations=self.params.max_recommendations_per_cluster,
            min_rsrp_improvement=self.params.min_expected_rsrp_improvement_db,
        )

    def generate_recommendations(
        self,
        low_coverage_gdf: gpd.GeoDataFrame,
        gis_df: pd.DataFrame,
        grid_df: pd.DataFrame,
        environment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate uptilt recommendations for low coverage clusters.

        Args:
            low_coverage_gdf: GeoDataFrame with low coverage cluster polygons
            gis_df: Cell GIS data with tilt, height, coordinates
            grid_df: Grid measurements with RSRP
            environment_df: Optional environment classification per cell

        Returns:
            DataFrame with recommendations:
            - cluster_id, band: Cluster identification
            - recommended_cell: Cell to uptilt
            - current_tilt_deg: Current total tilt
            - recommended_uptilt_deg: Suggested uptilt (1 or 2 degrees)
            - expected_rsrp_gain_db: Predicted RSRP improvement
            - distance_to_cluster_m: Distance from cell to cluster centroid
            - confidence: HIGH/MEDIUM/LOW
            - note: Any special notes (e.g., "NEW_SITE_REQUIRED")
        """
        if len(low_coverage_gdf) == 0:
            logger.info("no_low_coverage_clusters_to_recommend")
            return pd.DataFrame()

        logger.info(
            "generating_low_coverage_recommendations",
            clusters=len(low_coverage_gdf),
        )

        # Build cell environment map
        cell_env_map = {}
        if environment_df is not None and 'cell_name' in environment_df.columns:
            cell_env_map = dict(zip(
                environment_df['cell_name'].astype(str),
                environment_df['environment'].str.lower()
            ))

        # Build BallTree for cell locations
        gis_df = gis_df.copy()
        gis_df['cell_name'] = gis_df['cell_name'].astype(str)
        cell_coords = gis_df[['latitude', 'longitude']].values
        cell_coords_rad = np.radians(cell_coords)
        cell_tree = BallTree(cell_coords_rad, metric='haversine')

        all_recommendations = []

        for _, cluster in low_coverage_gdf.iterrows():
            cluster_recs = self._recommend_for_cluster(
                cluster,
                gis_df,
                grid_df,
                cell_tree,
                cell_env_map,
            )
            all_recommendations.extend(cluster_recs)

        if len(all_recommendations) == 0:
            logger.info("no_recommendations_generated")
            return pd.DataFrame()

        recommendations_df = pd.DataFrame(all_recommendations)

        # Sort by cluster then by expected gain
        recommendations_df = recommendations_df.sort_values(
            ['cluster_id', 'expected_rsrp_gain_db'],
            ascending=[True, False]
        )

        logger.info(
            "recommendations_generated",
            total_recommendations=len(recommendations_df),
            clusters_with_recommendations=recommendations_df['cluster_id'].nunique(),
            new_site_required=len(recommendations_df[recommendations_df['note'] == 'NEW_SITE_REQUIRED']),
        )

        return recommendations_df

    def _recommend_for_cluster(
        self,
        cluster: pd.Series,
        gis_df: pd.DataFrame,
        grid_df: pd.DataFrame,
        cell_tree: BallTree,
        cell_env_map: dict,
    ) -> List[dict]:
        """Generate recommendations for a single cluster."""
        cluster_id = cluster['cluster_id']
        band = cluster.get('band', 'unknown')
        centroid_lat = cluster['centroid_lat']
        centroid_lon = cluster['centroid_lon']
        rsrp_threshold = cluster.get('avg_rsrp_dbm', -115)
        rsrp_deficit = cluster.get('rsrp_deficit_db', 10)

        # Determine environment from serving cells
        serving_cells = cluster.get('serving_cells', '')
        environment = self._determine_cluster_environment(serving_cells, cell_env_map)

        # Get search radius based on environment
        search_radius_km = self._get_search_radius(environment)
        search_radius_rad = search_radius_km / EARTH_RADIUS_KM

        # Find candidate cells within search radius
        centroid_rad = np.radians([[centroid_lat, centroid_lon]])
        indices = cell_tree.query_radius(centroid_rad, r=search_radius_rad)[0]

        if len(indices) == 0:
            # No cells in range - flag for new site
            return [{
                'cluster_id': cluster_id,
                'band': band,
                'recommended_cell': None,
                'current_tilt_deg': None,
                'recommended_uptilt_deg': None,
                'expected_rsrp_gain_db': 0.0,
                'distance_to_cluster_m': None,
                'confidence': 'N/A',
                'note': 'NEW_SITE_REQUIRED',
            }]

        # Evaluate each candidate cell
        candidate_scores = []
        for idx in indices:
            cell_row = gis_df.iloc[idx]
            cell_name = str(cell_row['cell_name'])

            # Filter to same band if possible
            cell_band = cell_row.get('band', '')
            if cell_band and band and self._normalise_band(cell_band) != band:
                continue

            # Get cell parameters
            tilt_mech = cell_row.get('tilt_mech', 0) or 0
            tilt_elc = cell_row.get('tilt_elc', 0) or 0
            current_tilt = tilt_mech + tilt_elc
            antenna_height = cell_row.get('antenna_height', self.params.default_antenna_height_m) or self.params.default_antenna_height_m

            # Calculate distance to cluster
            cell_lat = cell_row['latitude']
            cell_lon = cell_row['longitude']
            distance_m = self._haversine_distance(cell_lat, cell_lon, centroid_lat, centroid_lon)

            # Get path loss exponent for cell's environment
            cell_env = cell_env_map.get(cell_name, environment)
            path_loss_exp = self._get_path_loss_exponent(cell_env)

            # Check if cell can uptilt (tilt > 0)
            if current_tilt <= 0:
                # Cell at minimum tilt - can still flag but with constraint note
                candidate_scores.append({
                    'cell_name': cell_name,
                    'current_tilt': current_tilt,
                    'distance_m': distance_m,
                    'rsrp_gain_1deg': 0,
                    'rsrp_gain_2deg': 0,
                    'constraint': 'MIN_TILT_REACHED',
                })
                continue

            # Estimate RSRP gain from uptilt
            rsrp_gain_1deg = self._estimate_rsrp_gain(
                distance_m, current_tilt, antenna_height, -1.0, path_loss_exp
            )
            rsrp_gain_2deg = self._estimate_rsrp_gain(
                distance_m, current_tilt, antenna_height, -2.0, path_loss_exp
            )

            candidate_scores.append({
                'cell_name': cell_name,
                'current_tilt': current_tilt,
                'distance_m': distance_m,
                'rsrp_gain_1deg': rsrp_gain_1deg,
                'rsrp_gain_2deg': rsrp_gain_2deg,
                'constraint': None,
            })

        if len(candidate_scores) == 0:
            return [{
                'cluster_id': cluster_id,
                'band': band,
                'recommended_cell': None,
                'current_tilt_deg': None,
                'recommended_uptilt_deg': None,
                'expected_rsrp_gain_db': 0.0,
                'distance_to_cluster_m': None,
                'confidence': 'N/A',
                'note': 'NO_SAME_BAND_CELLS_IN_RANGE',
            }]

        # Rank candidates by expected RSRP gain
        candidate_scores.sort(key=lambda x: max(x['rsrp_gain_1deg'], x['rsrp_gain_2deg']), reverse=True)

        # Generate top N recommendations
        recommendations = []
        for i, candidate in enumerate(candidate_scores[:self.params.max_recommendations_per_cluster]):
            # Determine recommended uptilt
            if candidate['constraint'] == 'MIN_TILT_REACHED':
                rec_uptilt = 0
                rsrp_gain = 0
                note = 'PHYSICAL_CONSTRAINT - Cell at 0° tilt'
                confidence = 'N/A'
            elif candidate['rsrp_gain_2deg'] >= self.params.min_expected_rsrp_improvement_db:
                rec_uptilt = 2
                rsrp_gain = candidate['rsrp_gain_2deg']
                note = None
                confidence = self._calculate_confidence(rsrp_gain, rsrp_deficit)
            elif candidate['rsrp_gain_1deg'] >= self.params.min_expected_rsrp_improvement_db:
                rec_uptilt = 1
                rsrp_gain = candidate['rsrp_gain_1deg']
                note = None
                confidence = self._calculate_confidence(rsrp_gain, rsrp_deficit)
            else:
                rec_uptilt = 0
                rsrp_gain = max(candidate['rsrp_gain_1deg'], candidate['rsrp_gain_2deg'])
                note = 'INSUFFICIENT_GAIN'
                confidence = 'LOW'

            # Check if this resolves the gap
            if rsrp_gain < rsrp_deficit - self.params.new_site_threshold_db and i == 0:
                note = 'NEW_SITE_MAY_BE_REQUIRED'

            recommendations.append({
                'cluster_id': cluster_id,
                'band': band,
                'recommended_cell': candidate['cell_name'],
                'current_tilt_deg': candidate['current_tilt'],
                'recommended_uptilt_deg': rec_uptilt,
                'expected_rsrp_gain_db': round(rsrp_gain, 1),
                'distance_to_cluster_m': round(candidate['distance_m'], 0),
                'confidence': confidence,
                'note': note,
            })

        return recommendations

    def _determine_cluster_environment(self, serving_cells: str, cell_env_map: dict) -> str:
        """Determine cluster environment from serving cells."""
        if not serving_cells:
            return 'suburban'

        cell_list = serving_cells.split(',')
        environments = [cell_env_map.get(c.strip(), 'suburban') for c in cell_list]

        # Return most common environment
        if environments:
            return max(set(environments), key=environments.count)
        return 'suburban'

    def _get_search_radius(self, environment: str) -> float:
        """Get search radius based on environment."""
        if environment == 'urban':
            return self.params.search_radius_urban_km
        elif environment == 'rural':
            return self.params.search_radius_rural_km
        else:
            return self.params.search_radius_suburban_km

    def _get_path_loss_exponent(self, environment: str) -> float:
        """Get path loss exponent based on environment."""
        if environment == 'urban':
            return self.params.path_loss_exponent_urban
        elif environment == 'rural':
            return self.params.path_loss_exponent_rural
        else:
            return self.params.path_loss_exponent_suburban

    def _estimate_rsrp_gain(
        self,
        distance_m: float,
        current_tilt_deg: float,
        antenna_height_m: float,
        delta_tilt_deg: float,
        path_loss_exponent: float,
    ) -> float:
        """
        Estimate RSRP gain from tilt change using 3GPP antenna model.

        Uses same model as undershooting detector for consistency.
        """
        if distance_m <= 0 or antenna_height_m <= 0:
            return 0.0

        import math

        # Elevation angle from site to cluster
        theta_e_deg = math.degrees(math.atan2(antenna_height_m, distance_m))

        # 3GPP vertical attenuation before/after
        A_before = self._vertical_attenuation(theta_e_deg, current_tilt_deg)
        A_after = self._vertical_attenuation(theta_e_deg, current_tilt_deg + delta_tilt_deg)

        # Gain change (dB) - negative attenuation change = positive gain
        delta_gain_db = -(A_after - A_before)

        # For uptilt (negative delta), we expect positive gain at far distances
        return max(0.0, delta_gain_db)

    def _vertical_attenuation(self, theta_deg: float, alpha_deg: float) -> float:
        """3GPP parabolic attenuation in vertical plane (dB)."""
        return min(
            12.0 * (((theta_deg - alpha_deg) / self.params.hpbw_v_deg) ** 2),
            self.params.sla_v_db
        )

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        import math
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_M * c

    def _calculate_confidence(self, rsrp_gain: float, rsrp_deficit: float) -> str:
        """Calculate confidence level based on expected vs required gain."""
        if rsrp_gain >= rsrp_deficit:
            return 'HIGH'
        elif rsrp_gain >= rsrp_deficit * 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    @staticmethod
    def _normalise_band(band_value) -> str:
        """Normalise band value to 'L800' format."""
        if pd.isna(band_value):
            return None
        band_str = str(band_value).upper().strip()
        if band_str.endswith('.0'):
            band_str = band_str[:-2]
        if not band_str.startswith('L'):
            band_str = 'L' + band_str
        return band_str
