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
            base_params = config.get("low_coverage_detection", {})

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

            return cls(**base_params)

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

    def detect(
        self,
        hulls: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        gis_data: Optional[pd.DataFrame] = None,
        bands: Optional[List[str]] = None
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Detect low coverage clusters per band.

        Args:
            hulls: Cell convex hulls (requires 'band' column or join with gis_data)
            grid_data: Grid measurements with RSRP per band
            gis_data: Optional GIS data to join for band info
            bands: Optional list of specific bands to process (None = all bands)

        Returns:
            Dict mapping band names to GeoDataFrames of low coverage clusters
            e.g., {'Band 20': gdf, 'Band 3': gdf}

        Example:
            >>> hulls_gdf = load_cell_hulls("data/cell_hulls.csv")
            >>> low_cov = detector.detect(hulls_gdf, grid_data, gis_data)
            >>> for band, clusters in low_cov.items():
            ...     print(f"{band}: {len(clusters)} clusters")
        """
        logger.info("starting_low_coverage_detection", cell_hulls=len(hulls))

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

        band_results = {}
        for band in available_bands:
            logger.info("processing_band", band=band)
            band_clusters = self._detect_band_low_coverage(
                hulls_with_band[hulls_with_band['band'] == band],
                grid_data,
                band,
                gis_data=gis_data
            )

            if len(band_clusters) > 0:
                band_results[band] = band_clusters
                logger.info("band_low_coverage_detected", band=band, clusters=len(band_clusters))

        logger.info("low_coverage_detection_complete", bands_with_low_coverage=len(band_results))
        return band_results

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
        gis_data: Optional[pd.DataFrame] = None
    ) -> gpd.GeoDataFrame:
        """Detect low coverage for a specific band"""
        # Step 1: Find single-server regions (no overlap)
        single_server_polygons = self._find_single_server_regions(band_hulls)

        if len(single_server_polygons) == 0:
            logger.info("no_single_server_regions", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 2: Get geohashes in single-server regions
        candidate_geohashes = self._geohashes_in_polygons(single_server_polygons)

        if len(candidate_geohashes) == 0:
            logger.info("no_candidate_geohashes", band=band)
            return gpd.GeoDataFrame(columns=['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry', 'band'])

        # Step 3: Filter by RSRP threshold
        low_rsrp_geohashes = self._filter_by_rsrp(
            set(candidate_geohashes),
            grid_data,
            band
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

            cluster_records.append({
                'cluster_id': int(cluster_id),
                'n_points': len(cluster_points),
                'centroid_lat': cluster_points['latitude'].mean(),
                'centroid_lon': cluster_points['longitude'].mean(),
                'area_km2': area_km2,
                'serving_cells': ','.join(sorted(serving_cells)[:10]) if serving_cells else '',  # Limit to 10 cells
                'serving_cell_names': ','.join(serving_cell_names) if serving_cell_names else '',
                'n_serving_cells': len(serving_cells),
                'geometry': alpha_shape
            })

        gdf = gpd.GeoDataFrame(cluster_records, crs="EPSG:4326")
        logger.info("cluster_polygons_created_with_cells", clusters=len(gdf))

        return gdf

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
        grid_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Find nearest serving cells for each gap cluster.

        Args:
            gap_clusters: GeoDataFrame with gap cluster polygons
            grid_data: DataFrame with grid measurements (must have cell_id, latitude, longitude)

        Returns:
            DataFrame with gap analysis results including nearby cells

        Example:
            >>> gap_analysis = analyzer.find_cells_for_gaps(gap_clusters, grid_df)
            >>> print(gap_analysis[['cluster_id', 'nearby_cell_count', 'avg_distance_to_coverage_m']])
        """
        if len(gap_clusters) == 0:
            return pd.DataFrame()

        logger.info("analyzing_gap_clusters", clusters=len(gap_clusters))

        # Build BallTree for grid points
        grid_coords = grid_data[['latitude', 'longitude']].values
        grid_coords_rad = np.radians(grid_coords)
        tree = BallTree(grid_coords_rad, metric='haversine')

        results = []

        for _, cluster in gap_clusters.iterrows():
            cluster_id = cluster['cluster_id']
            centroid_lat = cluster['centroid_lat']
            centroid_lon = cluster['centroid_lon']

            # Query k nearest grid points
            centroid_rad = np.radians([[centroid_lat, centroid_lon]])
            distances, indices = tree.query(centroid_rad, k=self.params.k_nearest_cells)

            # Convert distances to meters
            distances_m = distances[0] * EARTH_RADIUS_M

            # Get unique cells using canonical column name
            nearby_rows = grid_data.iloc[indices[0]]
            if 'cell_name' in nearby_rows.columns:
                nearby_cells = nearby_rows['cell_name'].unique().tolist()
            else:
                logger.warning("missing_cell_name_column", columns=list(grid_data.columns[:10]))
                nearby_cells = []

            results.append({
                'cluster_id': cluster_id,
                'nearby_cells': nearby_cells,
                'nearby_cell_count': len(nearby_cells),
                'avg_distance_to_coverage_m': float(distances_m.mean())
            })

        df = pd.DataFrame(results)
        logger.info("gap_analysis_complete", analyzed_clusters=len(df))

        return df
