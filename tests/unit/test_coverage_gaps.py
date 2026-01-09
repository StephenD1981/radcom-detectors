"""
Tests for coverage gap detection and analysis.

Tests cover:
- CoverageGapParams initialization and config loading
- CoverageGapDetector hull clustering and gap detection
- LowCoverageParams and LowCoverageDetector
- CoverageGapAnalyzer for finding nearby cells
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
import tempfile
import json

from ran_optimizer.recommendations.coverage_gaps import (
    CoverageGapParams,
    CoverageGapDetector,
    CoverageGapAnalyzer,
    LowCoverageParams,
    LowCoverageDetector,
)


# ============================================================================
# CoverageGapParams Tests
# ============================================================================

class TestCoverageGapParams:
    """Tests for CoverageGapParams dataclass."""

    def test_defaults(self):
        """Test default parameter initialization."""
        params = CoverageGapParams()

        assert params.cell_cluster_eps_km == 5.0
        assert params.cell_cluster_min_samples == 3
        assert params.k_ring_steps == 3
        assert params.min_missing_neighbors == 40
        assert params.hdbscan_min_cluster_size == 10
        assert params.alpha_shape_alpha is None
        assert params.k_nearest_cells == 5
        assert params.max_alphashape_points == 5000

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = CoverageGapParams(
            cell_cluster_eps_km=10.0,
            k_ring_steps=2,
            hdbscan_min_cluster_size=20
        )

        assert params.cell_cluster_eps_km == 10.0
        assert params.k_ring_steps == 2
        assert params.hdbscan_min_cluster_size == 20
        # Other params should be defaults
        assert params.cell_cluster_min_samples == 3

    def test_from_config_file(self):
        """Test loading parameters from JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"

            config_data = {
                "coverage_gap_detection": {
                    "cell_cluster_eps_km": 8.0,
                    "k_ring_steps": 4,
                    "min_missing_neighbors": 35,
                    "hdbscan_min_cluster_size": 15
                },
                "environment_overrides": {
                    "urban": {
                        "cell_cluster_eps_km": 3.0,
                        "min_missing_neighbors": 30
                    }
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            # Test default environment
            params = CoverageGapParams.from_config(config_path)
            assert params.cell_cluster_eps_km == 8.0
            assert params.k_ring_steps == 4
            assert params.min_missing_neighbors == 35

            # Test urban environment override
            params_urban = CoverageGapParams.from_config(config_path, environment="urban")
            assert params_urban.cell_cluster_eps_km == 3.0
            assert params_urban.min_missing_neighbors == 30
            # Non-overridden values should come from base
            assert params_urban.k_ring_steps == 4

    def test_from_config_missing_file(self):
        """Test that missing config file returns defaults with warning."""
        params = CoverageGapParams.from_config(Path("nonexistent_config.json"))
        # Should return defaults
        assert params.cell_cluster_eps_km == 5.0
        assert params.k_ring_steps == 3


# ============================================================================
# CoverageGapDetector Tests
# ============================================================================

@pytest.fixture
def sample_hulls():
    """Create sample convex hull polygons for testing."""
    # Create two cell hulls with a gap between them
    hull1 = Polygon([
        (-8.50, 51.90),
        (-8.50, 52.00),
        (-8.40, 52.00),
        (-8.40, 51.90),
        (-8.50, 51.90)
    ])

    hull2 = Polygon([
        (-8.35, 51.90),
        (-8.35, 52.00),
        (-8.25, 52.00),
        (-8.25, 51.90),
        (-8.35, 51.90)
    ])

    # Third hull to make a cluster
    hull3 = Polygon([
        (-8.45, 51.85),
        (-8.45, 51.89),
        (-8.35, 51.89),
        (-8.35, 51.85),
        (-8.45, 51.85)
    ])

    return gpd.GeoDataFrame({
        'cell_name': ['CELL001', 'CELL002', 'CELL003'],
        'geometry': [hull1, hull2, hull3],
    }, crs="EPSG:4326")


class TestCoverageGapDetector:
    """Tests for CoverageGapDetector class."""

    def test_initialization_default(self):
        """Test detector initialization with defaults."""
        detector = CoverageGapDetector()
        assert detector.params.cell_cluster_eps_km == 5.0
        assert detector.params.k_ring_steps == 3

    def test_initialization_custom_params(self):
        """Test detector initialization with custom params."""
        params = CoverageGapParams(cell_cluster_eps_km=10.0, k_ring_steps=2)
        detector = CoverageGapDetector(params)
        assert detector.params.cell_cluster_eps_km == 10.0
        assert detector.params.k_ring_steps == 2

    def test_detect_returns_geodataframe(self, sample_hulls):
        """Test that detect returns a GeoDataFrame."""
        detector = CoverageGapDetector()
        result = detector.detect(sample_hulls)

        assert isinstance(result, gpd.GeoDataFrame)

    def test_detect_empty_hulls(self):
        """Test detection with empty hulls returns empty GeoDataFrame."""
        detector = CoverageGapDetector()
        empty_hulls = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

        result = detector.detect(empty_hulls)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_detect_result_columns(self, sample_hulls):
        """Test that detect result has expected columns."""
        detector = CoverageGapDetector()
        result = detector.detect(sample_hulls)

        # Should have standard columns even if empty
        expected_columns = ['cluster_id', 'n_points', 'centroid_lat', 'centroid_lon', 'geometry']
        for col in expected_columns:
            assert col in result.columns


# ============================================================================
# LowCoverageParams Tests
# ============================================================================

class TestLowCoverageParams:
    """Tests for LowCoverageParams dataclass."""

    def test_defaults(self):
        """Test default parameter initialization."""
        params = LowCoverageParams()

        assert params.rsrp_threshold_dbm == -115
        assert params.k_ring_steps == 3
        assert params.min_missing_neighbors == 40
        assert params.hdbscan_min_cluster_size == 10

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = LowCoverageParams(
            rsrp_threshold_dbm=-110,
            k_ring_steps=2
        )

        assert params.rsrp_threshold_dbm == -110
        assert params.k_ring_steps == 2

    def test_from_config_file(self):
        """Test loading parameters from JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_low_cov.json"

            config_data = {
                "low_coverage_detection": {
                    "rsrp_threshold_dbm": -112,
                    "k_ring_steps": 4,
                    "hdbscan_min_cluster_size": 15
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            params = LowCoverageParams.from_config(config_path)
            assert params.rsrp_threshold_dbm == -112
            assert params.k_ring_steps == 4
            assert params.hdbscan_min_cluster_size == 15


# ============================================================================
# LowCoverageDetector Tests
# ============================================================================

@pytest.fixture
def sample_hulls_with_band():
    """Create sample hulls with band information."""
    hull1 = Polygon([
        (-8.50, 51.90),
        (-8.50, 52.00),
        (-8.40, 52.00),
        (-8.40, 51.90),
    ])

    hull2 = Polygon([
        (-8.35, 51.90),
        (-8.35, 52.00),
        (-8.25, 52.00),
        (-8.25, 51.90),
    ])

    return gpd.GeoDataFrame({
        'cell_name': ['CELL001', 'CELL002'],
        'band': [800, 1800],
        'geometry': [hull1, hull2],
    }, crs="EPSG:4326")


@pytest.fixture
def sample_grid_data():
    """Create sample grid data for low coverage testing."""
    return pd.DataFrame({
        'grid': ['gc4p000', 'gc4p001', 'gc4p002', 'gc4p003'],
        'cell_name': ['CELL001', 'CELL001', 'CELL002', 'CELL002'],
        'band': [800, 800, 1800, 1800],
        'avg_rsrp': [-85, -90, -88, -92],
        'latitude': [51.95, 51.96, 51.95, 51.96],
        'longitude': [-8.45, -8.44, -8.30, -8.29],
    })


@pytest.fixture
def sample_gis_data():
    """Create sample GIS data."""
    return pd.DataFrame({
        'cell_name': ['CELL001', 'CELL002'],
        'Band': ['L800', 'L1800'],
        'latitude': [51.95, 51.95],
        'longitude': [-8.45, -8.30],
    })


class TestLowCoverageDetector:
    """Tests for LowCoverageDetector class."""

    def test_initialization_default(self):
        """Test detector initialization with defaults."""
        detector = LowCoverageDetector()
        assert detector.params.rsrp_threshold_dbm == -115

    def test_initialization_custom_params(self):
        """Test detector initialization with custom params."""
        params = LowCoverageParams(rsrp_threshold_dbm=-110)
        detector = LowCoverageDetector(params)
        assert detector.params.rsrp_threshold_dbm == -110

    def test_detect_returns_dict(self, sample_hulls_with_band, sample_grid_data, sample_gis_data):
        """Test that detect returns a dictionary keyed by band."""
        detector = LowCoverageDetector()
        result = detector.detect(sample_hulls_with_band, sample_grid_data, sample_gis_data)

        assert isinstance(result, dict)

    def test_detect_empty_hulls(self, sample_grid_data, sample_gis_data):
        """Test detection with empty hulls returns empty dict."""
        detector = LowCoverageDetector()
        empty_hulls = gpd.GeoDataFrame(columns=['geometry', 'band'], crs="EPSG:4326")

        result = detector.detect(empty_hulls, sample_grid_data, sample_gis_data)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_detect_specific_bands(self, sample_hulls_with_band, sample_grid_data, sample_gis_data):
        """Test detection for specific bands only."""
        detector = LowCoverageDetector()
        result = detector.detect(
            sample_hulls_with_band,
            sample_grid_data,
            sample_gis_data,
            bands=[800]
        )

        # Should only have band 800 in results
        assert all(band == 800 for band in result.keys())


# ============================================================================
# CoverageGapAnalyzer Tests
# ============================================================================

@pytest.fixture
def sample_gap_clusters():
    """Create sample gap cluster GeoDataFrame."""
    return gpd.GeoDataFrame({
        'cluster_id': [0, 1],
        'centroid_lat': [51.85, 52.05],
        'centroid_lon': [-8.55, -8.15],
        'n_points': [50, 30],
        'geometry': [
            Point(-8.55, 51.85),
            Point(-8.15, 52.05)
        ]
    }, crs="EPSG:4326")


@pytest.fixture
def sample_grid_for_analyzer():
    """Create sample grid data for analyzer testing."""
    return pd.DataFrame({
        'cell_name': ['CELL001', 'CELL001', 'CELL002', 'CELL002', 'CELL003'],
        'latitude': [51.90, 51.88, 52.00, 52.02, 51.95],
        'longitude': [-8.50, -8.48, -8.20, -8.18, -8.35],
        'avg_rsrp': [-85, -88, -90, -92, -87],
    })


class TestCoverageGapAnalyzer:
    """Tests for CoverageGapAnalyzer class."""

    def test_initialization_default(self):
        """Test analyzer initialization with defaults."""
        analyzer = CoverageGapAnalyzer()
        assert analyzer.params.k_nearest_cells == 5

    def test_initialization_custom_params(self):
        """Test analyzer initialization with custom params."""
        params = CoverageGapParams(k_nearest_cells=10)
        analyzer = CoverageGapAnalyzer(params)
        assert analyzer.params.k_nearest_cells == 10

    def test_find_cells_for_gaps(self, sample_gap_clusters, sample_grid_for_analyzer):
        """Test finding cells near gap clusters."""
        analyzer = CoverageGapAnalyzer()
        result = analyzer.find_cells_for_gaps(sample_gap_clusters, sample_grid_for_analyzer)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_gap_clusters)

        # Check expected columns
        assert 'cluster_id' in result.columns
        assert 'nearby_cells' in result.columns
        assert 'nearby_cell_count' in result.columns
        assert 'avg_distance_to_coverage_m' in result.columns

    def test_find_cells_for_gaps_empty_clusters(self, sample_grid_for_analyzer):
        """Test with empty gap clusters."""
        analyzer = CoverageGapAnalyzer()
        empty_clusters = gpd.GeoDataFrame(columns=['cluster_id', 'centroid_lat', 'centroid_lon', 'geometry'])

        result = analyzer.find_cells_for_gaps(empty_clusters, sample_grid_for_analyzer)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_nearby_cells_are_lists(self, sample_gap_clusters, sample_grid_for_analyzer):
        """Test that nearby_cells column contains lists."""
        analyzer = CoverageGapAnalyzer()
        result = analyzer.find_cells_for_gaps(sample_gap_clusters, sample_grid_for_analyzer)

        for cells in result['nearby_cells']:
            assert isinstance(cells, list)

    def test_cell_count_matches_list_length(self, sample_gap_clusters, sample_grid_for_analyzer):
        """Test that nearby_cell_count matches list length."""
        analyzer = CoverageGapAnalyzer()
        result = analyzer.find_cells_for_gaps(sample_gap_clusters, sample_grid_for_analyzer)

        for _, row in result.iterrows():
            assert row['nearby_cell_count'] == len(row['nearby_cells'])
