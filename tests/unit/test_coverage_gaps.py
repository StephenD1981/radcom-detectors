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
    LowCoverageRecommender,
    BAND_RSRP_THRESHOLDS,
    MIN_SAMPLE_COUNT_BY_ENV,
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


# ============================================================================
# Telecom Review Fix Tests
# ============================================================================

class TestBandRsrpThresholds:
    """Tests for band-specific RSRP thresholds (Telecom Review Fix #1)."""

    def test_sub1ghz_stricter_than_midband(self):
        """Verify L700/L800 thresholds are stricter (higher) than L1800/L2100."""
        # Sub-1GHz should have stricter (higher/less negative) thresholds
        for env in ['urban', 'suburban', 'rural']:
            assert BAND_RSRP_THRESHOLDS['L700'][env] > BAND_RSRP_THRESHOLDS['L1800'][env]
            assert BAND_RSRP_THRESHOLDS['L800'][env] > BAND_RSRP_THRESHOLDS['L2100'][env]

    def test_midband_stricter_than_highband(self):
        """Verify mid-band thresholds are stricter than high-band (5G NR C-band)."""
        for env in ['urban', 'suburban', 'rural']:
            assert BAND_RSRP_THRESHOLDS['L1800'][env] > BAND_RSRP_THRESHOLDS['L3500'][env]
            assert BAND_RSRP_THRESHOLDS['L2100'][env] > BAND_RSRP_THRESHOLDS['L3700'][env]

    def test_urban_stricter_than_rural(self):
        """Verify urban thresholds are stricter than rural for each band."""
        for band in BAND_RSRP_THRESHOLDS:
            assert BAND_RSRP_THRESHOLDS[band]['urban'] > BAND_RSRP_THRESHOLDS[band]['rural']
            assert BAND_RSRP_THRESHOLDS[band]['suburban'] > BAND_RSRP_THRESHOLDS[band]['rural']
            assert BAND_RSRP_THRESHOLDS[band]['urban'] > BAND_RSRP_THRESHOLDS[band]['suburban']

    def test_5g_nr_bands_present(self):
        """Verify 5G NR bands are included in thresholds."""
        nr_bands = ['L3500', 'L3700', 'N1', 'N3', 'N7', 'N28', 'N78', 'N77']
        for band in nr_bands:
            assert band in BAND_RSRP_THRESHOLDS, f"Missing 5G NR band: {band}"
            assert 'urban' in BAND_RSRP_THRESHOLDS[band]
            assert 'suburban' in BAND_RSRP_THRESHOLDS[band]
            assert 'rural' in BAND_RSRP_THRESHOLDS[band]

    def test_threshold_values_in_valid_range(self):
        """Verify all threshold values are within valid RSRP range."""
        for band, envs in BAND_RSRP_THRESHOLDS.items():
            for env, threshold in envs.items():
                assert -140 <= threshold <= -70, f"Invalid threshold {threshold} for {band}/{env}"


class TestMinSampleCountByEnv:
    """Tests for environment-aware min_sample_count (Telecom Review Fix #3)."""

    def test_rural_has_lowest_threshold(self):
        """Verify rural has lowest sample count requirement."""
        assert MIN_SAMPLE_COUNT_BY_ENV['rural'] < MIN_SAMPLE_COUNT_BY_ENV['suburban']
        assert MIN_SAMPLE_COUNT_BY_ENV['rural'] < MIN_SAMPLE_COUNT_BY_ENV['urban']

    def test_urban_has_highest_threshold(self):
        """Verify urban has highest sample count requirement."""
        assert MIN_SAMPLE_COUNT_BY_ENV['urban'] > MIN_SAMPLE_COUNT_BY_ENV['suburban']
        assert MIN_SAMPLE_COUNT_BY_ENV['urban'] > MIN_SAMPLE_COUNT_BY_ENV['rural']

    def test_expected_values(self):
        """Verify expected default values."""
        assert MIN_SAMPLE_COUNT_BY_ENV['urban'] == 10
        assert MIN_SAMPLE_COUNT_BY_ENV['suburban'] == 5
        assert MIN_SAMPLE_COUNT_BY_ENV['rural'] == 2

    def test_all_positive(self):
        """Verify all sample counts are positive."""
        for env, count in MIN_SAMPLE_COUNT_BY_ENV.items():
            assert count > 0, f"Sample count for {env} must be positive"


class TestLowCoverageParamsSparseAreaConfig:
    """Tests for configurable sparse area thresholds (Telecom Review Fix #2 config)."""

    def test_sparse_area_params_exist(self):
        """Verify sparse area parameters exist in LowCoverageParams."""
        params = LowCoverageParams()
        assert hasattr(params, 'min_measured_neighbors_absolute')
        assert hasattr(params, 'min_measured_neighbors_pct')
        assert hasattr(params, 'min_low_rsrp_evidence_absolute')
        assert hasattr(params, 'min_low_rsrp_evidence_pct')

    def test_sparse_area_defaults(self):
        """Verify default values for sparse area parameters."""
        params = LowCoverageParams()
        assert params.min_measured_neighbors_absolute == 5
        assert params.min_measured_neighbors_pct == 0.2
        assert params.min_low_rsrp_evidence_absolute == 2
        assert params.min_low_rsrp_evidence_pct == 0.1

    def test_sparse_area_custom_values(self):
        """Test custom sparse area parameter initialization."""
        params = LowCoverageParams(
            min_measured_neighbors_absolute=10,
            min_measured_neighbors_pct=0.3,
            min_low_rsrp_evidence_absolute=5,
            min_low_rsrp_evidence_pct=0.15
        )
        assert params.min_measured_neighbors_absolute == 10
        assert params.min_measured_neighbors_pct == 0.3
        assert params.min_low_rsrp_evidence_absolute == 5
        assert params.min_low_rsrp_evidence_pct == 0.15


class TestNeighborImpactRiskAssessment:
    """Tests for neighbor impact risk assessment (Telecom Review Fix #4)."""

    @pytest.fixture
    def sample_gis_df(self):
        """Create sample GIS data for recommender tests."""
        return pd.DataFrame({
            'cell_name': ['CELL001', 'CELL002', 'CELL003'],
            'latitude': [53.35, 53.36, 53.37],
            'longitude': [-6.26, -6.27, -6.28],
            'tilt_mech': [2, 3, 1],
            'tilt_elc': [4, 5, 2],
            'antenna_height': [30, 25, 35],
            'band': ['L800', 'L800', 'L800'],
        })

    @pytest.fixture
    def sample_grid_df(self):
        """Create sample grid data for recommender tests."""
        return pd.DataFrame({
            'cell_name': ['CELL001'] * 10 + ['CELL002'] * 10,
            'grid': [f'gc7abc{i}' for i in range(10)] + [f'gc7def{i}' for i in range(10)],
            'avg_rsrp': [-100, -105, -110, -115, -120, -108, -112, -95, -98, -102,
                        -105, -110, -115, -120, -125, -118, -122, -100, -103, -107],
            'distance_to_cell': [500, 1000, 1500, 2000, 2500, 800, 1200, 300, 600, 900,
                                600, 1200, 1800, 2400, 3000, 1000, 1500, 400, 700, 1100],
        })

    def test_recommender_initialization(self):
        """Test LowCoverageRecommender can be initialized."""
        recommender = LowCoverageRecommender()
        assert recommender is not None
        assert hasattr(recommender, '_assess_neighbor_impact_risk')

    def test_assess_neighbor_impact_risk_method_exists(self):
        """Verify the neighbor impact risk assessment method exists."""
        recommender = LowCoverageRecommender()
        assert callable(getattr(recommender, '_assess_neighbor_impact_risk', None))

    def test_assess_neighbor_impact_risk_returns_valid_level(self, sample_grid_df):
        """Test that risk assessment returns valid risk levels."""
        recommender = LowCoverageRecommender()

        # Test with sample data
        risk = recommender._assess_neighbor_impact_risk(
            cell_name='CELL001',
            grid_df=sample_grid_df,
            target_distance_m=2000,
            current_tilt=6,
            antenna_height=30
        )

        assert risk in ['LOW', 'MEDIUM', 'HIGH']

    def test_assess_neighbor_impact_risk_missing_cell(self, sample_grid_df):
        """Test risk assessment with non-existent cell returns LOW."""
        recommender = LowCoverageRecommender()

        risk = recommender._assess_neighbor_impact_risk(
            cell_name='NONEXISTENT',
            grid_df=sample_grid_df,
            target_distance_m=2000,
            current_tilt=6,
            antenna_height=30
        )

        assert risk == 'LOW'

    def test_assess_neighbor_impact_risk_empty_df(self):
        """Test risk assessment with empty DataFrame returns LOW."""
        recommender = LowCoverageRecommender()
        empty_df = pd.DataFrame(columns=['cell_name', 'grid', 'avg_rsrp', 'distance_to_cell'])

        risk = recommender._assess_neighbor_impact_risk(
            cell_name='CELL001',
            grid_df=empty_df,
            target_distance_m=2000,
            current_tilt=6,
            antenna_height=30
        )

        assert risk == 'LOW'

    def test_low_tilt_increases_risk(self, sample_grid_df):
        """Test that low tilt (<4 degrees) increases risk score."""
        recommender = LowCoverageRecommender()

        # Low tilt (should be higher risk)
        risk_low_tilt = recommender._assess_neighbor_impact_risk(
            cell_name='CELL001',
            grid_df=sample_grid_df,
            target_distance_m=2000,
            current_tilt=2,  # Very low tilt
            antenna_height=30
        )

        # High tilt (should be lower risk)
        risk_high_tilt = recommender._assess_neighbor_impact_risk(
            cell_name='CELL001',
            grid_df=sample_grid_df,
            target_distance_m=2000,
            current_tilt=10,  # High tilt
            antenna_height=30
        )

        # Low tilt should have equal or higher risk than high tilt
        risk_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        assert risk_order[risk_low_tilt] >= risk_order[risk_high_tilt]
