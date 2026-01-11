"""
Tests for interference detection and root cause analysis.

Tests cover:
- InterferenceParams initialization and config loading
- InterferenceDetector detection pipeline
- RootCauseParams initialization
- InterferenceRootCauseAnalyzer root cause identification
- Convenience functions
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
import tempfile
import json

from ran_optimizer.recommendations.interference import (
    InterferenceParams,
    InterferenceDetector,
    RootCauseParams,
    InterferenceRootCauseAnalyzer,
    detect_interference,
    analyze_interference_root_causes,
    MAX_RECORDS_PER_BAND,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_grid_data():
    """Create sample grid data for testing."""
    # Create data simulating interference scenario
    # Multiple cells with similar RSRP in same grids
    # Using valid geohash characters (base32: 0-9, bcdefghjkmnpqrstuvwxyz - no 'a', 'i', 'l', 'o')
    data = []
    grids = ['gc7x9r7', 'gc7x9r8', 'gc7x9r9', 'gc7x9rb', 'gc7x9rc',
             'gc7x9rd', 'gc7x9re', 'gc7x9rf', 'gc7x9rg', 'gc7x9rh']
    cells = ['CellA', 'CellB', 'CellC', 'CellD', 'CellE']

    for grid in grids:
        for cell in cells:
            # All cells have similar RSRP (within 5 dB)
            rsrp = -95 + np.random.uniform(-2, 2)
            data.append({
                'grid': grid,
                'cell_name': cell,
                'avg_rsrp': rsrp,
                'band': 'L1800',
                'event_count': 50,
                'perc_grid_events': 0.2,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_grid_data_with_sinr():
    """Create sample grid data with SINR column."""
    # Using valid geohash characters (base32: no 'a', 'i', 'l', 'o')
    data = []
    grids = ['gc7x9r7', 'gc7x9r8', 'gc7x9r9', 'gc7x9rb', 'gc7x9rc']
    cells = ['CellA', 'CellB', 'CellC', 'CellD']

    for grid in grids:
        for cell in cells:
            rsrp = -95 + np.random.uniform(-2, 2)
            sinr = -2 + np.random.uniform(-1, 1)  # Low SINR indicating interference
            data.append({
                'grid': grid,
                'cell_name': cell,
                'avg_rsrp': rsrp,
                'avg_sinr': sinr,
                'band': 'L1800',
                'event_count': 50,
                'perc_grid_events': 0.25,
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_gis_data():
    """Create sample GIS cell configuration data."""
    return pd.DataFrame({
        'cell_name': ['CellA', 'CellB', 'CellC', 'CellD', 'CellE'],
        'latitude': [53.350, 53.351, 53.352, 53.349, 53.348],
        'longitude': [-6.260, -6.261, -6.259, -6.258, -6.262],
        'bearing': [90, 95, 180, 270, 45],  # Azimuths
        'tilt_elc': [2, 3, 6, 8, 2],  # Electrical tilts
        'tilt_mech': [0, 0, 0, 0, 1],  # Mechanical tilts
        'tx_power': [43, 43, 43, 40, 46],  # dBm
    })


@pytest.fixture
def sample_interference_gdf():
    """Create sample interference GeoDataFrame for root cause testing."""
    return gpd.GeoDataFrame({
        'cluster_id': ['L1800_0', 'L1800_1'],
        'band': ['L1800', 'L1800'],
        'n_grids': [25, 15],
        'n_cells': [4, 3],
        'cells': [['CellA', 'CellB', 'CellC', 'CellD'], ['CellC', 'CellD', 'CellE']],
        'centroid_lat': [53.35, 53.36],
        'centroid_lon': [-6.26, -6.27],
        'avg_rsrp': [-95.0, -98.0],
        'area_km2': [2.5, 1.8],
        'geometry': [
            Polygon([(-6.27, 53.34), (-6.25, 53.34), (-6.25, 53.36), (-6.27, 53.36)]),
            Polygon([(-6.28, 53.35), (-6.26, 53.35), (-6.26, 53.37), (-6.28, 53.37)])
        ]
    }, crs='EPSG:4326')


# ============================================================================
# InterferenceParams Tests
# ============================================================================

class TestInterferenceParams:
    """Tests for InterferenceParams dataclass."""

    def test_defaults(self):
        """Test default parameter initialization."""
        params = InterferenceParams()

        assert params.min_filtered_cells_per_grid == 4
        assert params.min_cell_event_count == 2
        assert params.perc_grid_events == 0.05
        assert params.dominant_perc_grid_events == 0.3
        assert params.dominance_diff == 5.0
        assert params.max_rsrp_diff == 5.0
        assert params.sinr_threshold_db == 0.0
        assert params.k == 3
        assert params.perc_interference == 0.33
        assert params.hdbscan_min_cluster_size == 5
        assert params.alpha_shape_alpha is None
        assert params.max_alphashape_points == 2000
        assert params.environment_overrides is None

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = InterferenceParams(
            min_filtered_cells_per_grid=5,
            max_rsrp_diff=6.0,
            sinr_threshold_db=-3.0,
            k=2
        )

        assert params.min_filtered_cells_per_grid == 5
        assert params.max_rsrp_diff == 6.0
        assert params.sinr_threshold_db == -3.0
        assert params.k == 2
        # Other params should be defaults
        assert params.dominance_diff == 5.0

    def test_from_config_file(self):
        """Test loading parameters from JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"

            config_data = {
                "default": {
                    "min_filtered_cells_per_grid": 5,
                    "max_rsrp_diff_db": 6.0,
                    "sinr_threshold_db": -2.0,
                    "k_ring_steps": 2,
                    "clustering": {
                        "algo": "fixed",
                        "fixed_width_db": 4.0
                    },
                    "polygon_clustering": {
                        "hdbscan_min_cluster_size": 10
                    }
                },
                "environment_overrides": {
                    "urban": {
                        "min_filtered_cells_per_grid": 6
                    }
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            params = InterferenceParams.from_config(str(config_path))

            assert params.min_filtered_cells_per_grid == 5
            assert params.max_rsrp_diff == 6.0
            assert params.sinr_threshold_db == -2.0
            assert params.k == 2
            assert params.fixed_width == 4.0
            assert params.hdbscan_min_cluster_size == 10
            assert params.environment_overrides == {"urban": {"min_filtered_cells_per_grid": 6}}

    def test_from_config_missing_file(self):
        """Test that missing config file returns defaults."""
        params = InterferenceParams.from_config("/nonexistent/path/config.json")

        # Should return default values
        assert params.min_filtered_cells_per_grid == 4
        assert params.max_rsrp_diff == 5.0


# ============================================================================
# InterferenceDetector Tests
# ============================================================================

class TestInterferenceDetector:
    """Tests for InterferenceDetector class."""

    def test_initialization_default(self):
        """Test detector initialization with default params."""
        detector = InterferenceDetector()

        assert detector.params is not None
        assert detector.params.min_filtered_cells_per_grid == 4

    def test_initialization_custom_params(self):
        """Test detector initialization with custom params."""
        params = InterferenceParams(min_filtered_cells_per_grid=6)
        detector = InterferenceDetector(params)

        assert detector.params.min_filtered_cells_per_grid == 6

    def test_detect_returns_geodataframe(self, sample_grid_data):
        """Test that detect returns a GeoDataFrame."""
        detector = InterferenceDetector()
        result = detector.detect(sample_grid_data)

        assert isinstance(result, gpd.GeoDataFrame)

    def test_detect_empty_input(self):
        """Test detect with empty DataFrame raises error."""
        detector = InterferenceDetector()
        empty_df = pd.DataFrame(columns=['cell_name', 'avg_rsrp', 'grid', 'band'])

        with pytest.raises(ValueError, match="empty"):
            detector.detect(empty_df)

    def test_detect_missing_columns(self):
        """Test detect with missing required columns raises error."""
        detector = InterferenceDetector()
        df = pd.DataFrame({
            'cell_name': ['A'],
            'avg_rsrp': [-90],
            # Missing 'grid' and 'band'
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df)

    def test_detect_invalid_data_type(self, sample_grid_data):
        """Test detect with invalid data_type raises error."""
        detector = InterferenceDetector()

        with pytest.raises(ValueError, match="Invalid data_type"):
            detector.detect(sample_grid_data, data_type='invalid')

    def test_detect_result_columns(self, sample_grid_data):
        """Test that result has expected columns."""
        detector = InterferenceDetector()
        result = detector.detect(sample_grid_data)

        expected_cols = ['cluster_id', 'band', 'n_grids', 'n_cells', 'cells',
                        'centroid_lat', 'centroid_lon', 'area_km2', 'avg_rsrp', 'geometry']

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_detect_with_sinr_data(self, sample_grid_data_with_sinr):
        """Test detection with SINR data includes avg_sinr column."""
        detector = InterferenceDetector()
        result = detector.detect(sample_grid_data_with_sinr)

        # If clusters found, should have avg_sinr
        if len(result) > 0:
            assert 'avg_sinr' in result.columns

    def test_detect_with_environment(self, sample_grid_data):
        """Test detection with environment parameter."""
        params = InterferenceParams(
            environment_overrides={
                'urban': {'min_filtered_cells_per_grid': 3}
            }
        )
        detector = InterferenceDetector(params)

        # Should not raise
        result = detector.detect(sample_grid_data, environment='urban')
        assert isinstance(result, gpd.GeoDataFrame)

    def test_detect_no_interference(self):
        """Test detection when no interference patterns exist."""
        # Create data with clear dominant cell per grid
        data = []
        for i, grid in enumerate(['u0v8g7', 'u0v8g8', 'u0v8g9']):
            # Dominant cell
            data.append({
                'grid': grid,
                'cell_name': f'Cell{i}',
                'avg_rsrp': -85,
                'band': 'L1800',
                'event_count': 100,
                'perc_grid_events': 0.8,
            })
            # Weak cell
            data.append({
                'grid': grid,
                'cell_name': f'CellWeak{i}',
                'avg_rsrp': -110,  # Much weaker
                'band': 'L1800',
                'event_count': 10,
                'perc_grid_events': 0.1,
            })

        df = pd.DataFrame(data)
        detector = InterferenceDetector()
        result = detector.detect(df)

        # Should return empty or very few clusters
        assert len(result) == 0 or result['n_grids'].sum() < 3

    def test_memory_safeguard_constant(self):
        """Test that MAX_RECORDS_PER_BAND constant exists."""
        assert MAX_RECORDS_PER_BAND == 5_000_000


class TestInterferenceDetectorHelpers:
    """Tests for InterferenceDetector helper methods."""

    def test_calculate_areas_batch(self, sample_interference_gdf):
        """Test batch area calculation."""
        detector = InterferenceDetector()
        areas = detector._calculate_areas_batch(sample_interference_gdf)

        assert len(areas) == len(sample_interference_gdf)
        assert all(a >= 0 for a in areas)

    def test_calculate_areas_batch_empty(self):
        """Test batch area calculation with empty GeoDataFrame."""
        detector = InterferenceDetector()
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        areas = detector._calculate_areas_batch(empty_gdf)

        assert areas == []

    def test_calculate_area_km2_single(self):
        """Test single geometry area calculation."""
        detector = InterferenceDetector()
        polygon = Polygon([(-6.27, 53.34), (-6.25, 53.34), (-6.25, 53.36), (-6.27, 53.36)])
        area = detector._calculate_area_km2(polygon)

        assert area > 0
        # Rough check: ~2km x ~2km = ~4 km²
        assert 1 < area < 10

    def test_calculate_area_km2_none(self):
        """Test area calculation with None geometry."""
        detector = InterferenceDetector()
        area = detector._calculate_area_km2(None)

        assert area == 0.0

    def test_create_alpha_shape(self):
        """Test alpha shape creation."""
        detector = InterferenceDetector()
        coords = np.array([
            [-6.26, 53.35],
            [-6.25, 53.35],
            [-6.25, 53.36],
            [-6.26, 53.36],
            [-6.255, 53.355],
        ])
        shape = detector._create_alpha_shape(coords)

        assert shape is not None
        assert not shape.is_empty

    def test_create_alpha_shape_subsampling(self):
        """Test alpha shape with subsampling for large coordinate sets."""
        params = InterferenceParams(max_alphashape_points=10)
        detector = InterferenceDetector(params)

        # Create more points than max_alphashape_points
        n_points = 50
        coords = np.column_stack([
            np.random.uniform(-6.27, -6.25, n_points),
            np.random.uniform(53.34, 53.36, n_points)
        ])

        shape = detector._create_alpha_shape(coords)

        # Should still work with subsampling
        assert shape is not None


# ============================================================================
# RootCauseParams Tests
# ============================================================================

class TestRootCauseParams:
    """Tests for RootCauseParams dataclass."""

    def test_defaults(self):
        """Test default parameter initialization."""
        params = RootCauseParams()

        assert params.low_tilt_threshold_deg == 4.0
        assert params.high_tilt_threshold_deg == 10.0
        assert params.tilt_spread_threshold_deg == 3.0
        assert params.azimuth_convergence_threshold_deg == 60.0
        assert params.close_proximity_km == 0.5
        assert params.normal_proximity_km == 1.5
        assert params.min_tilt_increase_deg == 1.0
        assert params.max_tilt_increase_deg == 4.0

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = RootCauseParams(
            low_tilt_threshold_deg=5.0,
            close_proximity_km=0.3
        )

        assert params.low_tilt_threshold_deg == 5.0
        assert params.close_proximity_km == 0.3
        # Others should be defaults
        assert params.azimuth_convergence_threshold_deg == 60.0


# ============================================================================
# InterferenceRootCauseAnalyzer Tests
# ============================================================================

class TestInterferenceRootCauseAnalyzer:
    """Tests for InterferenceRootCauseAnalyzer class."""

    def test_initialization_default(self):
        """Test analyzer initialization with default params."""
        analyzer = InterferenceRootCauseAnalyzer()

        assert analyzer.params is not None
        assert analyzer.params.low_tilt_threshold_deg == 4.0

    def test_initialization_custom_params(self):
        """Test analyzer initialization with custom params."""
        params = RootCauseParams(low_tilt_threshold_deg=5.0)
        analyzer = InterferenceRootCauseAnalyzer(params)

        assert analyzer.params.low_tilt_threshold_deg == 5.0

    def test_analyze_returns_geodataframe(self, sample_interference_gdf, sample_gis_data):
        """Test that analyze returns a GeoDataFrame with expected columns."""
        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(sample_interference_gdf, sample_gis_data)

        assert isinstance(result, gpd.GeoDataFrame)
        assert 'root_cause' in result.columns
        assert 'root_cause_details' in result.columns
        assert 'recommendations' in result.columns
        assert 'priority' in result.columns

    def test_analyze_empty_input(self, sample_gis_data):
        """Test analyze with empty interference GeoDataFrame."""
        analyzer = InterferenceRootCauseAnalyzer()
        empty_gdf = gpd.GeoDataFrame(columns=['cells', 'geometry'], crs='EPSG:4326')

        result = analyzer.analyze(empty_gdf, sample_gis_data)

        assert len(result) == 0

    def test_analyze_missing_gis_columns(self, sample_interference_gdf):
        """Test analyze with missing required GIS columns raises error."""
        analyzer = InterferenceRootCauseAnalyzer()
        incomplete_gis = pd.DataFrame({'cell_name': ['A']})  # Missing lat/lon

        with pytest.raises(ValueError, match="missing required columns"):
            analyzer.analyze(sample_interference_gdf, incomplete_gis)

    def test_analyze_identifies_low_tilt(self, sample_interference_gdf, sample_gis_data):
        """Test that analyzer identifies low tilt as root cause."""
        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(sample_interference_gdf, sample_gis_data)

        # First cluster has cells with low tilt
        assert result.iloc[0]['root_cause'] == 'low_tilt'
        assert 'low_tilt_cells' in result.iloc[0]['root_cause_details']

    def test_analyze_generates_recommendations(self, sample_interference_gdf, sample_gis_data):
        """Test that analyzer generates recommendations."""
        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(sample_interference_gdf, sample_gis_data)

        # Should have recommendations
        for idx, row in result.iterrows():
            assert isinstance(row['recommendations'], list)

    def test_analyze_priority_assignment(self, sample_interference_gdf, sample_gis_data):
        """Test that analyzer assigns priority correctly."""
        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(sample_interference_gdf, sample_gis_data)

        for priority in result['priority']:
            assert priority in ['high', 'medium', 'low']

    def test_analyze_azimuth_convergence(self):
        """Test detection of azimuth convergence."""
        # Create interference with converging azimuths
        interference_gdf = gpd.GeoDataFrame({
            'cluster_id': ['test_0'],
            'band': ['L1800'],
            'n_grids': [10],
            'n_cells': [3],
            'cells': [['CellX', 'CellY', 'CellZ']],
            'centroid_lat': [53.35],
            'centroid_lon': [-6.26],
            'avg_rsrp': [-95.0],
            'area_km2': [1.0],
            'geometry': [Polygon([(-6.27, 53.34), (-6.25, 53.34), (-6.25, 53.36), (-6.27, 53.36)])]
        }, crs='EPSG:4326')

        # GIS with converging azimuths and adequate tilts
        gis_df = pd.DataFrame({
            'cell_name': ['CellX', 'CellY', 'CellZ'],
            'latitude': [53.35, 53.36, 53.34],
            'longitude': [-6.26, -6.27, -6.25],
            'bearing': [90, 95, 92],  # Very similar azimuths
            'tilt_elc': [6, 7, 6],  # Adequate tilts
            'tilt_mech': [0, 0, 0],
        })

        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(interference_gdf, gis_df)

        # Should identify azimuth convergence
        details = result.iloc[0]['root_cause_details']
        assert 'min_azimuth_diff_deg' in details
        assert details['min_azimuth_diff_deg'] < 60

    def test_analyze_close_proximity(self):
        """Test detection of close cell proximity."""
        interference_gdf = gpd.GeoDataFrame({
            'cluster_id': ['test_0'],
            'band': ['L1800'],
            'n_grids': [10],
            'n_cells': [2],
            'cells': [['CellP', 'CellQ']],
            'centroid_lat': [53.35],
            'centroid_lon': [-6.26],
            'avg_rsrp': [-95.0],
            'area_km2': [1.0],
            'geometry': [Polygon([(-6.27, 53.34), (-6.25, 53.34), (-6.25, 53.36), (-6.27, 53.36)])]
        }, crs='EPSG:4326')

        # GIS with very close cells
        gis_df = pd.DataFrame({
            'cell_name': ['CellP', 'CellQ'],
            'latitude': [53.350, 53.3505],  # Very close
            'longitude': [-6.260, -6.2605],
            'bearing': [90, 270],  # Opposite directions
            'tilt_elc': [6, 6],  # Adequate tilts
            'tilt_mech': [0, 0],
        })

        analyzer = InterferenceRootCauseAnalyzer()
        result = analyzer.analyze(interference_gdf, gis_df)

        details = result.iloc[0]['root_cause_details']
        assert 'min_cell_distance_km' in details
        assert details['min_cell_distance_km'] < 0.5


class TestRootCauseAnalyzerHelpers:
    """Tests for InterferenceRootCauseAnalyzer helper methods."""

    def test_haversine_km(self):
        """Test haversine distance calculation."""
        # Dublin to Belfast is roughly 140 km
        dist = InterferenceRootCauseAnalyzer._haversine_km(53.35, -6.26, 54.60, -5.93)
        assert 130 < dist < 150

    def test_haversine_km_same_point(self):
        """Test haversine with same point returns 0."""
        dist = InterferenceRootCauseAnalyzer._haversine_km(53.35, -6.26, 53.35, -6.26)
        assert dist == 0.0

    def test_find_converging_cells(self):
        """Test finding cells with converging azimuths."""
        analyzer = InterferenceRootCauseAnalyzer()
        cell_configs = [
            {'cell_name': 'A', 'azimuth': 90},
            {'cell_name': 'B', 'azimuth': 95},  # Close to A
            {'cell_name': 'C', 'azimuth': 180},  # Different
        ]

        converging = analyzer._find_converging_cells(cell_configs, 60.0)

        assert 'A' in converging
        assert 'B' in converging
        assert 'C' not in converging

    def test_calculate_cell_distances(self):
        """Test pairwise distance calculation."""
        analyzer = InterferenceRootCauseAnalyzer()
        cell_configs = [
            {'cell_name': 'A', 'lat': 53.35, 'lon': -6.26},
            {'cell_name': 'B', 'lat': 53.36, 'lon': -6.26},  # ~1.1 km north
        ]

        distances = analyzer._calculate_cell_distances(cell_configs)

        assert len(distances) == 1
        assert 1.0 < distances[0] < 1.2

    def test_determine_primary_cause(self):
        """Test primary cause determination priority."""
        analyzer = InterferenceRootCauseAnalyzer()

        # Low tilt should be highest priority
        causes = ['close_proximity', 'low_tilt', 'azimuth_convergence']
        primary = analyzer._determine_primary_cause(causes)
        assert primary == 'low_tilt'

        # Empty list
        primary = analyzer._determine_primary_cause([])
        assert primary == 'undetermined'

    def test_determine_priority(self):
        """Test priority determination."""
        analyzer = InterferenceRootCauseAnalyzer()
        cluster = pd.Series({'n_grids': 10, 'n_cells': 3})

        # High priority for low_tilt
        priority = analyzer._determine_priority(['low_tilt'], {}, cluster)
        assert priority == 'high'

        # Large cluster should be high priority
        large_cluster = pd.Series({'n_grids': 60, 'n_cells': 8})
        priority = analyzer._determine_priority([], {}, large_cluster)
        assert priority == 'high'

    def test_format_recommendations(self):
        """Test recommendation formatting."""
        analyzer = InterferenceRootCauseAnalyzer()

        recommendations = [
            {
                'action': 'increase_tilt',
                'cell': 'CellA',
                'current_tilt': 2.0,
                'suggested_tilt': 4.0,
                'reason': 'Low tilt'
            },
            {
                'action': 'review_site_design',
                'reason': 'Cells too close'
            }
        ]

        formatted = analyzer._format_recommendations(recommendations)

        assert len(formatted) == 2
        assert 'CellA' in formatted[0]
        assert '2.0°' in formatted[0]
        assert 'Site design' in formatted[1]


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_interference_function(self, sample_grid_data):
        """Test detect_interference convenience function."""
        result = detect_interference(sample_grid_data)

        assert isinstance(result, gpd.GeoDataFrame)

    def test_detect_interference_with_params(self, sample_grid_data):
        """Test detect_interference with custom params."""
        params = InterferenceParams(min_filtered_cells_per_grid=3)
        result = detect_interference(sample_grid_data, params=params)

        assert isinstance(result, gpd.GeoDataFrame)

    def test_detect_interference_with_environment(self, sample_grid_data):
        """Test detect_interference with environment parameter."""
        result = detect_interference(sample_grid_data, environment='urban')

        assert isinstance(result, gpd.GeoDataFrame)

    def test_analyze_interference_root_causes_function(self, sample_interference_gdf, sample_gis_data):
        """Test analyze_interference_root_causes convenience function."""
        result = analyze_interference_root_causes(sample_interference_gdf, sample_gis_data)

        assert isinstance(result, gpd.GeoDataFrame)
        assert 'root_cause' in result.columns
        assert 'recommendations' in result.columns

    def test_analyze_interference_root_causes_with_params(self, sample_interference_gdf, sample_gis_data):
        """Test analyze_interference_root_causes with custom params."""
        params = RootCauseParams(low_tilt_threshold_deg=5.0)
        result = analyze_interference_root_causes(sample_interference_gdf, sample_gis_data, params=params)

        assert isinstance(result, gpd.GeoDataFrame)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflow."""

    def test_full_workflow(self, sample_grid_data, sample_gis_data):
        """Test complete detection and analysis workflow."""
        # Step 1: Detect interference
        interference_gdf = detect_interference(sample_grid_data)

        # Step 2: Analyze root causes (if any clusters found)
        if len(interference_gdf) > 0:
            enriched_gdf = analyze_interference_root_causes(interference_gdf, sample_gis_data)

            # Verify enrichment
            assert 'root_cause' in enriched_gdf.columns
            assert 'recommendations' in enriched_gdf.columns

            # All original columns should be preserved
            for col in interference_gdf.columns:
                assert col in enriched_gdf.columns

    def test_workflow_with_sinr(self, sample_grid_data_with_sinr, sample_gis_data):
        """Test workflow with SINR filtering."""
        params = InterferenceParams(sinr_threshold_db=5.0)  # Filter high SINR
        interference_gdf = detect_interference(sample_grid_data_with_sinr, params=params)

        assert isinstance(interference_gdf, gpd.GeoDataFrame)
        # SINR column should be present if clusters found
        if len(interference_gdf) > 0:
            assert 'avg_sinr' in interference_gdf.columns
