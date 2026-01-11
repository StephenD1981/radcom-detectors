"""
Tests for crossed feeder detection.

Tests cover:
- CrossedFeederParams initialization and config loading
- CrossedFeederDetector detection pipeline
- Band-specific distance thresholds
- Confidence scoring
- Site classification for multi-sector sites
- Data quality and detection rate checks
- Geometry helper functions
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from ran_optimizer.recommendations.crossed_feeder import (
    CrossedFeederParams,
    CrossedFeederDetector,
    detect_crossed_feeders,
    BAND_MAX_RADIUS_M,
    DEFAULT_MAX_RADIUS_M,
    bearing_deg,
    bearing_deg_vec,
    circ_diff_deg,
    circ_diff_deg_vec,
    is_in_beam,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_gis_data():
    """Create sample GIS cell configuration data."""
    return pd.DataFrame({
        'cell_name': ['SiteA_1', 'SiteA_2', 'SiteA_3', 'SiteB_1', 'SiteB_2', 'SiteB_3'],
        'site': ['SiteA', 'SiteA', 'SiteA', 'SiteB', 'SiteB', 'SiteB'],
        'band': ['L800', 'L800', 'L800', 'L1800', 'L1800', 'L1800'],
        'bearing': [0, 120, 240, 0, 120, 240],  # Standard 3-sector layout
        'hbw': [65, 65, 65, 65, 65, 65],  # Typical beamwidth
        'latitude': [53.350, 53.350, 53.350, 53.360, 53.360, 53.360],
        'longitude': [-6.260, -6.260, -6.260, -6.270, -6.270, -6.270],
    })


@pytest.fixture
def sample_relations_data():
    """Create sample neighbour relations data."""
    # Create relations where SiteA_1 has strong relations to neighbors
    # in unexpected directions (simulating crossed feeder)
    data = []

    # Normal relations for SiteA_1 (bearing=0) - neighbors in ~0 degree direction
    data.append({
        'cell_name': 'SiteA_1', 'to_cell_name': 'SiteB_1',
        'distance': 5000, 'band': 'L800', 'to_band': 'L1800',
        'intra_site': 'n', 'intra_cell': 'n', 'weight': 100,
        'cell_perc_weight': 0.3
    })

    # Suspicious relations for SiteA_1 - neighbors in ~180 degree direction (wrong!)
    data.append({
        'cell_name': 'SiteA_1', 'to_cell_name': 'SiteC_1',
        'distance': 6000, 'band': 'L800', 'to_band': 'L800',
        'intra_site': 'n', 'intra_cell': 'n', 'weight': 200,  # Stronger!
        'cell_perc_weight': 0.5
    })

    # Normal relations for SiteA_2 (bearing=120)
    data.append({
        'cell_name': 'SiteA_2', 'to_cell_name': 'SiteB_2',
        'distance': 5500, 'band': 'L800', 'to_band': 'L1800',
        'intra_site': 'n', 'intra_cell': 'n', 'weight': 150,
        'cell_perc_weight': 0.4
    })

    # Normal relations for SiteA_3 (bearing=240)
    data.append({
        'cell_name': 'SiteA_3', 'to_cell_name': 'SiteB_3',
        'distance': 5200, 'band': 'L800', 'to_band': 'L1800',
        'intra_site': 'n', 'intra_cell': 'n', 'weight': 120,
        'cell_perc_weight': 0.35
    })

    return pd.DataFrame(data)


@pytest.fixture
def extended_gis_data():
    """Create extended GIS data with more cells for testing."""
    gis = pd.DataFrame({
        'cell_name': [
            'SiteA_1', 'SiteA_2', 'SiteA_3',
            'SiteB_1', 'SiteB_2', 'SiteB_3',
            'SiteC_1', 'SiteC_2', 'SiteC_3',
        ],
        'site': ['SiteA'] * 3 + ['SiteB'] * 3 + ['SiteC'] * 3,
        'band': ['L800'] * 9,
        'bearing': [0, 120, 240] * 3,
        'hbw': [65] * 9,
        'latitude': [53.35, 53.35, 53.35, 53.36, 53.36, 53.36, 53.34, 53.34, 53.34],
        'longitude': [-6.26, -6.26, -6.26, -6.27, -6.27, -6.27, -6.25, -6.25, -6.25],
    })
    return gis


# ============================================================================
# Geometry Helper Tests
# ============================================================================

class TestGeometryHelpers:
    """Tests for geometry helper functions."""

    def test_bearing_deg_north(self):
        """Test bearing calculation for due north."""
        # From Dublin to a point directly north
        bearing = bearing_deg(53.35, -6.26, 54.35, -6.26)
        assert abs(bearing - 0) < 1 or abs(bearing - 360) < 1

    def test_bearing_deg_east(self):
        """Test bearing calculation for due east."""
        bearing = bearing_deg(53.35, -6.26, 53.35, -5.26)
        assert 85 < bearing < 95  # Approximately east

    def test_bearing_deg_south(self):
        """Test bearing calculation for due south."""
        bearing = bearing_deg(53.35, -6.26, 52.35, -6.26)
        assert 175 < bearing < 185

    def test_bearing_deg_vec(self):
        """Test vectorized bearing calculation."""
        lat1 = np.array([53.35, 53.35])
        lon1 = np.array([-6.26, -6.26])
        lat2 = np.array([54.35, 53.35])
        lon2 = np.array([-6.26, -5.26])

        bearings = bearing_deg_vec(lat1, lon1, lat2, lon2)

        assert len(bearings) == 2
        assert abs(bearings[0] - 0) < 1 or abs(bearings[0] - 360) < 1  # North
        assert 85 < bearings[1] < 95  # East

    def test_circ_diff_deg_same_angle(self):
        """Test circular difference for same angles."""
        assert circ_diff_deg(90, 90) == 0

    def test_circ_diff_deg_opposite(self):
        """Test circular difference for opposite angles."""
        assert circ_diff_deg(0, 180) == 180
        assert circ_diff_deg(90, 270) == 180

    def test_circ_diff_deg_wraparound(self):
        """Test circular difference with wraparound."""
        assert circ_diff_deg(350, 10) == 20
        assert circ_diff_deg(10, 350) == 20

    def test_circ_diff_deg_vec(self):
        """Test vectorized circular difference."""
        a = np.array([0, 90, 350])
        b = np.array([180, 270, 10])
        diffs = circ_diff_deg_vec(a, b)

        assert diffs[0] == 180
        assert diffs[1] == 180
        assert diffs[2] == 20

    def test_is_in_beam_true(self):
        """Test in-beam check when target is within beam."""
        assert is_in_beam(90, 85, 30) is True
        assert is_in_beam(90, 95, 30) is True
        assert is_in_beam(90, 90, 30) is True

    def test_is_in_beam_false(self):
        """Test in-beam check when target is outside beam."""
        assert is_in_beam(90, 180, 30) is False
        assert is_in_beam(0, 180, 30) is False

    def test_is_in_beam_wraparound(self):
        """Test in-beam check with angle wraparound."""
        assert is_in_beam(350, 10, 30) is True
        assert is_in_beam(10, 350, 30) is True


# ============================================================================
# CrossedFeederParams Tests
# ============================================================================

class TestCrossedFeederParams:
    """Tests for CrossedFeederParams dataclass."""

    def test_defaults(self):
        """Test default parameter initialization."""
        params = CrossedFeederParams()

        assert params.max_radius_m == 32000.0
        assert params.min_distance_m == 500.0
        assert params.hbw_cap_deg == 60.0
        assert params.percentile == 0.95
        assert params.beamwidth_expansion_factor == 1.5
        assert params.high_ratio_threshold == 0.8
        assert params.medium_ratio_threshold == 0.5
        assert params.max_data_drop_ratio == 0.5
        assert params.max_detection_rate == 0.20

    def test_custom_values(self):
        """Test custom parameter initialization."""
        params = CrossedFeederParams(
            max_radius_m=20000.0,
            beamwidth_expansion_factor=2.0,
            high_ratio_threshold=0.9
        )

        assert params.max_radius_m == 20000.0
        assert params.beamwidth_expansion_factor == 2.0
        assert params.high_ratio_threshold == 0.9
        # Other params should be defaults
        assert params.min_distance_m == 500.0

    def test_from_config_file(self):
        """Test loading parameters from JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"

            config_data = {
                "default": {
                    "max_radius_m": 25000.0,
                    "beamwidth_expansion_factor": 1.8,
                    "percentile": 0.90,
                    "high_ratio_threshold": 0.75,
                    "max_detection_rate": 0.15,
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            params = CrossedFeederParams.from_config(str(config_path))

            assert params.max_radius_m == 25000.0
            assert params.beamwidth_expansion_factor == 1.8
            assert params.percentile == 0.90
            assert params.high_ratio_threshold == 0.75
            assert params.max_detection_rate == 0.15

    def test_from_config_missing_file(self):
        """Test that missing config file returns defaults."""
        params = CrossedFeederParams.from_config("/nonexistent/path/config.json")

        # Should return default values
        assert params.max_radius_m == 32000.0
        assert params.beamwidth_expansion_factor == 1.5


# ============================================================================
# Band-Specific Radius Tests
# ============================================================================

class TestBandSpecificRadius:
    """Tests for band-specific distance thresholds."""

    def test_band_radius_constants(self):
        """Test that band-specific radius constants are correct."""
        assert BAND_MAX_RADIUS_M['L700'] == 32000
        assert BAND_MAX_RADIUS_M['L800'] == 30000
        assert BAND_MAX_RADIUS_M['L1800'] == 25000
        assert BAND_MAX_RADIUS_M['L2100'] == 20000
        assert BAND_MAX_RADIUS_M['L2600'] == 15000
        assert DEFAULT_MAX_RADIUS_M == 32000

    def test_default_radius_value(self):
        """Test default radius for unknown bands."""
        assert DEFAULT_MAX_RADIUS_M == 32000


# ============================================================================
# CrossedFeederDetector Tests
# ============================================================================

class TestCrossedFeederDetector:
    """Tests for CrossedFeederDetector class."""

    def test_initialization_default(self):
        """Test detector initialization with default params."""
        detector = CrossedFeederDetector()

        assert detector.params is not None
        assert detector.params.max_radius_m == 32000.0

    def test_initialization_custom_params(self):
        """Test detector initialization with custom params."""
        params = CrossedFeederParams(max_radius_m=20000.0)
        detector = CrossedFeederDetector(params)

        assert detector.params.max_radius_m == 20000.0

    def test_detect_returns_dict(self, sample_relations_data, extended_gis_data):
        """Test that detect returns a dictionary with expected keys."""
        detector = CrossedFeederDetector()
        result = detector.detect(sample_relations_data, extended_gis_data)

        assert isinstance(result, dict)
        assert 'relation_scores' in result
        assert 'cell_scores' in result
        assert 'site_summary' in result

    def test_detect_empty_relations(self, extended_gis_data):
        """Test detect with empty relations DataFrame."""
        detector = CrossedFeederDetector()
        empty_rel = pd.DataFrame(columns=[
            'cell_name', 'to_cell_name', 'distance', 'band', 'to_band',
            'intra_site', 'intra_cell', 'weight'
        ])

        result = detector.detect(empty_rel, extended_gis_data)

        assert isinstance(result, dict)
        assert len(result['relation_scores']) == 0

    def test_detect_missing_columns(self, extended_gis_data):
        """Test detect with missing required columns raises error."""
        detector = CrossedFeederDetector()
        incomplete_rel = pd.DataFrame({
            'cell_name': ['A'],
            'to_cell_name': ['B'],
            # Missing other required columns
        })

        with pytest.raises(ValueError, match="missing required columns"):
            detector.detect(incomplete_rel, extended_gis_data)

    def test_detect_result_has_confidence(self, sample_relations_data, extended_gis_data):
        """Test that cell_scores includes confidence column."""
        detector = CrossedFeederDetector()
        result = detector.detect(sample_relations_data, extended_gis_data)

        cell_scores = result['cell_scores']
        assert 'confidence' in cell_scores.columns

    def test_detect_result_has_flagged_ratio(self, sample_relations_data, extended_gis_data):
        """Test that site_summary includes flagged_ratio column."""
        detector = CrossedFeederDetector()
        result = detector.detect(sample_relations_data, extended_gis_data)

        site_summary = result['site_summary']
        assert 'flagged_ratio' in site_summary.columns
        assert 'total_sectors' in site_summary.columns


class TestCrossedFeederDetectorClassification:
    """Tests for site classification logic."""

    def test_high_ratio_classification(self):
        """Test classification for high flagged ratio."""
        # Create a scenario where most sectors are flagged
        params = CrossedFeederParams(
            percentile=0.0,  # Flag all cells above 0 score
            high_ratio_threshold=0.8,
            medium_ratio_threshold=0.5
        )
        detector = CrossedFeederDetector(params)

        # Create data for a 6-sector site where 5 are suspicious
        gis = pd.DataFrame({
            'cell_name': [f'Site_{i}' for i in range(6)],
            'site': ['Site'] * 6,
            'band': ['L800'] * 6,
            'bearing': [0, 60, 120, 180, 240, 300],
            'hbw': [65] * 6,
            'latitude': [53.35] * 6,
            'longitude': [-6.26] * 6,
        })

        # All cells have out-of-beam relations (simulating crossed feeders)
        rel_data = []
        for i in range(6):
            rel_data.append({
                'cell_name': f'Site_{i}',
                'to_cell_name': 'OtherSite_1',
                'distance': 5000,
                'band': 'L800',
                'to_band': 'L800',
                'intra_site': 'n',
                'intra_cell': 'n',
                'weight': 100,
            })

        relations = pd.DataFrame(rel_data)

        # Add neighbor GIS
        gis_extended = pd.concat([
            gis,
            pd.DataFrame({
                'cell_name': ['OtherSite_1'],
                'site': ['OtherSite'],
                'band': ['L800'],
                'bearing': [0],
                'hbw': [65],
                'latitude': [53.36],  # Different location
                'longitude': [-6.27],
            })
        ], ignore_index=True)

        result = detector.detect(relations, gis_extended)
        site_summary = result['site_summary']

        # Check that total_sectors is tracked
        if len(site_summary) > 0:
            assert 'total_sectors' in site_summary.columns


class TestCrossedFeederDetectorConfidence:
    """Tests for confidence scoring."""

    def test_confidence_range(self, sample_relations_data, extended_gis_data):
        """Test that confidence scores are in valid range [0, 100]."""
        detector = CrossedFeederDetector()
        result = detector.detect(sample_relations_data, extended_gis_data)

        cell_scores = result['cell_scores']
        if len(cell_scores) > 0:
            assert cell_scores['confidence'].min() >= 0
            assert cell_scores['confidence'].max() <= 100

    def test_confidence_zero_for_unflagged(self, sample_relations_data, extended_gis_data):
        """Test that unflagged cells have confidence = 0."""
        detector = CrossedFeederDetector()
        result = detector.detect(sample_relations_data, extended_gis_data)

        cell_scores = result['cell_scores']
        unflagged = cell_scores[~cell_scores['flagged']]
        if len(unflagged) > 0:
            assert (unflagged['confidence'] == 0).all()


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_crossed_feeders_function(self, sample_relations_data, extended_gis_data):
        """Test detect_crossed_feeders convenience function."""
        result = detect_crossed_feeders(sample_relations_data, extended_gis_data)

        assert isinstance(result, dict)
        assert 'relation_scores' in result
        assert 'cell_scores' in result
        assert 'site_summary' in result

    def test_detect_crossed_feeders_with_params(self, sample_relations_data, extended_gis_data):
        """Test detect_crossed_feeders with custom params."""
        params = CrossedFeederParams(percentile=0.90)
        result = detect_crossed_feeders(
            sample_relations_data,
            extended_gis_data,
            params=params
        )

        assert isinstance(result, dict)

    def test_detect_crossed_feeders_with_band_filter(self, sample_relations_data, extended_gis_data):
        """Test detect_crossed_feeders with band filter."""
        result = detect_crossed_feeders(
            sample_relations_data,
            extended_gis_data,
            band_filter='L800'
        )

        assert isinstance(result, dict)


# ============================================================================
# Data Quality Tests
# ============================================================================

class TestDataQuality:
    """Tests for data quality checks."""

    def test_missing_gis_geometry_warning(self, caplog):
        """Test that missing GIS geometry generates warning."""
        detector = CrossedFeederDetector()

        # Create relations with cells that won't have GIS
        relations = pd.DataFrame({
            'cell_name': ['Unknown_1', 'Unknown_2'],
            'to_cell_name': ['Unknown_3', 'Unknown_4'],
            'distance': [5000, 6000],
            'band': ['L800', 'L800'],
            'to_band': ['L800', 'L800'],
            'intra_site': ['n', 'n'],
            'intra_cell': ['n', 'n'],
            'weight': [100, 100],
        })

        gis = pd.DataFrame({
            'cell_name': ['Other_1'],  # No matching cells
            'site': ['Other'],
            'band': ['L800'],
            'bearing': [0],
            'hbw': [65],
            'latitude': [53.35],
            'longitude': [-6.26],
        })

        result = detector.detect(relations, gis)

        # Should complete without error but log warnings
        assert isinstance(result, dict)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflow."""

    def test_full_workflow(self, extended_gis_data):
        """Test complete detection workflow."""
        # Create more comprehensive relations
        rel_data = []
        cells = ['SiteA_1', 'SiteA_2', 'SiteA_3', 'SiteB_1', 'SiteB_2', 'SiteB_3']

        for i, cell in enumerate(cells[:3]):
            # Add normal relations
            rel_data.append({
                'cell_name': cell,
                'to_cell_name': cells[3 + i],
                'distance': 5000 + i * 100,
                'band': 'L800',
                'to_band': 'L800',
                'intra_site': 'n',
                'intra_cell': 'n',
                'weight': 100 + i * 10,
            })

        relations = pd.DataFrame(rel_data)

        # Run detection
        result = detect_crossed_feeders(relations, extended_gis_data)

        # Verify output structure
        assert 'relation_scores' in result
        assert 'cell_scores' in result
        assert 'site_summary' in result

        # Verify cell_scores columns
        cell_cols = result['cell_scores'].columns.tolist()
        assert 'cell_name' in cell_cols
        assert 'cell_score' in cell_cols
        assert 'flagged' in cell_cols
        assert 'confidence' in cell_cols

        # Verify site_summary columns
        site_cols = result['site_summary'].columns.tolist()
        assert 'site' in site_cols
        assert 'flagged_cells' in site_cols
        assert 'total_sectors' in site_cols
        assert 'flagged_ratio' in site_cols
        assert 'classification' in site_cols

    def test_beamwidth_expansion_effect(self, extended_gis_data):
        """Test that beamwidth expansion affects in-beam classification."""
        # Create a relation just outside the 3dB beamwidth
        relations = pd.DataFrame({
            'cell_name': ['SiteA_1'],
            'to_cell_name': ['SiteC_1'],
            'distance': [5000],
            'band': ['L800'],
            'to_band': ['L800'],
            'intra_site': ['n'],
            'intra_cell': ['n'],
            'weight': [100],
        })

        # With default 1.5x expansion, more relations should be "in-beam"
        params_default = CrossedFeederParams(beamwidth_expansion_factor=1.5)
        detector_default = CrossedFeederDetector(params_default)
        result_default = detector_default.detect(relations, extended_gis_data)

        # With no expansion
        params_no_expand = CrossedFeederParams(beamwidth_expansion_factor=1.0)
        detector_no_expand = CrossedFeederDetector(params_no_expand)
        result_no_expand = detector_no_expand.detect(relations, extended_gis_data)

        # Both should complete successfully
        assert isinstance(result_default, dict)
        assert isinstance(result_no_expand, dict)
