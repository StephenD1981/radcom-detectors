"""
Unit tests for crossed feeder detection with swap pattern analysis.
"""

import numpy as np
import pandas as pd
import pytest

from ran_optimizer.recommendations.crossed_feeder import (
    CrossedFeederDetector,
    CrossedFeederParams,
    bearing_deg,
    bearing_deg_vec,
    circ_diff_deg,
    circ_diff_deg_vec,
    is_in_beam,
    weighted_circular_mean,
    detect_crossed_feeders,
    BAND_MAX_RADIUS_M,
    DEFAULT_MAX_RADIUS_M,
    _get_tech_from_band,
)


class TestGeometryHelpers:
    """Test geometry helper functions."""

    def test_bearing_deg_north(self):
        """Bearing due north should be 0 degrees."""
        result = bearing_deg(0, 0, 1, 0)
        assert abs(result - 0) < 0.1

    def test_bearing_deg_east(self):
        """Bearing due east should be ~90 degrees."""
        result = bearing_deg(0, 0, 0, 1)
        assert abs(result - 90) < 0.1

    def test_bearing_deg_south(self):
        """Bearing due south should be 180 degrees."""
        result = bearing_deg(1, 0, 0, 0)
        assert abs(result - 180) < 0.1

    def test_bearing_deg_vec(self):
        """Vectorized bearing calculation."""
        lat1 = np.array([0, 0])
        lon1 = np.array([0, 0])
        lat2 = np.array([1, 0])
        lon2 = np.array([0, 1])
        result = bearing_deg_vec(lat1, lon1, lat2, lon2)
        assert len(result) == 2
        assert abs(result[0] - 0) < 0.1  # North
        assert abs(result[1] - 90) < 0.1  # East

    def test_circ_diff_deg_same_angle(self):
        """Same angles should have zero difference."""
        assert circ_diff_deg(90, 90) == 0

    def test_circ_diff_deg_opposite(self):
        """Opposite angles should have 180 degree difference."""
        assert circ_diff_deg(0, 180) == 180

    def test_circ_diff_deg_wraparound(self):
        """Should handle wraparound correctly."""
        assert abs(circ_diff_deg(10, 350) - 20) < 0.1

    def test_circ_diff_deg_vec(self):
        """Vectorized circular difference."""
        a = np.array([90, 0, 10])
        b = np.array([90, 180, 350])
        result = circ_diff_deg_vec(a, b)
        assert len(result) == 3
        assert result[0] == 0
        assert result[1] == 180
        assert abs(result[2] - 20) < 0.1

    def test_is_in_beam_true(self):
        """Target within beam should return True."""
        assert is_in_beam(90, 100, 30) is True  # 100 is within 90 +/- 30

    def test_is_in_beam_false(self):
        """Target outside beam should return False."""
        assert is_in_beam(90, 180, 30) is False  # 180 is outside 90 +/- 30

    def test_is_in_beam_wraparound(self):
        """Should handle wraparound correctly."""
        assert is_in_beam(350, 10, 30) is True  # 10 is within 350 +/- 30 (wraps around)


class TestWeightedCircularMean:
    """Test weighted circular mean calculation."""

    def test_single_angle(self):
        """Single angle should return that angle."""
        result = weighted_circular_mean(np.array([90.0]), np.array([1.0]))
        assert abs(result - 90) < 0.1

    def test_equal_weights(self):
        """Equal weights should give simple mean for non-wrapping angles."""
        result = weighted_circular_mean(np.array([80.0, 100.0]), np.array([1.0, 1.0]))
        assert abs(result - 90) < 0.1

    def test_unequal_weights(self):
        """Unequal weights should bias toward heavier weight."""
        result = weighted_circular_mean(np.array([0.0, 90.0]), np.array([3.0, 1.0]))
        assert result < 45  # Should be closer to 0

    def test_wraparound(self):
        """Should handle wraparound correctly."""
        result = weighted_circular_mean(np.array([350.0, 10.0]), np.array([1.0, 1.0]))
        assert abs(result - 0) < 1 or abs(result - 360) < 1

    def test_empty_array(self):
        """Empty array should return NaN."""
        result = weighted_circular_mean(np.array([]), np.array([]))
        assert np.isnan(result)

    def test_zero_weights(self):
        """Zero weights should return NaN."""
        result = weighted_circular_mean(np.array([90.0]), np.array([0.0]))
        assert np.isnan(result)


class TestTechFromBand:
    """Test technology extraction from band string."""

    def test_lte_bands(self):
        assert _get_tech_from_band("L700") == "LTE"
        assert _get_tech_from_band("L800") == "LTE"
        assert _get_tech_from_band("L1800") == "LTE"

    def test_nr_bands(self):
        assert _get_tech_from_band("N78") == "NR"
        assert _get_tech_from_band("N258") == "NR"

    def test_umts_bands(self):
        assert _get_tech_from_band("U900") == "UMTS"
        assert _get_tech_from_band("U2100") == "UMTS"

    def test_gsm_bands(self):
        assert _get_tech_from_band("G900") == "GSM"
        assert _get_tech_from_band("G1800") == "GSM"

    def test_unknown(self):
        assert _get_tech_from_band("X123") == "UNKNOWN"
        assert _get_tech_from_band("") == "UNKNOWN"
        assert _get_tech_from_band(None) == "UNKNOWN"


class TestCrossedFeederParams:
    """Test parameter handling."""

    def test_defaults(self):
        """Default parameters should be set correctly."""
        params = CrossedFeederParams()
        assert params.swap_angle_tolerance_deg == 30.0
        assert params.min_out_of_beam_ratio == 0.5  # 50% threshold
        assert params.min_out_of_beam_weight == 5.0
        assert params.beamwidth_expansion_factor == 1.5

    def test_custom_values(self):
        """Custom parameters should be accepted."""
        params = CrossedFeederParams(
            swap_angle_tolerance_deg=20.0,
            min_out_of_beam_ratio=0.4,
        )
        assert params.swap_angle_tolerance_deg == 20.0
        assert params.min_out_of_beam_ratio == 0.4

    def test_from_config_missing_file(self):
        """Missing config file should return defaults."""
        params = CrossedFeederParams.from_config("/nonexistent/path.json")
        assert params.swap_angle_tolerance_deg == 30.0


class TestBandSpecificRadius:
    """Test band-specific radius constants."""

    def test_band_radius_constants(self):
        """Band radius constants should be defined correctly."""
        assert BAND_MAX_RADIUS_M['L700'] == 32000
        assert BAND_MAX_RADIUS_M['L800'] == 30000
        assert BAND_MAX_RADIUS_M['L1800'] == 25000
        assert BAND_MAX_RADIUS_M['L2100'] == 20000
        assert BAND_MAX_RADIUS_M['L2600'] == 15000

    def test_default_radius(self):
        """Default radius should be set."""
        assert DEFAULT_MAX_RADIUS_M == 32000


def _create_test_gis(cells_config):
    """Create test GIS DataFrame.

    cells_config: list of dicts with keys:
        cell_name, site, band, bearing, hbw, latitude, longitude
    """
    return pd.DataFrame(cells_config)


def _create_test_relations(relations_config):
    """Create test relations DataFrame.

    relations_config: list of dicts with keys:
        cell_name, to_cell_name, distance, band, to_band, intra_site, intra_cell, weight
    """
    return pd.DataFrame(relations_config)


class TestCrossedFeederDetector:
    """Test the CrossedFeederDetector class."""

    def test_initialization(self):
        """Detector should initialize with default params."""
        detector = CrossedFeederDetector()
        assert detector.params is not None
        assert detector.params.swap_angle_tolerance_deg == 30.0

    def test_initialization_custom_params(self):
        """Detector should accept custom params."""
        params = CrossedFeederParams(swap_angle_tolerance_deg=20.0)
        detector = CrossedFeederDetector(params)
        assert detector.params.swap_angle_tolerance_deg == 20.0

    def test_detect_returns_dict(self):
        """Detection should return dictionary with expected keys."""
        gis_df = _create_test_gis([
            {"cell_name": "CELL1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "CELL2", "site": "SITE1", "band": "L800", "bearing": 120, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        relations_df = _create_test_relations([
            {"cell_name": "CELL1", "to_cell_name": "EXT1", "distance": 5000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 10},
        ])

        # Add external cell to GIS
        gis_df = pd.concat([gis_df, pd.DataFrame([
            {"cell_name": "EXT1", "site": "EXT", "band": "L800", "bearing": 180, "hbw": 65, "latitude": 53.05, "longitude": -6.0},
        ])], ignore_index=True)

        detector = CrossedFeederDetector()
        result = detector.detect(relations_df, gis_df)

        assert isinstance(result, dict)
        assert 'cells' in result
        assert 'sites' in result
        assert 'swap_pairs' in result
        assert 'relation_details' in result

    def test_detect_empty_relations(self):
        """Empty relations should return empty results."""
        gis_df = _create_test_gis([
            {"cell_name": "CELL1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        # Create empty DataFrame with required columns
        relations_df = pd.DataFrame(columns=["cell_name", "to_cell_name", "distance", "band", "to_band", "intra_site", "intra_cell", "weight"])

        detector = CrossedFeederDetector()
        result = detector.detect(relations_df, gis_df)

        assert len(result['cells']) == 0
        assert len(result['swap_pairs']) == 0

    def test_detect_missing_columns(self):
        """Missing required columns should raise ValueError."""
        gis_df = pd.DataFrame({"cell_name": ["CELL1"]})  # Missing columns
        relations_df = _create_test_relations([
            {"cell_name": "CELL1", "to_cell_name": "EXT1", "distance": 5000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 10},
        ])

        detector = CrossedFeederDetector()
        with pytest.raises(ValueError):
            detector.detect(relations_df, gis_df)


class TestSwapPatternDetection:
    """Test swap pattern detection logic."""

    @pytest.fixture
    def swap_scenario_gis(self):
        """Create GIS data for a swap scenario.

        Site has 3 cells pointing at 0°, 120°, 240°.
        """
        return _create_test_gis([
            # Site cells
            {"cell_name": "SITE1_A", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "SITE1_B", "site": "SITE1", "band": "L800", "bearing": 120, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "SITE1_C", "site": "SITE1", "band": "L800", "bearing": 240, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            # External cells in different directions
            {"cell_name": "NORTH", "site": "NORTH_SITE", "band": "L800", "bearing": 180, "hbw": 65, "latitude": 53.1, "longitude": -6.0},
            {"cell_name": "SOUTHEAST", "site": "SE_SITE", "band": "L800", "bearing": 300, "hbw": 65, "latitude": 52.95, "longitude": -5.9},
            {"cell_name": "SOUTHWEST", "site": "SW_SITE", "band": "L800", "bearing": 60, "hbw": 65, "latitude": 52.95, "longitude": -6.1},
        ])

    def test_no_swap_normal_traffic(self, swap_scenario_gis):
        """Normal traffic (in-beam) should not flag swap."""
        # Cell A (0°) has traffic to NORTH (0° direction) - in beam
        relations_df = _create_test_relations([
            {"cell_name": "SITE1_A", "to_cell_name": "NORTH", "distance": 10000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
        ])

        detector = CrossedFeederDetector()
        result = detector.detect(relations_df, swap_scenario_gis)

        # Should not detect any high confidence issues
        high_conf = result['cells'][result['cells']['confidence_level'] == 'HIGH']
        assert len(high_conf) == 0
        assert len(result['swap_pairs']) == 0

    def test_swap_detected_reciprocal_pattern(self, swap_scenario_gis):
        """Reciprocal swap pattern should be detected as HIGH confidence.

        Cell A (0°) has traffic toward 120° (Cell B's direction)
        Cell B (120°) has traffic toward 0° (Cell A's direction)
        """
        relations_df = _create_test_relations([
            # Cell A (0° azimuth) has strong traffic to SOUTHEAST (~120° direction)
            {"cell_name": "SITE1_A", "to_cell_name": "SOUTHEAST", "distance": 8000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
            # Cell B (120° azimuth) has strong traffic to NORTH (~0° direction)
            {"cell_name": "SITE1_B", "to_cell_name": "NORTH", "distance": 10000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
        ])

        params = CrossedFeederParams(
            min_out_of_beam_ratio=0.3,
            min_out_of_beam_weight=5.0,
            swap_angle_tolerance_deg=30.0,
            min_total_relations=1,  # Lower for unit testing
            min_out_of_beam_relations=1,  # Lower for unit testing
        )
        detector = CrossedFeederDetector(params)
        result = detector.detect(relations_df, swap_scenario_gis)

        # Should detect swap pair
        assert len(result['swap_pairs']) == 1
        swap = result['swap_pairs'].iloc[0]
        assert set([swap['cell_a'], swap['cell_b']]) == {'SITE1_A', 'SITE1_B'}

        # Both cells should be HIGH confidence
        high_conf = result['cells'][result['cells']['confidence_level'] == 'HIGH']
        assert len(high_conf) == 2
        assert set(high_conf['cell_name']) == {'SITE1_A', 'SITE1_B'}

    def test_single_anomaly_low_confidence(self, swap_scenario_gis):
        """Single cell with out-of-beam traffic should be LOW confidence."""
        # Only Cell A has anomalous traffic
        relations_df = _create_test_relations([
            {"cell_name": "SITE1_A", "to_cell_name": "SOUTHEAST", "distance": 8000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
        ])

        params = CrossedFeederParams(
            min_out_of_beam_ratio=0.3,
            min_out_of_beam_weight=5.0,
            min_total_relations=1,  # Lower for unit testing
            min_out_of_beam_relations=1,  # Lower for unit testing
        )
        detector = CrossedFeederDetector(params)
        result = detector.detect(relations_df, swap_scenario_gis)

        # Should be LOW confidence (single anomaly, no swap partner)
        cell_a = result['cells'][result['cells']['cell_name'] == 'SITE1_A']
        assert len(cell_a) == 1
        assert cell_a.iloc[0]['confidence_level'] == 'LOW'
        assert len(result['swap_pairs']) == 0

    def test_multiple_anomalies_no_swap_medium_confidence(self, swap_scenario_gis):
        """Multiple anomalies without clean swap should be MEDIUM confidence."""
        # Cell A and B both have anomalous traffic but NOT toward each other's azimuths
        relations_df = _create_test_relations([
            # Cell A (0°) has traffic toward 240° (Cell C's direction, not B)
            {"cell_name": "SITE1_A", "to_cell_name": "SOUTHWEST", "distance": 8000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
            # Cell B (120°) has traffic toward 240° (Cell C's direction, not A)
            {"cell_name": "SITE1_B", "to_cell_name": "SOUTHWEST", "distance": 8000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
        ])

        params = CrossedFeederParams(
            min_out_of_beam_ratio=0.3,
            min_out_of_beam_weight=5.0,
            swap_angle_tolerance_deg=30.0,
            min_total_relations=1,  # Lower for unit testing
            min_out_of_beam_relations=1,  # Lower for unit testing
        )
        detector = CrossedFeederDetector(params)
        result = detector.detect(relations_df, swap_scenario_gis)

        # Both should be MEDIUM (multiple anomalies, but not a swap pattern)
        medium_conf = result['cells'][result['cells']['confidence_level'] == 'MEDIUM']
        assert len(medium_conf) == 2
        assert len(result['swap_pairs']) == 0


class TestSiteSummary:
    """Test site-level summary generation."""

    def test_site_severity_high(self):
        """Site with swap pair should have HIGH severity."""
        gis_df = _create_test_gis([
            {"cell_name": "A1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "B1", "site": "SITE1", "band": "L800", "bearing": 120, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "EXT1", "site": "EXT", "band": "L800", "bearing": 180, "hbw": 65, "latitude": 53.1, "longitude": -6.0},
            {"cell_name": "EXT2", "site": "EXT", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 52.9, "longitude": -5.9},
        ])

        relations_df = _create_test_relations([
            # Swap pattern
            {"cell_name": "A1", "to_cell_name": "EXT2", "distance": 8000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
            {"cell_name": "B1", "to_cell_name": "EXT1", "distance": 10000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 50},
        ])

        params = CrossedFeederParams(
            min_total_relations=1,  # Lower for unit testing
            min_out_of_beam_relations=1,  # Lower for unit testing
        )
        detector = CrossedFeederDetector(params)
        result = detector.detect(relations_df, gis_df)

        site = result['sites'][result['sites']['site'] == 'SITE1']
        assert len(site) == 1
        assert site.iloc[0]['severity'] == 'HIGH'
        assert 'CROSSED FEEDER' in site.iloc[0]['classification']


class TestConvenienceFunction:
    """Test the detect_crossed_feeders convenience function."""

    def test_convenience_function(self):
        """Convenience function should work like detector."""
        gis_df = _create_test_gis([
            {"cell_name": "CELL1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "EXT1", "site": "EXT", "band": "L800", "bearing": 180, "hbw": 65, "latitude": 53.1, "longitude": -6.0},
        ])
        relations_df = _create_test_relations([
            {"cell_name": "CELL1", "to_cell_name": "EXT1", "distance": 10000, "band": "L800", "to_band": "L800", "intra_site": "n", "intra_cell": "n", "weight": 10},
        ])

        result = detect_crossed_feeders(relations_df, gis_df)

        assert isinstance(result, dict)
        assert 'cells' in result
        assert 'swap_pairs' in result

    def test_convenience_function_with_params(self):
        """Convenience function should accept custom params."""
        gis_df = _create_test_gis([
            {"cell_name": "CELL1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        # Create empty DataFrame with required columns
        relations_df = pd.DataFrame(columns=["cell_name", "to_cell_name", "distance", "band", "to_band", "intra_site", "intra_cell", "weight"])

        params = CrossedFeederParams(swap_angle_tolerance_deg=20.0)
        result = detect_crossed_feeders(relations_df, gis_df, params=params)

        assert isinstance(result, dict)

    def test_convenience_function_with_band_filter(self):
        """Convenience function should accept band filter."""
        gis_df = _create_test_gis([
            {"cell_name": "CELL1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "CELL2", "site": "SITE1", "band": "L1800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        # Create empty DataFrame with required columns
        relations_df = pd.DataFrame(columns=["cell_name", "to_cell_name", "distance", "band", "to_band", "intra_site", "intra_cell", "weight"])

        result = detect_crossed_feeders(relations_df, gis_df, band_filter="L800")

        assert isinstance(result, dict)


class TestDataQuality:
    """Test data quality handling."""

    def test_cross_tech_filtered(self):
        """Cross-technology relations should be filtered out."""
        gis_df = _create_test_gis([
            {"cell_name": "LTE1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "UMTS1", "site": "SITE1", "band": "U900", "bearing": 120, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        relations_df = _create_test_relations([
            # Cross-tech relation (LTE to UMTS) - should be filtered
            {"cell_name": "LTE1", "to_cell_name": "UMTS1", "distance": 1000, "band": "L800", "to_band": "U900", "intra_site": "y", "intra_cell": "n", "weight": 50},
        ])

        detector = CrossedFeederDetector()
        result = detector.detect(relations_df, gis_df)

        # Should have empty results (cross-tech filtered, then intra-site filtered)
        assert len(result['cells']) == 0

    def test_intra_site_filtered(self):
        """Intra-site relations should be filtered out."""
        gis_df = _create_test_gis([
            {"cell_name": "A1", "site": "SITE1", "band": "L800", "bearing": 0, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
            {"cell_name": "B1", "site": "SITE1", "band": "L800", "bearing": 120, "hbw": 65, "latitude": 53.0, "longitude": -6.0},
        ])
        relations_df = _create_test_relations([
            # Intra-site relation - should be filtered
            {"cell_name": "A1", "to_cell_name": "B1", "distance": 0, "band": "L800", "to_band": "L800", "intra_site": "y", "intra_cell": "n", "weight": 50},
        ])

        detector = CrossedFeederDetector()
        result = detector.detect(relations_df, gis_df)

        # Should have empty relation_details (intra-site filtered)
        assert len(result['relation_details']) == 0
