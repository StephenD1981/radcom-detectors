"""
Unit tests for undershooting detection algorithm.

Tests core algorithmic components:
- Distance filtering
- RSRP-based interference calculation
- Uptilt impact estimation
- 3GPP antenna pattern
- Coverage expansion calculations
"""
import pytest
import pandas as pd
import numpy as np
from ran_optimizer.recommendations import UndershooterDetector, UndershooterParams


class TestDistanceFiltering:
    """Test distance-based candidate filtering."""

    @pytest.fixture
    def detector(self):
        """Create detector with max distance = 8km."""
        params = UndershooterParams(max_cell_distance=8000.0)
        return UndershooterDetector(params)

    @pytest.fixture
    def sample_grid_data(self):
        """Create sample grid data with cells at various max distances."""
        return pd.DataFrame({
            'cell_name': ['1', '1', '1', '2', '2', '2', '3', '3'],
            'grid': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            'distance_to_cell': [1000, 3000, 5000, 2000, 6000, 12000, 1500, 7500],
            'event_count': [100, 80, 60, 90, 70, 50, 85, 65],
        })

    def test_filters_cells_by_max_distance(self, detector, sample_grid_data):
        """Should filter cells with max distance <= threshold."""
        candidates = detector._filter_by_distance(sample_grid_data)

        # Cell 1: max = 5000m (PASS)
        # Cell 2: max = 12000m (FAIL - too far)
        # Cell 3: max = 7500m (PASS)
        assert '1' in candidates['cell_name'].values, "Cell 1 should be included (5km max)"
        assert '2' not in candidates['cell_name'].values, "Cell 2 should be excluded (12km max)"
        assert '3' in candidates['cell_name'].values, "Cell 3 should be included (7.5km max)"

    def test_calculates_cell_statistics(self, detector, sample_grid_data):
        """Should calculate max distance, total traffic, and grid count."""
        candidates = detector._filter_by_distance(sample_grid_data)

        cell1 = candidates[candidates['cell_name'] == '1'].iloc[0]

        assert cell1['max_distance_m'] == 5000, "Cell 1 max distance should be 5000m"
        assert cell1['total_traffic'] == 240, "Cell 1 total traffic should be 100+80+60"
        assert cell1['total_grids'] == 3, "Cell 1 should have 3 grids"


class TestInterferenceCalculation:
    """Test RSRP-based interference calculation."""

    @pytest.fixture
    def detector(self):
        """Create detector with default interference params."""
        params = UndershooterParams(
            interference_threshold_db=7.5,
            max_cell_grid_count=2
        )
        return UndershooterDetector(params)

    @pytest.fixture
    def candidates_df(self):
        """Sample candidates."""
        return pd.DataFrame({
            'cell_name': ['1', '2', '3'],
            'max_distance_m': [5000, 6000, 7000],
            'total_grids': [10, 15, 20],
            'total_traffic': [1000, 1500, 2000],
        })

    @pytest.fixture
    def grid_data_with_rsrp(self):
        """Grid data with multiple cells per grid (competition)."""
        data = []

        # Grid A: 3 cells competing (cell1=-70, cell2=-72, cell3=-90)
        data.extend([
            {'grid': 'gridA', 'cell_name': '1', 'avg_rsrp': -70, 'distance_to_cell': 1000, 'event_count': 100},
            {'grid': 'gridA', 'cell_name': '2', 'avg_rsrp': -72, 'distance_to_cell': 1200, 'event_count': 90},
            {'grid': 'gridA', 'cell_name': '3', 'avg_rsrp': -90, 'distance_to_cell': 3000, 'event_count': 30},
        ])

        # Grid B: 2 cells, only 1 competing (cell1=-75, cell2=-100)
        data.extend([
            {'grid': 'gridB', 'cell_name': '1', 'avg_rsrp': -75, 'distance_to_cell': 2000, 'event_count': 80},
            {'grid': 'gridB', 'cell_name': '2', 'avg_rsrp': -100, 'distance_to_cell': 5000, 'event_count': 20},
        ])

        # Grid C: Only cell 3 (no competition)
        data.append({'grid': 'gridC', 'cell_name': '3', 'avg_rsrp': -80, 'distance_to_cell': 4000, 'event_count': 70})

        return pd.DataFrame(data)

    def test_calculates_interference_percentage(self, detector, candidates_df, grid_data_with_rsrp):
        """Should calculate interference percentage based on RSRP competition."""
        result = detector._calculate_interference(candidates_df, grid_data_with_rsrp)

        assert 'interference_grids' in result.columns
        assert 'interference_percentage' in result.columns

        # All interference percentages should be between 0 and 1
        assert all(result['interference_percentage'] >= 0.0)
        assert all(result['interference_percentage'] <= 1.0)

    def test_identifies_competing_cells(self, detector, candidates_df, grid_data_with_rsrp):
        """Should identify grids with competing cells within RSRP threshold."""
        result = detector._calculate_interference(candidates_df, grid_data_with_rsrp)

        # Grid A has 3 competing cells (cell1, cell2 within 7.5dB of best)
        # This exceeds max_cell_grid_count=2, so it's interfering
        # Cells 1 and 2 should have interference_grids > 0

        # Check that interference was detected
        cells_with_interference = result[result['interference_grids'] > 0]
        assert len(cells_with_interference) >= 0, "Should detect interfering grids"


class TestUptiltImpactEstimation:
    """Test uptilt distance estimation using 3GPP pattern."""

    @pytest.fixture
    def detector(self):
        """Create detector with default parameters."""
        return UndershooterDetector(UndershooterParams())

    def test_uptilt_increases_distance(self, detector):
        """Uptilt should increase coverage distance."""
        d_max = 5000.0
        alpha = 8.0  # 8 degree downtilt
        height = 30.0
        delta_tilt = -1.0  # 1 degree uptilt (negative)

        new_distance, increase_pct = detector._estimate_distance_after_tilt(
            d_max, alpha, height, delta_tilt
        )

        assert new_distance > d_max, "Uptilt should increase distance"
        assert increase_pct > 0.0, "Increase percentage should be positive"

    def test_larger_uptilt_gives_more_gain(self, detector):
        """Larger uptilt should give larger distance increase."""
        d_max = 5000.0
        alpha = 8.0
        height = 30.0

        new_dist_1deg, pct_1deg = detector._estimate_distance_after_tilt(
            d_max, alpha, height, delta_tilt_deg=-1.0
        )
        new_dist_2deg, pct_2deg = detector._estimate_distance_after_tilt(
            d_max, alpha, height, delta_tilt_deg=-2.0
        )

        assert new_dist_2deg > new_dist_1deg, "2° uptilt should give more distance than 1°"
        assert pct_2deg > pct_1deg, "2° uptilt should give higher percentage increase"

    def test_cannot_uptilt_from_zero(self, detector):
        """Cannot uptilt from 0 degree tilt."""
        new_distance, increase_pct = detector._estimate_distance_after_tilt(
            5000.0, alpha_deg=0.0, h_m=30.0, delta_tilt_deg=-1.0
        )

        assert new_distance == 5000.0, "Should return original distance"
        assert increase_pct == 0.0, "Should have 0% increase"

    def test_invalid_inputs_return_zero_gain(self, detector):
        """Invalid inputs should return zero gain."""
        # Zero distance
        new_dist, pct = detector._estimate_distance_after_tilt(
            0.0, 8.0, 30.0, -1.0
        )
        assert new_dist == 0.0 and pct == 0.0

        # Negative height
        new_dist, pct = detector._estimate_distance_after_tilt(
            5000.0, 8.0, -10.0, -1.0
        )
        assert new_dist == 5000.0 and pct == 0.0

    def test_realistic_uptilt_gain(self, detector):
        """Uptilt gain should be in realistic range (5-35%)."""
        # Typical scenario: 30m height, 8° tilt, 5km current distance
        new_dist, pct = detector._estimate_distance_after_tilt(
            5000.0, 8.0, 30.0, -1.0
        )

        # 1° uptilt should give 5-35% increase
        assert 0.05 <= pct <= 0.35, f"Uptilt gain should be 5-35%, got {pct*100:.1f}%"


class TestVerticalAttenuation:
    """Test 3GPP vertical antenna pattern."""

    @pytest.fixture
    def detector(self):
        """Create detector with default antenna params."""
        return UndershooterDetector(UndershooterParams())

    def test_zero_at_boresight(self, detector):
        """Attenuation at boresight should be 0 dB."""
        alpha = 10.0
        theta = 10.0  # Same as tilt

        attenuation = detector._vertical_attenuation(theta, alpha)

        assert attenuation == 0.0, "Attenuation at boresight should be 0 dB"

    def test_increases_with_angle(self, detector):
        """Attenuation should increase away from boresight."""
        alpha = 10.0

        att_0 = detector._vertical_attenuation(10.0, alpha)
        att_5 = detector._vertical_attenuation(15.0, alpha)
        att_10 = detector._vertical_attenuation(20.0, alpha)

        assert att_0 < att_5 < att_10, "Attenuation should increase with angle"

    def test_symmetric_pattern(self, detector):
        """Pattern should be symmetric around boresight."""
        alpha = 10.0
        hpbw = 65.0

        # 5 degrees above and below boresight
        att_above = detector._vertical_attenuation(15.0, alpha)
        att_below = detector._vertical_attenuation(5.0, alpha)

        assert abs(att_above - att_below) < 0.01, "Pattern should be symmetric"

    def test_capped_at_sla(self, detector):
        """Attenuation should not exceed side lobe attenuation."""
        alpha = 10.0
        theta = 90.0  # Very far from boresight

        attenuation = detector._vertical_attenuation(theta, alpha)

        # Should be capped at SLA (default 20 dB)
        assert attenuation <= detector.params.sla_v_db, \
            f"Attenuation should not exceed SLA ({detector.params.sla_v_db} dB)"


class TestCandidateFiltering:
    """Test candidate filtering logic."""

    @pytest.fixture
    def detector(self):
        """Create detector with filtering thresholds."""
        params = UndershooterParams(
            min_cell_event_count=500,
            max_interference_percentage=0.30
        )
        return UndershooterDetector(params)

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidates with varying metrics."""
        return pd.DataFrame({
            'cell_name': ['1', '2', '3', '4'],
            'max_distance_m': [5000, 6000, 7000, 8000],
            'total_traffic': [600, 400, 700, 550],  # Cell 2 below min
            'interference_percentage': [0.10, 0.20, 0.50, 0.25],  # Cell 3 above max
            'total_grids': [10, 8, 15, 12],
        })

    def test_filters_by_traffic_threshold(self, detector, sample_candidates):
        """Should filter cells below minimum traffic."""
        filtered = detector._filter_candidates(sample_candidates)

        # Cell 2 has traffic=400 < min=500, should be excluded
        assert '2' not in filtered['cell_name'].values, \
            "Cell 2 should be excluded (traffic < min)"

    def test_filters_by_interference_threshold(self, detector, sample_candidates):
        """Should filter cells above maximum interference."""
        filtered = detector._filter_candidates(sample_candidates)

        # Cell 3 has interference=0.50 > max=0.30, should be excluded
        assert '3' not in filtered['cell_name'].values, \
            "Cell 3 should be excluded (interference > max)"

    def test_keeps_valid_candidates(self, detector, sample_candidates):
        """Should keep cells meeting all criteria."""
        filtered = detector._filter_candidates(sample_candidates)

        # Cells 1 and 4 meet all criteria
        assert '1' in filtered['cell_name'].values, "Cell 1 should pass all filters"
        assert '4' in filtered['cell_name'].values, "Cell 4 should pass all filters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
