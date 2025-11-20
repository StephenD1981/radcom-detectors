"""
Tests for geometry functions.
"""
import pytest
import math
from ran_optimizer.core.geometry import (
    haversine_distance,
    calculate_bearing,
    bearing_difference,
    get_distance_and_bearing,
    is_within_sector,
    EARTH_RADIUS_M,
)


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_point(self):
        """Distance from a point to itself should be zero."""
        distance = haversine_distance(53.3498, -6.2603, 53.3498, -6.2603)
        assert distance == pytest.approx(0.0, abs=0.1)

    def test_dublin_to_cork(self):
        """Test known distance: Dublin to Cork, Ireland (~219 km)."""
        # Dublin coordinates
        dublin_lat, dublin_lon = 53.3498, -6.2603
        # Cork coordinates
        cork_lat, cork_lon = 51.8985, -8.4756

        distance = haversine_distance(dublin_lat, dublin_lon, cork_lat, cork_lon)

        # Expected distance is approximately 219 km
        assert distance == pytest.approx(219400, rel=0.01)  # Within 1%

    def test_short_distance(self):
        """Test short distance calculation (typical cell coverage)."""
        # Points about 750m apart
        lat1, lon1 = 53.3498, -6.2603
        lat2, lon2 = 53.3498, -6.2489  # ~750m east

        distance = haversine_distance(lat1, lon1, lat2, lon2)

        assert 700 < distance < 800  # Approximately 750m

    def test_equator_distance(self):
        """Test distance calculation on the equator."""
        # 1 degree of longitude at equator ≈ 111.32 km
        distance = haversine_distance(0, 0, 0, 1)

        assert distance == pytest.approx(111320, rel=0.01)

    def test_north_south_distance(self):
        """Test pure north-south distance."""
        # 1 degree of latitude ≈ 111.32 km everywhere
        distance = haversine_distance(53, -6, 54, -6)

        assert distance == pytest.approx(111190, rel=0.01)

    def test_negative_coordinates(self):
        """Test with negative coordinates (Southern/Western hemispheres)."""
        # Sydney to Melbourne, Australia
        sydney_lat, sydney_lon = -33.8688, 151.2093
        melbourne_lat, melbourne_lon = -37.8136, 144.9631

        distance = haversine_distance(
            sydney_lat, sydney_lon,
            melbourne_lat, melbourne_lon
        )

        # Expected distance ~714 km
        assert distance == pytest.approx(714000, rel=0.02)


class TestCalculateBearing:
    """Tests for calculate_bearing function."""

    def test_bearing_north(self):
        """Bearing directly north should be 0° (or 360°)."""
        bearing = calculate_bearing(53.0, -6.0, 54.0, -6.0)
        assert bearing == pytest.approx(0.0, abs=0.5) or \
               bearing == pytest.approx(360.0, abs=0.5)

    def test_bearing_east(self):
        """Bearing directly east should be 90°."""
        bearing = calculate_bearing(53.0, -6.0, 53.0, -5.0)
        assert bearing == pytest.approx(90.0, abs=0.5)

    def test_bearing_south(self):
        """Bearing directly south should be 180°."""
        bearing = calculate_bearing(54.0, -6.0, 53.0, -6.0)
        assert bearing == pytest.approx(180.0, abs=0.5)

    def test_bearing_west(self):
        """Bearing directly west should be 270°."""
        bearing = calculate_bearing(53.0, -5.0, 53.0, -6.0)
        assert bearing == pytest.approx(270.0, abs=0.5)

    def test_bearing_northeast(self):
        """Bearing northeast should be ~31°."""
        bearing = calculate_bearing(53.0, -6.0, 53.1, -5.9)
        assert 28 < bearing < 34

    def test_bearing_southwest(self):
        """Bearing southwest should be ~211°."""
        bearing = calculate_bearing(53.1, -5.9, 53.0, -6.0)
        assert 208 < bearing < 214

    def test_bearing_range(self):
        """Bearing should always be in range [0, 360)."""
        # Test various points
        test_points = [
            (53.0, -6.0, 54.0, -5.0),  # NE
            (53.0, -6.0, 52.0, -5.0),  # SE
            (53.0, -6.0, 52.0, -7.0),  # SW
            (53.0, -6.0, 54.0, -7.0),  # NW
        ]

        for lat1, lon1, lat2, lon2 in test_points:
            bearing = calculate_bearing(lat1, lon1, lat2, lon2)
            assert 0 <= bearing < 360

    def test_same_point_bearing(self):
        """Bearing from a point to itself is undefined but should not error."""
        bearing = calculate_bearing(53.0, -6.0, 53.0, -6.0)
        # Should return a valid number (likely 0 or based on rounding)
        assert 0 <= bearing < 360


class TestBearingDifference:
    """Tests for bearing_difference function."""

    def test_same_bearing(self):
        """Difference between same bearing should be 0."""
        diff = bearing_difference(45, 45)
        assert diff == 0.0

    def test_opposite_bearings(self):
        """Difference between opposite bearings should be 180°."""
        diff = bearing_difference(0, 180)
        assert diff == 180.0

        diff = bearing_difference(45, 225)
        assert diff == 180.0

    def test_small_difference(self):
        """Test small angular difference."""
        diff = bearing_difference(45, 50)
        assert diff == 5.0

    def test_wraparound_difference(self):
        """Test difference across 0°/360° boundary."""
        # 350° to 10° should be 20°, not 340°
        diff = bearing_difference(350, 10)
        assert diff == 20.0

        # 10° to 350° should also be 20°
        diff = bearing_difference(10, 350)
        assert diff == 20.0

    def test_large_difference(self):
        """Test large angular difference."""
        diff = bearing_difference(0, 120)
        assert diff == 120.0

    def test_difference_symmetry(self):
        """Difference should be symmetric."""
        diff1 = bearing_difference(45, 90)
        diff2 = bearing_difference(90, 45)
        assert diff1 == diff2

    def test_difference_range(self):
        """Difference should always be in range [0, 180]."""
        test_cases = [
            (0, 0), (0, 90), (0, 180), (0, 270), (0, 359),
            (45, 90), (45, 225), (90, 270), (180, 0), (270, 90)
        ]

        for b1, b2 in test_cases:
            diff = bearing_difference(b1, b2)
            assert 0 <= diff <= 180


class TestGetDistanceAndBearing:
    """Tests for get_distance_and_bearing function."""

    def test_returns_tuple(self):
        """Should return tuple of (distance, bearing)."""
        result = get_distance_and_bearing(53.0, -6.0, 54.0, -6.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matches_individual_functions(self):
        """Should match calling haversine_distance and calculate_bearing separately."""
        lat1, lon1 = 53.3498, -6.2603
        lat2, lon2 = 53.3500, -6.2600

        distance, bearing = get_distance_and_bearing(lat1, lon1, lat2, lon2)

        expected_distance = haversine_distance(lat1, lon1, lat2, lon2)
        expected_bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        assert distance == pytest.approx(expected_distance)
        assert bearing == pytest.approx(expected_bearing)

    def test_typical_cell_to_bin(self):
        """Test typical cell-to-grid-bin calculation."""
        # Cell location
        cell_lat, cell_lon = 53.3498, -6.2603
        # Bin location (500m northeast)
        bin_lat, bin_lon = 53.3543, -6.2550

        distance, bearing = get_distance_and_bearing(
            cell_lat, cell_lon, bin_lat, bin_lon
        )

        # Should be roughly 500-700m northeast (30-60°)
        assert 400 < distance < 800
        assert 30 < bearing < 60


class TestIsWithinSector:
    """Tests for is_within_sector function."""

    def test_directly_ahead(self):
        """Measurement directly ahead of antenna should be in sector."""
        antenna_azimuth = 45
        measurement_bearing = 45
        assert is_within_sector(measurement_bearing, antenna_azimuth)

    def test_within_beamwidth(self):
        """Measurement within beamwidth should be in sector."""
        antenna_azimuth = 45
        measurement_bearing = 60  # 15° off-boresight
        assert is_within_sector(measurement_bearing, antenna_azimuth, beamwidth=65)

    def test_outside_beamwidth(self):
        """Measurement outside beamwidth should not be in sector."""
        antenna_azimuth = 45
        measurement_bearing = 120  # 75° off-boresight
        assert not is_within_sector(measurement_bearing, antenna_azimuth, beamwidth=65)

    def test_behind_antenna(self):
        """Measurement behind antenna should not be in sector."""
        antenna_azimuth = 45
        measurement_bearing = 225  # 180° opposite
        assert not is_within_sector(measurement_bearing, antenna_azimuth, beamwidth=65)

    def test_edge_of_sector(self):
        """Measurement at edge of sector should be included."""
        antenna_azimuth = 45
        half_beamwidth = 32.5
        measurement_bearing = 45 + half_beamwidth  # Exactly at edge

        assert is_within_sector(measurement_bearing, antenna_azimuth, beamwidth=65)

    def test_narrow_beamwidth(self):
        """Test with narrow beamwidth (e.g., directional antenna)."""
        antenna_azimuth = 0
        assert is_within_sector(15, antenna_azimuth, beamwidth=30)
        assert not is_within_sector(20, antenna_azimuth, beamwidth=30)

    def test_wraparound_sector(self):
        """Test sector that wraps around 0°/360°."""
        antenna_azimuth = 350  # Pointing NNW
        # 10° is 20° away from 350° (crossing 0°)
        assert is_within_sector(10, antenna_azimuth, beamwidth=65)


class TestEarthRadius:
    """Tests for EARTH_RADIUS_M constant."""

    def test_earth_radius_value(self):
        """Earth radius should be approximately 6,371 km."""
        assert EARTH_RADIUS_M == pytest.approx(6371000, rel=0.01)

    def test_earth_radius_used_correctly(self):
        """Verify Earth radius is used in distance calculation."""
        # 1 degree at equator should be ~111.32 km
        # This verifies EARTH_RADIUS_M is used correctly
        distance = haversine_distance(0, 0, 0, 1)
        expected = 2 * math.pi * EARTH_RADIUS_M / 360
        assert distance == pytest.approx(expected, rel=0.01)
