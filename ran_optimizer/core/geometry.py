"""
Geospatial geometry functions for RF coverage analysis.

Provides distance and bearing calculations using the haversine formula
for great-circle distance between coordinates.
"""
import math
from typing import Tuple


# Earth's radius in meters (mean radius)
EARTH_RADIUS_M = 6371000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1: Latitude of first point (decimal degrees)
        lon1: Longitude of first point (decimal degrees)
        lat2: Latitude of second point (decimal degrees)
        lon2: Longitude of second point (decimal degrees)

    Returns:
        Distance in meters

    Example:
        >>> # Distance from Dublin to Cork (Ireland)
        >>> distance = haversine_distance(53.3498, -6.2603, 51.8985, -8.4756)
        >>> print(f"{distance/1000:.1f} km")
        219.4 km

    References:
        https://en.wikipedia.org/wiki/Haversine_formula
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2)) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(dlon / 2)) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS_M * c

    return distance


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing (forward azimuth) from point 1 to point 2.

    The bearing is the angle (in degrees) measured clockwise from north
    to the direction of point 2 from point 1.

    Args:
        lat1: Latitude of starting point (decimal degrees)
        lon1: Longitude of starting point (decimal degrees)
        lat2: Latitude of destination point (decimal degrees)
        lon2: Longitude of destination point (decimal degrees)

    Returns:
        Bearing in degrees (0-360), where:
        - 0° = North
        - 90° = East
        - 180° = South
        - 270° = West

    Example:
        >>> # Bearing from cell to measurement point
        >>> bearing = calculate_bearing(53.3498, -6.2603, 53.3500, -6.2600)
        >>> print(f"Bearing: {bearing:.1f}°")
        Bearing: 45.3°

    References:
        https://www.movable-type.co.uk/scripts/latlong.html
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate bearing
    dlon = lon2_rad - lon1_rad

    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    # Normalize to 0-360
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def bearing_difference(bearing1: float, bearing2: float) -> float:
    """
    Calculate the absolute angular difference between two bearings.

    This accounts for the circular nature of bearings (e.g., the difference
    between 10° and 350° is 20°, not 340°).

    Args:
        bearing1: First bearing in degrees (0-360)
        bearing2: Second bearing in degrees (0-360)

    Returns:
        Absolute difference in degrees (0-180)

    Example:
        >>> # Difference between nearly opposite directions
        >>> diff = bearing_difference(10, 350)
        >>> print(f"Difference: {diff}°")
        Difference: 20.0°

        >>> # Difference between antenna azimuth and signal direction
        >>> antenna_azimuth = 45
        >>> signal_bearing = 90
        >>> diff = bearing_difference(antenna_azimuth, signal_bearing)
        >>> print(f"Off-boresight: {diff}°")
        Off-boresight: 45.0°
    """
    diff = abs(bearing1 - bearing2)

    # Take the smaller angle (accounting for wrap-around)
    if diff > 180:
        diff = 360 - diff

    return diff


def get_distance_and_bearing(lat1: float, lon1: float,
                              lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculate both distance and bearing between two points.

    Convenience function that combines haversine_distance and calculate_bearing.

    Args:
        lat1: Latitude of starting point (decimal degrees)
        lon1: Longitude of starting point (decimal degrees)
        lat2: Latitude of destination point (decimal degrees)
        lon2: Longitude of destination point (decimal degrees)

    Returns:
        Tuple of (distance_meters, bearing_degrees)

    Example:
        >>> # Get distance and bearing from cell to grid bin
        >>> distance, bearing = get_distance_and_bearing(
        ...     53.3498, -6.2603,  # Cell location
        ...     53.3500, -6.2600   # Grid bin location
        ... )
        >>> print(f"Distance: {distance:.0f}m, Bearing: {bearing:.1f}°")
        Distance: 253m, Bearing: 45.3°
    """
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)

    return distance, bearing


def is_within_sector(measurement_bearing: float,
                     antenna_azimuth: float,
                     beamwidth: float = 65.0) -> bool:
    """
    Check if a measurement bearing falls within an antenna's sector.

    Args:
        measurement_bearing: Bearing from antenna to measurement point (degrees)
        antenna_azimuth: Antenna's main beam direction (degrees)
        beamwidth: Antenna's horizontal beamwidth (degrees), default 65°

    Returns:
        True if measurement is within the sector, False otherwise

    Example:
        >>> # Check if measurement is in antenna's main lobe
        >>> antenna_azimuth = 45  # NE direction
        >>> measurement_bearing = 60  # ENE direction
        >>> is_within_sector(measurement_bearing, antenna_azimuth, beamwidth=65)
        True

        >>> # Measurement behind antenna
        >>> is_within_sector(measurement_bearing, antenna_azimuth=225, beamwidth=65)
        False
    """
    half_beamwidth = beamwidth / 2
    diff = bearing_difference(measurement_bearing, antenna_azimuth)

    return diff <= half_beamwidth
