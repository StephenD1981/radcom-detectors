"""
Geohash encoding and decoding utilities.

Provides functions to convert between geohash strings and lat/lon coordinates.
Implementation based on standard geohash algorithm.
"""
from typing import Tuple, Set
from shapely.geometry import Polygon


# Base32 encoding for geohash
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def decode(geohash: str) -> Tuple[float, float]:
    """
    Decode a geohash string to latitude/longitude coordinates.

    Returns the center point of the geohash box.

    Args:
        geohash: Geohash string (e.g., "gc7x3r4")

    Returns:
        Tuple of (latitude, longitude) in decimal degrees

    Example:
        >>> lat, lon = decode("gc7x3r4")
        >>> print(f"{lat:.5f}, {lon:.5f}")
        53.34961, -6.26037

    References:
        https://en.wikipedia.org/wiki/Geohash
    """
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]

    is_even = True  # Start with longitude

    for char in geohash.lower():
        if char not in BASE32:
            raise ValueError(f"Invalid geohash character: {char}")

        idx = BASE32.index(char)

        # Each character encodes 5 bits
        for i in range(4, -1, -1):
            bit = (idx >> i) & 1

            if is_even:  # Longitude bit
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit == 1:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:  # Latitude bit
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit == 1:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid

            is_even = not is_even

    # Return center of box
    latitude = (lat_range[0] + lat_range[1]) / 2
    longitude = (lon_range[0] + lon_range[1]) / 2

    return latitude, longitude


def encode(latitude: float, longitude: float, precision: int = 7) -> str:
    """
    Encode latitude/longitude to a geohash string.

    Args:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        precision: Number of characters in geohash (default 7)

    Returns:
        Geohash string

    Example:
        >>> geohash = encode(53.3498, -6.2603, precision=7)
        >>> print(geohash)
        gc7x3r4

    Raises:
        ValueError: If lat/lon out of valid range
    """
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude must be in [-90, 90], got {latitude}")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude must be in [-180, 180], got {longitude}")

    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]

    geohash = []
    bits = 0
    bit_count = 0
    is_even = True  # Start with longitude

    while len(geohash) < precision:
        if is_even:  # Longitude
            mid = (lon_range[0] + lon_range[1]) / 2
            if longitude >= mid:
                bits |= (1 << (4 - bit_count))
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:  # Latitude
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:
                bits |= (1 << (4 - bit_count))
                lat_range[0] = mid
            else:
                lat_range[1] = mid

        is_even = not is_even
        bit_count += 1

        if bit_count == 5:
            geohash.append(BASE32[bits])
            bits = 0
            bit_count = 0

    return ''.join(geohash)


def get_box_bounds(geohash: str) -> Tuple[float, float, float, float]:
    """
    Get the bounding box for a geohash.

    Args:
        geohash: Geohash string

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)

    Example:
        >>> bounds = get_box_bounds("gc7x3r4")
        >>> min_lat, max_lat, min_lon, max_lon = bounds
        >>> print(f"Lat: [{min_lat:.5f}, {max_lat:.5f}]")
        >>> print(f"Lon: [{min_lon:.5f}, {max_lon:.5f}]")
    """
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]

    is_even = True

    for char in geohash.lower():
        if char not in BASE32:
            raise ValueError(f"Invalid geohash character: {char}")

        idx = BASE32.index(char)

        for i in range(4, -1, -1):
            bit = (idx >> i) & 1

            if is_even:  # Longitude
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit == 1:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:  # Latitude
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit == 1:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid

            is_even = not is_even

    return lat_range[0], lat_range[1], lon_range[0], lon_range[1]


def get_precision_dimensions(precision: int) -> Tuple[float, float]:
    """
    Get approximate dimensions of a geohash box at given precision.

    Args:
        precision: Number of geohash characters (1-12)

    Returns:
        Tuple of (lat_error_km, lon_error_km) - approximate box size

    Example:
        >>> lat_km, lon_km = get_precision_dimensions(7)
        >>> print(f"7-character geohash: ~{lat_km:.0f}m x {lon_km:.0f}m")
        7-character geohash: ~153m x 153m

    Note:
        These are approximations at the equator. Actual size varies with latitude.
    """
    # Approximate error margins (in km) for different precisions
    # Source: https://en.wikipedia.org/wiki/Geohash
    precision_to_km = {
        1: (5000, 5000),     # ±2500km
        2: (625, 1250),      # ±313km x ±625km
        3: (156, 156),       # ±78km
        4: (19.5, 39.1),     # ±9.8km x ±19.5km
        5: (4.9, 4.9),       # ±2.4km
        6: (0.61, 1.22),     # ±305m x ±610m
        7: (0.153, 0.153),   # ±76m
        8: (0.019, 0.038),   # ±9.5m x ±19m
        9: (0.0047, 0.0047), # ±2.4m
        10: (0.0006, 0.0012),# ±30cm x ±60cm
        11: (0.00015, 0.00015), # ±7.5cm
        12: (0.000019, 0.000037), # ±1.9cm x ±3.7cm
    }

    if precision not in precision_to_km:
        raise ValueError(f"Precision must be 1-12, got {precision}")

    return precision_to_km[precision]


def neighbors_grid(geohash: str, radius: int = 1) -> list:
    """
    Get all geohash neighbors within a grid radius.

    Args:
        geohash: Center geohash string
        radius: Grid radius (1 = 3x3 grid = 9 cells, 2 = 5x5 = 25 cells, 3 = 7x7 = 49 cells)

    Returns:
        List of geohash strings including center and all neighbors

    Example:
        >>> neighbors = neighbors_grid("gc7x3r4", radius=1)
        >>> len(neighbors)  # 3x3 grid
        9
        >>> neighbors = neighbors_grid("gc7x3r4", radius=3)
        >>> len(neighbors)  # 7x7 grid
        49
    """
    # Get center point
    center_lat, center_lon = decode(geohash)

    # Get approximate cell size
    min_lat, max_lat, min_lon, max_lon = get_box_bounds(geohash)
    cell_height = max_lat - min_lat
    cell_width = max_lon - min_lon

    precision = len(geohash)
    neighbors = []

    # Generate grid of points around center
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            # Calculate neighbor position
            neighbor_lat = center_lat + (i * cell_height)
            neighbor_lon = center_lon + (j * cell_width)

            # Ensure within valid lat/lon bounds
            if -90 <= neighbor_lat <= 90 and -180 <= neighbor_lon <= 180:
                try:
                    neighbor_hash = encode(neighbor_lat, neighbor_lon, precision=precision)
                    neighbors.append(neighbor_hash)
                except ValueError:
                    # Skip invalid coordinates
                    pass

    return neighbors


def neighbors_8(geohash: str) -> Set[str]:
    """
    Get the 8 immediate neighbors of a geohash cell.

    Args:
        geohash: Center geohash string

    Returns:
        Set of 8 neighboring geohash strings (excluding center)

    Example:
        >>> neighbors = neighbors_8("gc7x3r4")
        >>> len(neighbors)
        8
    """
    # Get center point
    center_lat, center_lon = decode(geohash)

    # Get cell dimensions
    min_lat, max_lat, min_lon, max_lon = get_box_bounds(geohash)
    cell_height = max_lat - min_lat
    cell_width = max_lon - min_lon

    precision = len(geohash)
    neighbors = set()

    # 8 directions: N, S, E, W, NE, NW, SE, SW
    offsets = [
        (-1, 0),   # North
        (1, 0),    # South
        (0, 1),    # East
        (0, -1),   # West
        (-1, 1),   # Northeast
        (-1, -1),  # Northwest
        (1, 1),    # Southeast
        (1, -1),   # Southwest
    ]

    for lat_offset, lon_offset in offsets:
        neighbor_lat = center_lat + (lat_offset * cell_height)
        neighbor_lon = center_lon + (lon_offset * cell_width)

        # Ensure within valid bounds
        if -90 <= neighbor_lat <= 90 and -180 <= neighbor_lon <= 180:
            try:
                neighbor_hash = encode(neighbor_lat, neighbor_lon, precision=precision)
                neighbors.add(neighbor_hash)
            except ValueError:
                # Skip invalid coordinates
                pass

    return neighbors


def kring(geohash: str, k: int) -> Set[str]:
    """
    Get all geohashes within k neighbor steps using BFS (breadth-first search).

    This returns the k-ring of neighbors around a center geohash. For k=1, returns
    the center plus 8 immediate neighbors (9 total). For k=2, adds the neighbors
    of those neighbors, etc.

    Args:
        geohash: Center geohash string
        k: Number of neighbor steps (0 = just center, 1 = 3x3 = 9, 2 = 5x5 = 25, 3 = 7x7 = 49)

    Returns:
        Set of geohash strings within k steps (including center)

    Example:
        >>> ring = kring("gc7x3r4", k=1)
        >>> len(ring)
        9
        >>> ring = kring("gc7x3r4", k=3)
        >>> len(ring)
        49

    Note:
        Uses BFS to expand outward from center, adding one ring at a time.
    """
    if k == 0:
        return {geohash}

    seen = {geohash}
    frontier = {geohash}

    for _ in range(k):
        next_frontier = set()
        for gh in frontier:
            neighbors = neighbors_8(gh)
            next_frontier.update(neighbors)

        # Remove already seen geohashes
        next_frontier -= seen
        seen |= next_frontier
        frontier = next_frontier

    return seen


def geohash_polygon(geohash: str) -> Polygon:
    """
    Convert a geohash to a Shapely Polygon representing its bounding box.

    Args:
        geohash: Geohash string

    Returns:
        Shapely Polygon representing the geohash bounding box

    Example:
        >>> poly = geohash_polygon("gc7x3r4")
        >>> poly.area > 0
        True
    """
    min_lat, max_lat, min_lon, max_lon = get_box_bounds(geohash)

    return Polygon([
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ])
