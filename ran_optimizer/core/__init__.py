"""
Core algorithm modules for RAN optimization.

Contains geometry, RF modeling, and analysis functions extracted
from legacy prototype code.
"""
from ran_optimizer.core.geometry import (
    haversine_distance,
    calculate_bearing,
    bearing_difference,
    get_distance_and_bearing,
    is_within_sector,
    EARTH_RADIUS_M,
)

from ran_optimizer.core.environment_classifier import (
    EnvironmentClassifier,
    classify_cell_environments,
    load_or_create_cell_environments,
)

__all__ = [
    # Geometry
    'haversine_distance',
    'calculate_bearing',
    'bearing_difference',
    'get_distance_and_bearing',
    'is_within_sector',
    'EARTH_RADIUS_M',
    # Environment classification
    'EnvironmentClassifier',
    'classify_cell_environments',
    'load_or_create_cell_environments',
]
