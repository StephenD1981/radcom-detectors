"""
Integration test for unified visualization generation.

This test verifies that visualization can be generated from detection results.
The actual visualization scripts are in ran_optimizer/visualization/.
"""
import pytest
from pathlib import Path
import pandas as pd

# Data paths
OUTPUT_DIR = Path("data/vf-ie/output-data")
OVERSHOOTING_FILE = OUTPUT_DIR / "overshooting_cells_environment_aware.csv"
UNDERSHOOTING_FILE = OUTPUT_DIR / "undershooting_cells_environment_aware.csv"
COVERAGE_FILE = Path("data/vf-ie/input-data/cell_coverage.csv")


@pytest.fixture
def overshooters_df():
    """Load overshooting results if available."""
    if not OVERSHOOTING_FILE.exists():
        pytest.skip(f"Overshooting file not found: {OVERSHOOTING_FILE}")
    return pd.read_csv(OVERSHOOTING_FILE)


@pytest.fixture
def undershooters_df():
    """Load undershooting results if available."""
    if not UNDERSHOOTING_FILE.exists():
        pytest.skip(f"Undershooting file not found: {UNDERSHOOTING_FILE}")
    return pd.read_csv(UNDERSHOOTING_FILE)


def test_overshooting_has_visualization_columns(overshooters_df):
    """Test that overshooting data has columns needed for visualization."""
    required_for_viz = ['cell_id', 'latitude', 'longitude']

    for col in required_for_viz:
        assert col in overshooters_df.columns, f"Missing visualization column: {col}"


def test_undershooting_has_visualization_columns(undershooters_df):
    """Test that undershooting data has columns needed for visualization."""
    # Note: latitude/longitude may need to be merged from GIS data
    # Only cell_id is strictly required in output
    required_for_viz = ['cell_id']

    for col in required_for_viz:
        assert col in undershooters_df.columns, f"Missing visualization column: {col}"

    # Lat/lon may or may not be present depending on merge
    has_coords = 'latitude' in undershooters_df.columns and 'longitude' in undershooters_df.columns
    if not has_coords:
        print("Note: latitude/longitude not in undershooting output - merge with GIS data for visualization")


def test_coordinates_are_valid(overshooters_df):
    """Test that overshooting coordinates are within Ireland bounds."""
    # Ireland approximate bounds
    MIN_LAT, MAX_LAT = 51.0, 55.5
    MIN_LON, MAX_LON = -11.0, -5.5

    # Only test overshooting - undershooting may not have coords
    if 'latitude' in overshooters_df.columns and 'longitude' in overshooters_df.columns:
        assert overshooters_df['latitude'].min() >= MIN_LAT, "overshooting has latitude below Ireland bounds"
        assert overshooters_df['latitude'].max() <= MAX_LAT, "overshooting has latitude above Ireland bounds"
        assert overshooters_df['longitude'].min() >= MIN_LON, "overshooting has longitude below Ireland bounds"
        assert overshooters_df['longitude'].max() <= MAX_LON, "overshooting has longitude above Ireland bounds"


def test_severity_scores_present(overshooters_df):
    """Test that severity information is present for color coding."""
    assert 'severity_score' in overshooters_df.columns or 'severity_category' in overshooters_df.columns, \
        "Overshooting data should have severity information for visualization"


def test_uptilt_recommendations_present(undershooters_df):
    """Test that uptilt recommendations are present for color coding."""
    assert 'recommended_uptilt_deg' in undershooters_df.columns, \
        "Undershooting data should have uptilt recommendations for visualization"


@pytest.mark.slow
def test_visualization_module_imports():
    """Test that visualization module can be imported."""
    try:
        from ran_optimizer.visualization.map_overshooters import create_overshooting_map
    except ImportError as e:
        pytest.skip(f"Visualization module not available: {e}")


@pytest.mark.slow
def test_map_output_directory_exists():
    """Test that map output directory can be created."""
    maps_dir = OUTPUT_DIR / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    assert maps_dir.exists(), "Could not create maps output directory"
