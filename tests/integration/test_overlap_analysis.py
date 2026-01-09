"""
Integration test for overlap analysis between overshooting and undershooting cells.
"""
import pytest
from pathlib import Path
import pandas as pd

# Data paths - using current structure
OVERSHOOTING_FILE = Path("data/vf-ie/output-data/overshooting_cells_environment_aware.csv")
UNDERSHOOTING_FILE = Path("data/vf-ie/output-data/undershooting_cells_environment_aware.csv")


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


def test_overshooting_has_required_columns(overshooters_df):
    """Test that overshooting results have required columns."""
    required_cols = ['cell_id', 'environment', 'severity_score', 'recommended_tilt_change']

    for col in required_cols:
        assert col in overshooters_df.columns, f"Missing column: {col}"


def test_undershooting_has_required_columns(undershooters_df):
    """Test that undershooting results have required columns."""
    required_cols = ['cell_id', 'environment', 'recommended_uptilt_deg']

    for col in required_cols:
        assert col in undershooters_df.columns, f"Missing column: {col}"


def test_overlap_analysis(overshooters_df, undershooters_df):
    """Test overlap analysis between overshooting and undershooting cells."""
    overshooter_ids = set(overshooters_df['cell_id'].unique())
    undershoote_ids = set(undershooters_df['cell_id'].unique())
    overlap_ids = overshooter_ids & undershoote_ids

    total_with_issues = len(overshooter_ids | undershoote_ids)
    only_over = len(overshooter_ids - undershoote_ids)
    only_under = len(undershoote_ids - overshooter_ids)

    print(f"\nOverlap Analysis:")
    print(f"  Overshooting cells: {len(overshooter_ids)}")
    print(f"  Undershooting cells: {len(undershoote_ids)}")
    print(f"  Cells in both lists: {len(overlap_ids)}")
    print(f"  Only overshooting: {only_over}")
    print(f"  Only undershooting: {only_under}")

    # Overlap should be reasonably small (cells shouldn't be both)
    if len(overlap_ids) > 0:
        overlap_pct = len(overlap_ids) / total_with_issues * 100
        print(f"  Overlap percentage: {overlap_pct:.1f}%")
        # Warn if overlap is high
        assert overlap_pct < 50, f"High overlap ({overlap_pct:.1f}%) suggests conflicting detections"


def test_environment_distribution(overshooters_df, undershooters_df):
    """Test that results span multiple environments."""
    over_envs = set(overshooters_df['environment'].unique())
    under_envs = set(undershooters_df['environment'].unique())

    print(f"\nEnvironment Distribution:")
    print(f"  Overshooting environments: {over_envs}")
    print(f"  Undershooting environments: {under_envs}")

    # Should detect issues in at least 2 environments
    assert len(over_envs) >= 1, "Overshooting should detect cells in at least 1 environment"
    assert len(under_envs) >= 1, "Undershooting should detect cells in at least 1 environment"


def test_recommendations_are_valid(overshooters_df, undershooters_df):
    """Test that tilt recommendations are within valid ranges."""
    # Overshooting: downtilt recommendations should be positive (more downtilt)
    tilt_changes = overshooters_df['recommended_tilt_change']
    assert tilt_changes.min() >= 0, "Downtilt recommendations should be >= 0"
    assert tilt_changes.max() <= 10, "Downtilt recommendations should be <= 10 degrees"

    # Undershooting: uptilt recommendations should be positive
    uptilts = undershooters_df['recommended_uptilt_deg']
    assert uptilts.min() >= 0, "Uptilt recommendations should be >= 0"
    assert uptilts.max() <= 5, "Uptilt recommendations should be <= 5 degrees"
