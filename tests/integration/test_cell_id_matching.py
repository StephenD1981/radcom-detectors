"""
Integration test for cell ID matching between grid and GIS data.
"""
import pytest
from pathlib import Path
import pandas as pd

# Data paths - using current structure
GRID_FILE = Path("data/vf-ie/input-data/cell_coverage.csv")
GIS_FILE = Path("data/vf-ie/input-data/cork-gis.csv")


@pytest.fixture
def grid_df():
    """Load grid data if available."""
    if not GRID_FILE.exists():
        pytest.skip(f"Grid file not found: {GRID_FILE}")
    return pd.read_csv(GRID_FILE, nrows=50000, low_memory=False)


@pytest.fixture
def gis_df():
    """Load GIS data if available."""
    if not GIS_FILE.exists():
        pytest.skip(f"GIS file not found: {GIS_FILE}")
    return pd.read_csv(GIS_FILE)


def test_cell_id_columns_exist(grid_df, gis_df):
    """Test that cell ID columns exist in both datasets."""
    # Grid should have cilac or cell_id
    grid_cell_col = None
    for col in ['cilac', 'cell_id', 'CILAC']:
        if col in grid_df.columns:
            grid_cell_col = col
            break
    assert grid_cell_col is not None, f"No cell ID column found in grid data. Columns: {list(grid_df.columns)}"

    # GIS should have CILAC or cell_id
    gis_cell_col = None
    for col in ['CILAC', 'cell_id', 'cilac']:
        if col in gis_df.columns:
            gis_cell_col = col
            break
    assert gis_cell_col is not None, f"No cell ID column found in GIS data. Columns: {list(gis_df.columns)}"


def test_cell_ids_match(grid_df, gis_df):
    """Test that there are matching cell IDs between grid and GIS."""
    # Find cell ID columns
    grid_cell_col = 'cilac' if 'cilac' in grid_df.columns else 'cell_id'
    gis_cell_col = 'CILAC' if 'CILAC' in gis_df.columns else 'cell_id'

    grid_cells = set(grid_df[grid_cell_col].unique())
    gis_cells = set(gis_df[gis_cell_col].unique())

    matches = grid_cells & gis_cells
    assert len(matches) > 0, f"No matching cells! Grid has {len(grid_cells)} cells, GIS has {len(gis_cells)} cells"

    # Should have significant overlap
    overlap_pct = len(matches) / len(grid_cells) * 100
    print(f"\nCell ID matching: {len(matches)} matches ({overlap_pct:.1f}% of grid cells)")
    assert overlap_pct > 50, f"Low overlap: only {overlap_pct:.1f}% of grid cells match GIS"


def test_grid_has_required_columns(grid_df):
    """Test that grid data has required columns."""
    required_cols = ['avg_rsrp', 'distance_to_cell', 'event_count']
    aliases = {
        'avg_rsrp': ['rsrp', 'RSRP'],
        'distance_to_cell': ['distance_m', 'distance'],
        'event_count': ['total_traffic', 'traffic'],
    }

    for req_col in required_cols:
        found = req_col in grid_df.columns
        if not found:
            for alias in aliases.get(req_col, []):
                if alias in grid_df.columns:
                    found = True
                    break
        assert found, f"Required column '{req_col}' (or alias) not found in grid data"


def test_gis_has_required_columns(gis_df):
    """Test that GIS data has required columns."""
    required_cols = ['Latitude', 'Longitude', 'Bearing', 'Height', 'TiltE', 'TiltM']
    aliases = {
        'Latitude': ['latitude', 'lat'],
        'Longitude': ['longitude', 'lon', 'lng'],
        'Bearing': ['azimuth_deg', 'azimuth'],
        'Height': ['height_m', 'antenna_height'],
        'TiltE': ['electrical_tilt', 'etilt'],
        'TiltM': ['mechanical_tilt', 'mtilt'],
    }

    for req_col in required_cols:
        found = req_col in gis_df.columns
        if not found:
            for alias in aliases.get(req_col, []):
                if alias in gis_df.columns:
                    found = True
                    break
        assert found, f"Required column '{req_col}' (or alias) not found in GIS data"
