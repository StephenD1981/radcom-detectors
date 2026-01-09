"""
Integration tests for the enhanced unified visualization.

Tests the new enhanced_map module which combines all 4 algorithm outputs
into a single interactive dashboard.
"""
import pytest
from pathlib import Path
import pandas as pd
import geopandas as gpd

# Test data paths
OUTPUT_DIR = Path("data/vf-ie/output-data")
INPUT_DIR = Path("data/vf-ie/input-data")
GIS_FILE = INPUT_DIR / "cork-gis.csv"
OVERSHOOTING_FILE = OUTPUT_DIR / "overshooting_cells_environment_aware.csv"
UNDERSHOOTING_FILE = OUTPUT_DIR / "undershooting_cells_environment_aware.csv"
NO_COVERAGE_FILE = OUTPUT_DIR / "no_coverage_clusters.geojson"


@pytest.fixture
def gis_df():
    """Load GIS data if available."""
    if not GIS_FILE.exists():
        pytest.skip(f"GIS file not found: {GIS_FILE}")
    return pd.read_csv(GIS_FILE)


@pytest.fixture
def overshooting_df():
    """Load overshooting results if available."""
    if not OVERSHOOTING_FILE.exists():
        pytest.skip(f"Overshooting file not found: {OVERSHOOTING_FILE}")
    return pd.read_csv(OVERSHOOTING_FILE)


@pytest.fixture
def undershooting_df():
    """Load undershooting results if available."""
    if not UNDERSHOOTING_FILE.exists():
        pytest.skip(f"Undershooting file not found: {UNDERSHOOTING_FILE}")
    return pd.read_csv(UNDERSHOOTING_FILE)


@pytest.fixture
def no_coverage_gdf():
    """Load no coverage results if available."""
    if not NO_COVERAGE_FILE.exists():
        pytest.skip(f"No coverage file not found: {NO_COVERAGE_FILE}")
    return gpd.read_file(NO_COVERAGE_FILE)


@pytest.fixture
def low_coverage_gdfs():
    """Load low coverage results per band."""
    gdfs = {}
    for f in OUTPUT_DIR.glob("low_coverage_band_*.geojson"):
        band = f.stem.replace("low_coverage_band_", "")
        gdfs[band] = gpd.read_file(f)
    if not gdfs:
        pytest.skip("No low coverage files found")
    return gdfs


def test_enhanced_map_module_imports():
    """Test that enhanced map module can be imported."""
    from ran_optimizer.visualization.enhanced_map import (
        create_enhanced_map,
        generate_enhanced_map_from_files,
    )
    assert callable(create_enhanced_map)
    assert callable(generate_enhanced_map_from_files)


def test_enhanced_map_creates_html(gis_df, overshooting_df, undershooting_df, tmp_path):
    """Test that enhanced map generates valid HTML output."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "test_map.html"

    m = create_enhanced_map(
        overshooting_df=overshooting_df,
        undershooting_df=undershooting_df,
        gis_df=gis_df,
        output_file=output_file,
    )

    assert output_file.exists(), "Output HTML file was not created"
    assert output_file.stat().st_size > 0, "Output HTML file is empty"

    # Check HTML contains expected elements
    html_content = output_file.read_text()
    assert "<html" in html_content.lower(), "Not valid HTML"
    assert "folium" in html_content.lower() or "leaflet" in html_content.lower(), "Missing map library"


def test_enhanced_map_with_all_layers(
    gis_df, overshooting_df, undershooting_df, no_coverage_gdf, low_coverage_gdfs, tmp_path
):
    """Test enhanced map with all 4 algorithm outputs."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "full_map.html"

    m = create_enhanced_map(
        overshooting_df=overshooting_df,
        undershooting_df=undershooting_df,
        gis_df=gis_df,
        no_coverage_gdf=no_coverage_gdf,
        low_coverage_gdfs=low_coverage_gdfs,
        output_file=output_file,
        title="Test Dashboard",
    )

    assert output_file.exists()
    html_content = output_file.read_text()

    # Check for summary panel
    assert "Summary Statistics" in html_content, "Missing summary panel"

    # Check for filter controls
    assert "issueFilter" in html_content, "Missing issue type filter"
    assert "severityFilter" in html_content, "Missing severity filter"


def test_generate_from_files(tmp_path):
    """Test generating enhanced map from output files."""
    from ran_optimizer.visualization.enhanced_map import generate_enhanced_map_from_files

    if not OUTPUT_DIR.exists() or not GIS_FILE.exists():
        pytest.skip("Output data not available")

    output_file = tmp_path / "from_files_map.html"

    m = generate_enhanced_map_from_files(
        output_dir=OUTPUT_DIR,
        gis_file=GIS_FILE,
        output_file=output_file,
        environment_aware=True,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 100 * 1024, "Map file seems too small"


def test_map_has_legend(gis_df, overshooting_df, tmp_path):
    """Test that map includes legend panel."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "legend_test.html"

    create_enhanced_map(
        overshooting_df=overshooting_df,
        gis_df=gis_df,
        output_file=output_file,
    )

    html_content = output_file.read_text()
    assert "Legend" in html_content, "Missing legend panel"
    assert "CRITICAL" in html_content, "Missing severity labels in legend"
    assert "URBAN" in html_content or "SUBURBAN" in html_content, "Missing environment labels"


def test_map_has_export_functionality(gis_df, overshooting_df, tmp_path):
    """Test that map includes export functionality."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "export_test.html"

    create_enhanced_map(
        overshooting_df=overshooting_df,
        gis_df=gis_df,
        output_file=output_file,
    )

    html_content = output_file.read_text()
    assert "Export Statistics" in html_content, "Missing export button"
    assert "exportData" in html_content, "Missing export JavaScript function"


def test_map_center_calculation(gis_df, overshooting_df, tmp_path):
    """Test that map centers correctly on data."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "center_test.html"

    m = create_enhanced_map(
        overshooting_df=overshooting_df,
        gis_df=gis_df,
        output_file=output_file,
    )

    # Map should be centered somewhere in Ireland
    center = m.location
    assert 51.0 <= center[0] <= 55.5, f"Latitude {center[0]} outside Ireland bounds"
    assert -11.0 <= center[1] <= -5.5, f"Longitude {center[1]} outside Ireland bounds"


def test_empty_data_handling(gis_df, tmp_path):
    """Test that map handles empty/missing data gracefully."""
    from ran_optimizer.visualization.enhanced_map import create_enhanced_map

    output_file = tmp_path / "empty_test.html"

    # Should not raise exception with empty/None data
    m = create_enhanced_map(
        overshooting_df=None,
        undershooting_df=None,
        gis_df=gis_df,
        no_coverage_gdf=None,
        low_coverage_gdfs=None,
        output_file=output_file,
    )

    assert output_file.exists()


@pytest.mark.slow
def test_full_dashboard_integration():
    """Full integration test generating the dashboard from real data."""
    from ran_optimizer.visualization.enhanced_map import generate_enhanced_map_from_files

    if not OUTPUT_DIR.exists() or not GIS_FILE.exists():
        pytest.skip("Real data not available")

    output_file = OUTPUT_DIR / "maps" / "enhanced_dashboard_test.html"

    m = generate_enhanced_map_from_files(
        output_dir=OUTPUT_DIR,
        gis_file=GIS_FILE,
        output_file=output_file,
    )

    assert output_file.exists()
    print(f"\nDashboard generated: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
