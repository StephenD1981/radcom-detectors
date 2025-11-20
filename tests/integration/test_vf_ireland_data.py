"""
Integration test for loading Vodafone Ireland data.

Tests the data loading system with real VF Ireland Cork data.
"""
from pathlib import Path
import pandas as pd
from ran_optimizer.utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def test_inspect_vf_grid_data():
    """Inspect VF Ireland grid data structure."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")

    logger.info("inspecting_vf_grid_data", file=str(grid_file))

    if not grid_file.exists():
        logger.error("file_not_found", file=str(grid_file))
        return

    # Load a sample
    df = pd.read_csv(grid_file, nrows=5)

    logger.info("grid_data_sample",
                rows=len(df),
                columns=list(df.columns))

    print("\n" + "="*80)
    print("VF IRELAND GRID DATA STRUCTURE")
    print("="*80)
    print(f"\nFile: {grid_file}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    print(f"\nFirst 2 rows:")
    print(df.head(2).to_string())

    # Check for required fields
    print("\n" + "-"*80)
    print("FIELD MAPPING ANALYSIS:")
    print("-"*80)

    required_fields = {
        'geohash7': 'grid' if 'grid' in df.columns else 'NOT FOUND',
        'rsrp': 'avg_rsrp' if 'avg_rsrp' in df.columns else 'NOT FOUND',
        'rsrq': 'avg_rsrq' if 'avg_rsrq' in df.columns else 'NOT FOUND',
        'cell_id': 'global_cell_id' if 'global_cell_id' in df.columns else 'NOT FOUND',
        'cell_pci': 'NOT FOUND (need to check)',
    }

    for schema_field, data_field in required_fields.items():
        status = "✅" if data_field != 'NOT FOUND' else "❌"
        print(f"{status} {schema_field:20} -> {data_field}")

    return df


def test_inspect_vf_gis_data():
    """Inspect VF Ireland GIS data structure."""
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    logger.info("inspecting_vf_gis_data", file=str(gis_file))

    if not gis_file.exists():
        logger.error("file_not_found", file=str(gis_file))
        return

    # Load a sample
    df = pd.read_csv(gis_file, nrows=5)

    logger.info("gis_data_sample",
                rows=len(df),
                columns=list(df.columns))

    print("\n" + "="*80)
    print("VF IRELAND GIS DATA STRUCTURE")
    print("="*80)
    print(f"\nFile: {gis_file}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    print(f"\nFirst 2 rows:")
    print(df.head(2).to_string())

    # Check for required fields
    print("\n" + "-"*80)
    print("FIELD MAPPING ANALYSIS:")
    print("-"*80)

    required_fields = {
        'cell_id': 'CILAC' if 'CILAC' in df.columns else 'SectorID',
        'site_name': 'SiteID' if 'SiteID' in df.columns else 'NOT FOUND',
        'latitude': 'Latitude' if 'Latitude' in df.columns else 'NOT FOUND',
        'longitude': 'Longitude' if 'Longitude' in df.columns else 'NOT FOUND',
        'height_m': 'Height' if 'Height' in df.columns else 'NOT FOUND',
        'azimuth_deg': 'Bearing' if 'Bearing' in df.columns else 'NOT FOUND',
        'mechanical_tilt': 'TiltM' if 'TiltM' in df.columns else 'NOT FOUND',
        'electrical_tilt': 'TiltE' if 'TiltE' in df.columns else 'NOT FOUND',
        'on_air': 'AdminCellState' if 'AdminCellState' in df.columns else 'NOT FOUND',
    }

    for schema_field, data_field in required_fields.items():
        status = "✅" if data_field != 'NOT FOUND' else "❌"
        print(f"{status} {schema_field:20} -> {data_field}")

    # Show sample values for tilt fields
    if 'TiltE' in df.columns and 'TiltM' in df.columns:
        print(f"\nSample tilt values:")
        print(f"  TiltE (electrical): {df['TiltE'].head(3).tolist()}")
        print(f"  TiltM (mechanical): {df['TiltM'].head(3).tolist()}")

    return df


def test_data_statistics():
    """Show basic statistics about the VF Ireland data."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)

    if grid_file.exists():
        # Count rows without loading all into memory
        import subprocess
        result = subprocess.run(
            ['wc', '-l', str(grid_file)],
            capture_output=True,
            text=True
        )
        row_count = result.stdout.split()[0]
        print(f"\n📊 Grid Data: ~{row_count} rows")

        # Load sample to check for nulls
        df_grid = pd.read_csv(grid_file, nrows=1000)
        print(f"   Sample RSRP range: {df_grid['avg_rsrp'].min():.1f} to {df_grid['avg_rsrp'].max():.1f} dBm")
        print(f"   Sample RSRQ range: {df_grid['avg_rsrq'].min():.1f} to {df_grid['avg_rsrq'].max():.1f} dB")
        print(f"   Unique cells in sample: {df_grid['global_cell_id'].nunique()}")

    if gis_file.exists():
        df_gis = pd.read_csv(gis_file)
        print(f"\n📍 GIS Data: {len(df_gis)} cells")
        print(f"   Unique sites: {df_gis['SiteID'].nunique()}")
        if 'AdminCellState' in df_gis.columns:
            print(f"   On-air cells: {(df_gis['AdminCellState'] == 1).sum()}")
        if 'TiltE' in df_gis.columns:
            print(f"   Electrical tilt range: {df_gis['TiltE'].min():.1f}° to {df_gis['TiltE'].max():.1f}°")
        if 'TiltM' in df_gis.columns:
            print(f"   Mechanical tilt range: {df_gis['TiltM'].min():.1f}° to {df_gis['TiltM'].max():.1f}°")


if __name__ == "__main__":
    print("\n🔍 VODAFONE IRELAND DATA INSPECTION")
    print("="*80)

    # Inspect both datasets
    test_inspect_vf_grid_data()
    test_inspect_vf_gis_data()
    test_data_statistics()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Create column mapping adapter for VF Ireland data format")
    print("2. Update schemas to handle different column names")
    print("3. Add data transformation layer in loaders")
    print("="*80)
