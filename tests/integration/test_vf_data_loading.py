"""
Integration test for loading and validating Vodafone Ireland data.

Tests the complete pipeline: raw data -> adapter -> validation -> schemas
"""
from pathlib import Path
import pandas as pd
from ran_optimizer.data.adapters import VodafoneIrelandAdapter
from ran_optimizer.data.schemas import GridMeasurement, CellGIS
from ran_optimizer.data.loaders import _validate_dataframe, get_data_summary
from ran_optimizer.utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def test_load_and_validate_vf_gis():
    """Test loading VF Ireland GIS data with adapter and validation."""
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("TEST 1: VF IRELAND GIS DATA LOADING & VALIDATION")
    print("="*80)

    if not gis_file.exists():
        print(f"❌ File not found: {gis_file}")
        return

    # Load raw data
    print(f"\n📂 Loading: {gis_file}")
    df_raw = pd.read_csv(gis_file)
    print(f"   Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Apply adapter
    print(f"\n🔄 Applying VodafoneIrelandAdapter...")
    df_adapted = VodafoneIrelandAdapter.adapt_gis_data(df_raw)
    print(f"   Adapted columns: {', '.join([c for c in df_adapted.columns if c in CellGIS.model_fields])}")

    # Validate sample
    print(f"\n✅ Validating sample (first 100 rows)...")
    df_sample = df_adapted.head(100)
    validated_df, validation_errors = _validate_dataframe(df_sample, CellGIS)

    print(f"   Valid rows: {len(validated_df)}/{len(df_sample)}")
    print(f"   Validation errors: {len(validation_errors)}")

    if validation_errors:
        print(f"\n⚠️  Sample errors (first 3):")
        for i, error in enumerate(validation_errors[:3]):
            print(f"   {i+1}. Row {error['row_index']}: {error['errors'][0]['msg']}")

    # Show summary
    summary = get_data_summary(validated_df, "gis")
    print(f"\n📊 Data Summary:")
    print(f"   Total cells: {summary['total_rows']}")
    print(f"   Unique sites: {summary['unique_sites']}")
    print(f"   On-air cells: {summary['on_air_cells']}")
    print(f"   Avg mechanical tilt: {summary['avg_mechanical_tilt']:.1f}°")
    print(f"   Avg height: {summary['avg_height']:.1f}m")

    # Show sample validated cells
    print(f"\n📋 Sample validated cells:")
    print(validated_df[['cell_id', 'site_name', 'latitude', 'longitude', 'azimuth_deg', 'mechanical_tilt', 'electrical_tilt']].head(3).to_string(index=False))

    return validated_df, validation_errors


def test_load_and_validate_vf_grid():
    """Test loading VF Ireland grid data with adapter and validation."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")

    print("\n" + "="*80)
    print("TEST 2: VF IRELAND GRID DATA LOADING & VALIDATION")
    print("="*80)

    if not grid_file.exists():
        print(f"❌ File not found: {grid_file}")
        return

    # Load sample (grid data is large)
    print(f"\n📂 Loading sample: {grid_file}")
    df_raw = pd.read_csv(grid_file, nrows=1000)
    print(f"   Sample data: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Apply adapter
    print(f"\n🔄 Applying VodafoneIrelandAdapter...")
    df_adapted = VodafoneIrelandAdapter.adapt_grid_data(df_raw)
    print(f"   Adapted columns: {', '.join([c for c in df_adapted.columns if c in GridMeasurement.model_fields])}")

    # Validate
    print(f"\n✅ Validating...")
    validated_df, validation_errors = _validate_dataframe(df_adapted, GridMeasurement)

    print(f"   Valid rows: {len(validated_df)}/{len(df_adapted)}")
    print(f"   Validation errors: {len(validation_errors)}")
    print(f"   Error rate: {len(validation_errors)/len(df_adapted)*100:.1f}%")

    if validation_errors:
        print(f"\n⚠️  Sample errors (first 3):")
        for i, error in enumerate(validation_errors[:3]):
            print(f"   {i+1}. Row {error['row_index']}: {error['errors'][0]['msg']}")

    # Show summary
    summary = get_data_summary(validated_df, "grid")
    print(f"\n📊 Data Summary:")
    print(f"   Total measurements: {summary['total_rows']}")
    print(f"   Unique cells: {summary['unique_cells']}")
    print(f"   Unique geohashes: {summary['unique_geohashes']}")
    if 'rsrp_min' in summary:
        print(f"   RSRP range: {summary['rsrp_min']:.1f} to {summary['rsrp_max']:.1f} dBm")
        print(f"   Mean RSRP: {summary['rsrp_mean']:.1f} dBm")

    # Show sample measurements
    print(f"\n📋 Sample validated measurements:")
    print(validated_df[['geohash7', 'cell_id', 'rsrp', 'rsrq', 'total_traffic']].head(3).to_string(index=False))

    return validated_df, validation_errors


def test_validation_thresholds():
    """Test if VF Ireland data meets validation thresholds."""
    print("\n" + "="*80)
    print("TEST 3: VALIDATION THRESHOLD CHECK")
    print("="*80)

    # Test with larger sample
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    df_raw = pd.read_csv(grid_file, nrows=10000)
    df_adapted = VodafoneIrelandAdapter.adapt_grid_data(df_raw)
    validated_df, validation_errors = _validate_dataframe(df_adapted, GridMeasurement)

    error_rate = len(validation_errors) / len(df_adapted)

    print(f"\n📊 Large Sample Test (10,000 rows):")
    print(f"   Valid rows: {len(validated_df)}")
    print(f"   Invalid rows: {len(validation_errors)}")
    print(f"   Error rate: {error_rate*100:.2f}%")
    print(f"   Threshold: 10%")

    if error_rate <= 0.10:
        print(f"   ✅ PASS - Error rate below threshold")
    else:
        print(f"   ❌ FAIL - Error rate exceeds threshold")

    return error_rate


if __name__ == "__main__":
    print("\n🧪 VF IRELAND DATA LOADING INTEGRATION TESTS")
    print("="*80)

    # Run tests
    test_load_and_validate_vf_gis()
    test_load_and_validate_vf_grid()
    test_validation_thresholds()

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80)
