"""
Test enhanced metrics for overshooting and undershooting detection.

Verifies that both algorithms output the requested metrics:
- Overshooting: Interference reduction metrics
- Undershooting: Coverage expansion metrics
"""
from pathlib import Path
import pandas as pd
from ran_optimizer.recommendations import (
    OvershooterDetector,
    OvershooterParams,
    UndershooterDetector,
    UndershooterParams,
)
from ran_optimizer.utils.logging_config import configure_logging, get_logger

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def test_overshooting_enhanced_metrics():
    """Test overshooting detection with enhanced interference reduction metrics."""
    coverage_file = Path("data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv")

    print("\n" + "="*80)
    print("OVERSHOOTING - ENHANCED INTERFERENCE REDUCTION METRICS TEST")
    print("="*80)

    if not coverage_file.exists():
        print(f"❌ File not found: {coverage_file}")
        return None

    # Load data (small sample for quick test)
    print(f"\n📂 Loading data...")
    df = pd.read_csv(coverage_file, nrows=50000)
    print(f"   ✅ Loaded {len(df):,} rows")

    # Prepare grid DataFrame
    grid_df = df[[
        'grid', 'cilac', 'Band', 'avg_rsrp', 'avg_rsrq',
        'event_count', 'distance_to_cell',
    ]].copy()

    grid_df = grid_df.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'Band': 'Band',
        'avg_rsrp': 'rsrp',
        'avg_rsrq': 'rsrq',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Prepare GIS DataFrame
    gis_df = df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'TiltM', 'TiltE']].drop_duplicates('cilac').copy()
    gis_df = gis_df.rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'TiltM': 'mechanical_tilt',
        'TiltE': 'electrical_tilt',
    })

    # Run detection
    print(f"\n⚙️  Running overshooting detection...")
    params = OvershooterParams()
    detector = OvershooterDetector(params)
    overshooters = detector.detect(grid_df, gis_df)

    if len(overshooters) == 0:
        print(f"\n⚠️  No overshooters found in sample")
        return None

    print(f"\n✅ Found {len(overshooters)} overshooting cells")

    # Verify enhanced metrics exist
    required_cols = [
        'current_interference_grids',
        'current_interference_pct',
        'removed_interference_grids',
        'new_interference_grids',
        'new_interference_pct',
        'interference_reduction_pct',
        'recommended_tilt_change',
    ]

    missing_cols = [col for col in required_cols if col not in overshooters.columns]
    if missing_cols:
        print(f"\n❌ Missing columns: {missing_cols}")
        return None

    print(f"\n✅ All enhanced metrics present")

    # Display sample results
    print(f"\n📊 SAMPLE RESULTS (Top 5 by severity):")
    print("="*80)

    display_cols = [
        'cell_id',
        'recommended_tilt_change',
        'current_interference_grids',
        'current_interference_pct',
        'removed_interference_grids',
        'new_interference_grids',
        'new_interference_pct',
        'interference_reduction_pct',
    ]

    sample = overshooters[display_cols].head(5).copy()
    sample['current_interference_pct'] = (sample['current_interference_pct'] * 100).round(1)
    sample['new_interference_pct'] = (sample['new_interference_pct'] * 100).round(1)
    sample['interference_reduction_pct'] = sample['interference_reduction_pct'].round(1)

    print(sample.to_string(index=False))

    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("="*80)
    print(f"   Avg interference reduction: {overshooters['interference_reduction_pct'].mean():.1f}%")
    print(f"   Max interference reduction: {overshooters['interference_reduction_pct'].max():.1f}%")
    print(f"   Min interference reduction: {overshooters['interference_reduction_pct'].min():.1f}%")
    print(f"   Avg grids removed: {overshooters['removed_interference_grids'].mean():.1f}")
    print(f"   Total grids to be resolved: {overshooters['removed_interference_grids'].sum():,}")

    return overshooters


def test_undershooting_enhanced_metrics():
    """Test undershooting detection with enhanced coverage expansion metrics."""
    coverage_file = Path("data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv")

    print("\n" + "="*80)
    print("UNDERSHOOTING - ENHANCED COVERAGE EXPANSION METRICS TEST")
    print("="*80)

    if not coverage_file.exists():
        print(f"❌ File not found: {coverage_file}")
        return None

    # Load data (small sample for quick test)
    print(f"\n📂 Loading data...")
    df = pd.read_csv(coverage_file, nrows=50000)
    print(f"   ✅ Loaded {len(df):,} rows")

    # Prepare grid DataFrame
    grid_df = df[[
        'grid', 'cilac', 'avg_rsrp', 'event_count', 'distance_to_cell',
    ]].copy()

    grid_df = grid_df.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'avg_rsrp': 'rsrp',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Prepare GIS DataFrame
    gis_df = df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'TiltM', 'TiltE', 'Height']].drop_duplicates('cilac').copy()
    gis_df = gis_df.rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'TiltM': 'mechanical_tilt',
        'TiltE': 'electrical_tilt',
        'Height': 'height',
    })

    # Run detection
    print(f"\n⚙️  Running undershooting detection...")
    params = UndershooterParams()
    detector = UndershooterDetector(params)
    undershooters = detector.detect(grid_df, gis_df)

    if len(undershooters) == 0:
        print(f"\n⚠️  No undershooters found in sample")
        return None

    print(f"\n✅ Found {len(undershooters)} undershooting cells")

    # Verify enhanced metrics exist
    required_cols = [
        'current_coverage_grids',
        'current_distance_m',
        'distance_increase_m',
        'new_coverage_grids',
        'total_coverage_after_uptilt',
        'recommended_uptilt_deg',
        'new_max_distance_m',
        'coverage_increase_percentage',
    ]

    missing_cols = [col for col in required_cols if col not in undershooters.columns]
    if missing_cols:
        print(f"\n❌ Missing columns: {missing_cols}")
        return None

    print(f"\n✅ All enhanced metrics present")

    # Display sample results
    print(f"\n📊 SAMPLE RESULTS (Top 5 by coverage increase):")
    print("="*80)

    display_cols = [
        'cell_id',
        'recommended_uptilt_deg',
        'current_coverage_grids',
        'new_coverage_grids',
        'total_coverage_after_uptilt',
        'current_distance_m',
        'distance_increase_m',
        'new_max_distance_m',
        'coverage_increase_percentage',
    ]

    sample = undershooters[display_cols].head(5).copy()
    sample['current_distance_m'] = (sample['current_distance_m'] / 1000).round(2)
    sample['distance_increase_m'] = (sample['distance_increase_m']).round(0).astype(int)
    sample['new_max_distance_m'] = (sample['new_max_distance_m'] / 1000).round(2)
    sample['coverage_increase_percentage'] = (sample['coverage_increase_percentage'] * 100).round(1)

    # Rename for display
    sample = sample.rename(columns={
        'current_distance_m': 'current_dist_km',
        'new_max_distance_m': 'new_dist_km',
        'coverage_increase_percentage': 'coverage_inc_%',
    })

    print(sample.to_string(index=False))

    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("="*80)
    print(f"   Avg coverage increase: {undershooters['coverage_increase_percentage'].mean()*100:.1f}%")
    print(f"   Max coverage increase: {undershooters['coverage_increase_percentage'].max()*100:.1f}%")
    print(f"   Avg distance gain: {undershooters['distance_increase_m'].mean():.0f}m")
    print(f"   Max distance gain: {undershooters['distance_increase_m'].max():.0f}m")
    print(f"   Avg new grids: {undershooters['new_coverage_grids'].mean():.1f}")
    print(f"   Total new grids: {undershooters['new_coverage_grids'].sum():,}")

    return undershooters


if __name__ == "__main__":
    print("\n🚀 TESTING ENHANCED METRICS")
    print("="*80)
    print("Verifying interference reduction and coverage expansion metrics")
    print()

    # Test overshooting
    overshooters = test_overshooting_enhanced_metrics()

    # Test undershooting
    undershooters = test_undershooting_enhanced_metrics()

    # Summary
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)

    if overshooters is not None:
        print(f"   Overshooting: {len(overshooters)} cells with interference reduction metrics")
    else:
        print(f"   Overshooting: Test failed or no cells found")

    if undershooters is not None:
        print(f"   Undershooting: {len(undershooters)} cells with coverage expansion metrics")
    else:
        print(f"   Undershooting: Test failed or no cells found")

    print()
