"""
Run undershooting detection on FULL cell_coverage.csv dataset.

This will process all measurements to identify all undershooting cells.
"""
from pathlib import Path
import pandas as pd
import time
from ran_optimizer.recommendations import UndershooterParams, UndershooterDetector
from ran_optimizer.utils.logging_config import configure_logging, get_logger
from ran_optimizer.utils.dtypes import load_cell_coverage_csv

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def run_full_detection():
    """Run undershooting detection on complete dataset."""
    coverage_file = Path("data/vf-ie/input-data/cell_coverage.csv")

    print("\n" + "="*80)
    print("FULL DATASET UNDERSHOOTING DETECTION - VF IRELAND")
    print("="*80)

    if not coverage_file.exists():
        print(f"❌ File not found: {coverage_file}")
        return

    # Load FULL dataset
    print(f"\n📂 Loading FULL dataset...")
    print(f"   File: {coverage_file}")
    start_time = time.time()

    df = load_cell_coverage_csv(coverage_file)
    load_time = time.time() - start_time

    print(f"   ✅ Loaded {len(df):,} rows in {load_time:.1f}s")

    # Prepare data for detector
    print(f"\n🔄 Preparing data for detector...")
    start_time = time.time()

    # Create grid DataFrame
    grid_df = df[[
        'grid',
        'cilac',
        'avg_rsrp',
        'event_count',
        'distance_to_cell',
    ]].copy()

    grid_df = grid_df.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'avg_rsrp': 'rsrp',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Create GIS DataFrame
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

    prep_time = time.time() - start_time
    print(f"   ✅ Data prepared in {prep_time:.1f}s")

    # Run detection with DEFAULT parameters
    print(f"\n⚙️  Running detection with DEFAULT parameters...")
    params = UndershooterParams()
    print(f"   Max cell distance: {params.max_cell_distance}m ({params.max_cell_distance/1000}km)")
    print(f"   Min cell event count: {params.min_cell_event_count}")
    print(f"   Max interference percentage: {params.max_interference_percentage*100}%")

    detector = UndershooterDetector(params)

    start_time = time.time()
    undershooters = detector.detect(grid_df, gis_df)
    detection_time = time.time() - start_time

    print(f"   ✅ Detection complete in {detection_time:.1f}s")

    if len(undershooters) == 0:
        print(f"\n⚠️  No undershooting cells detected with current parameters")
        return None

    # Results
    print(f"\n" + "="*80)
    print(f"✅ UNDERSHOOTING DETECTION RESULTS - FULL DATASET")
    print(f"="*80)

    print(f"\n📊 Summary Statistics:")
    print(f"   Total cells in network: {len(gis_df):,}")
    print(f"   Undershooting cells detected: {len(undershooters):,}")
    print(f"   Percentage undershooting: {len(undershooters)/len(gis_df)*100:.1f}%")

    print(f"\n📊 Coverage Expansion Metrics:")
    print(f"   Avg coverage increase: {undershooters['coverage_increase_percentage'].mean()*100:.1f}%")
    print(f"   Max coverage increase: {undershooters['coverage_increase_percentage'].max()*100:.1f}%")
    print(f"   Avg distance gain: {undershooters['distance_increase_m'].mean():.0f}m")
    print(f"   Max distance gain: {undershooters['distance_increase_m'].max():.0f}m")
    print(f"   Total new grids: {undershooters['new_coverage_grids'].sum():,}")

    # Top 20 best opportunities (by coverage increase)
    print(f"\n📋 TOP 20 BEST OPPORTUNITIES (by coverage increase):")
    print(f"="*80)
    top20 = undershooters.nlargest(20, 'coverage_increase_percentage')

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

    # Format for display
    display_df = top20[display_cols].copy()
    display_df['current_distance_m'] = (display_df['current_distance_m'] / 1000).round(2)
    display_df['distance_increase_m'] = display_df['distance_increase_m'].round(0).astype(int)
    display_df['new_max_distance_m'] = (display_df['new_max_distance_m'] / 1000).round(2)
    display_df['coverage_increase_percentage'] = (display_df['coverage_increase_percentage'] * 100).round(1)

    # Rename for display
    display_df = display_df.rename(columns={
        'current_distance_m': 'current_km',
        'new_max_distance_m': 'new_km',
        'coverage_increase_percentage': 'cov_inc_%',
        'distance_increase_m': 'dist_gain_m',
        'current_coverage_grids': 'cur_grids',
        'new_coverage_grids': 'new_grids',
        'total_coverage_after_uptilt': 'total_after',
    })

    print(display_df.to_string(index=False))

    # Save results
    output_file = Path("data/vf-ie/output-data/undershooting_cells.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving results...")
    undershooters.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   Rows: {len(undershooters):,}")
    print(f"   Columns: {len(undershooters.columns)}")

    return undershooters


if __name__ == "__main__":
    print("\n🚀 FULL DATASET UNDERSHOOTING DETECTION")
    print("="*80)
    print("Processing all VF Ireland cell coverage data...")
    print()

    start = time.time()
    result = run_full_detection()
    total_time = time.time() - start

    print(f"\n" + "="*80)
    print(f"✅ COMPLETE")
    print(f"="*80)
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if result is not None:
        print(f"Undershooting cells identified: {len(result):,}")
        print(f"Results saved to: data/vf-ie/output-data/undershooting_cells.csv")

    print()
