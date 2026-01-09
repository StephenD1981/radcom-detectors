"""
Test overshooting detection with preprocessed cell_coverage.csv data.

This file already has distances calculated and cells matched with grid bins.
"""
from pathlib import Path
import pandas as pd
from ran_optimizer.recommendations import OvershooterParams, OvershooterDetector
from ran_optimizer.utils.logging_config import configure_logging, get_logger

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def test_with_cell_coverage_data():
    """Test overshooting detection with preprocessed cell_coverage.csv."""
    coverage_file = Path("data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv")

    print("\n" + "="*80)
    print("OVERSHOOTING DETECTION WITH PREPROCESSED DATA")
    print("="*80)

    if not coverage_file.exists():
        print(f"❌ File not found: {coverage_file}")
        return

    # Load preprocessed data
    print(f"\n📂 Loading preprocessed data...")
    print(f"   File: {coverage_file}")

    # Load sample for testing
    df = pd.read_csv(coverage_file, nrows=100000)
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {len(df.columns)}")

    # Check what we have
    print(f"\n📊 Data Overview:")
    print(f"   Unique cells (cilac): {df['cilac'].nunique()}")
    print(f"   Unique grid bins: {df['grid'].nunique()}")
    print(f"   Distance column exists: {'distance_to_cell' in df.columns}")
    print(f"   Cell location exists: {'Latitude' in df.columns and 'Longitude' in df.columns}")

    # Check distance distribution
    print(f"\n📊 Distance Statistics:")
    print(f"   Min: {df['distance_to_cell'].min():.1f}m")
    print(f"   Max: {df['distance_to_cell'].max():.1f}m ({df['distance_to_cell'].max()/1000:.1f}km)")
    print(f"   Mean: {df['distance_to_cell'].mean():.1f}m ({df['distance_to_cell'].mean()/1000:.1f}km)")
    print(f"   Median: {df['distance_to_cell'].median():.1f}m")

    # Bins over 4km
    over_4km = df[df['distance_to_cell'] >= 4000]
    print(f"\n📊 Measurements Beyond 4km:")
    print(f"   Count: {len(over_4km):,}")
    print(f"   Percentage: {len(over_4km)/len(df)*100:.1f}%")
    print(f"   Cells serving >4km: {over_4km['cilac'].nunique()}")

    # Now format the data for our detector
    print(f"\n🔄 Preparing data for detector...")

    # Create grid DataFrame (measurements per grid-cell pair)
    grid_df = df[[
        'grid',  # geohash7
        'cilac',  # cell_id
        'avg_rsrp',  # rsrp
        'avg_rsrq',  # rsrq
        'event_count',  # total_traffic
        'distance_to_cell',  # distance_m (already calculated!)
    ]].copy()

    # Rename to match our schema
    grid_df = grid_df.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'avg_rsrp': 'rsrp',
        'avg_rsrq': 'rsrq',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Create GIS DataFrame (one row per cell)
    gis_df = df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'TiltM', 'TiltE']].drop_duplicates('cilac').copy()
    gis_df = gis_df.rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'TiltM': 'mechanical_tilt',
        'TiltE': 'electrical_tilt',
    })

    print(f"   Grid measurements: {len(grid_df)}")
    print(f"   GIS cells: {len(gis_df)}")

    # Test with default parameters
    print(f"\n⚙️  Running detection with DEFAULT parameters...")
    params = OvershooterParams()
    print(f"   Edge traffic: {params.edge_traffic_percent*100}%")
    print(f"   Min distance: {params.min_cell_distance}m")
    print(f"   Min overshooting grids: {params.min_overshooting_grids}")

    try:
        # Since we already have distances, we can modify the detector
        # or just add the distance column and use the detector
        detector = OvershooterDetector(params)

        # Call detect, but grid_df already has distance_m!
        # The detector will try to calculate it again, so let's skip that step
        # by calling the internal methods directly

        print(f"\n🔍 Step 1: Identify edge bins...")
        edge_bins = detector._identify_edge_bins(grid_df)
        print(f"   Edge bins found: {len(edge_bins)}")

        if len(edge_bins) == 0:
            print(f"   ⚠️  No edge bins found!")
            print(f"   This means the quantile logic may need adjustment")
            return

        print(f"\n🔍 Step 2: Calculate cell metrics...")
        cell_metrics = detector._calculate_cell_metrics(grid_df, edge_bins)
        print(f"   Cells analyzed: {len(cell_metrics)}")

        # Filter for cells beyond min distance
        candidates = cell_metrics[cell_metrics['max_distance_m'] >= params.min_cell_distance]
        print(f"   Cells beyond {params.min_cell_distance}m: {len(candidates)}")

        print(f"\n🔍 Step 3: Apply overshooting filters...")
        overshooters = detector._apply_overshooting_filters(cell_metrics, edge_bins)
        print(f"   Overshooters found: {len(overshooters)}")

        if len(overshooters) > 0:
            print(f"\n🔍 Step 4: Calculate tilt recommendations...")
            overshooters = detector._calculate_tilt_recommendations(overshooters, gis_df)

            print(f"\n✅ RESULTS:")
            print(f"   Total overshooters: {len(overshooters)}")
            print(f"   Percentage of cells: {len(overshooters)/len(gis_df)*100:.1f}%")

            print(f"\n📋 Top 10 Overshooters:")
            top10 = overshooters.nlargest(10, 'overshooting_grids')
            display_cols = [
                'cell_id',
                'overshooting_grids',
                'total_grids',
                'percentage_overshooting',
                'max_distance_m',
                'recommended_tilt_change',
            ]
            avail_cols = [c for c in display_cols if c in top10.columns]
            print(top10[avail_cols].to_string(index=False))

            # Statistics
            print(f"\n📊 Overshooting Statistics:")
            print(f"   Avg overshooting grids: {overshooters['overshooting_grids'].mean():.1f}")
            print(f"   Max overshooting grids: {overshooters['overshooting_grids'].max()}")
            print(f"   Avg max distance: {overshooters['max_distance_m'].mean()/1000:.1f} km")
            print(f"   Avg recommended tilt: +{overshooters['recommended_tilt_change'].mean():.1f}°")

        else:
            print(f"\n⚠️  No overshooters detected with current parameters")
            print(f"   Try adjusting parameters to be more lenient")

    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    return overshooters if 'overshooters' in locals() and len(overshooters) > 0 else None


if __name__ == "__main__":
    print("\n🧪 OVERSHOOTING DETECTION - PREPROCESSED DATA TEST")
    print("="*80)

    result = test_with_cell_coverage_data()

    print("\n" + "="*80)
    if result is not None:
        print("✅ TEST COMPLETE - OVERSHOOTERS FOUND")
    else:
        print("✅ TEST COMPLETE - NO OVERSHOOTERS")
    print("="*80)
