"""
Integration test for overshooting detection with real VF Ireland data.

Tests the complete pipeline: load data -> detect overshooters -> generate recommendations
"""
from pathlib import Path
import pandas as pd
from ran_optimizer.data.loaders import load_grid_data, load_gis_data
from ran_optimizer.recommendations import (
    OvershooterDetector,
    OvershooterParams,
    detect_overshooting_cells,
)
from ran_optimizer.utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def test_vf_overshooting_detection():
    """Test overshooting detection on VF Ireland data."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("VF IRELAND OVERSHOOTING DETECTION TEST")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print(f"❌ Data files not found:")
        print(f"   Grid: {grid_file.exists()}")
        print(f"   GIS: {gis_file.exists()}")
        return

    # Step 1: Load data
    print(f"\n📂 Loading VF Ireland data...")
    print(f"   Grid: {grid_file}")
    print(f"   GIS: {gis_file}")

    # Load GIS data (small, load all)
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    print(f"   ✅ Loaded {len(gis_df)} cells")

    # Load grid data (large, load sample) with geohash decoding
    # NOTE: For full analysis, would load all ~3.4M rows
    # For testing, use sample
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=100000,
        decode_geohash=True,
        validate=False,  # Skip validation for speed
    )
    print(f"   ✅ Loaded {len(grid_df)} grid measurements (sample)")

    # Verify lat/lon added
    if 'latitude' not in grid_df.columns or 'longitude' not in grid_df.columns:
        print("⚠️  Grid data lacks lat/lon after loading")
        print("   This shouldn't happen with decode_geohash=True")
        return

    # Step 2: Configure parameters
    print(f"\n⚙️  Configuring detection parameters...")
    params = OvershooterParams(
        edge_traffic_percent=0.15,
        min_cell_distance=4000,  # 4 km
        percent_max_distance=0.7,
        min_cell_count_in_grid=3,
        max_percentage_grid_events=0.25,
        rsrp_degradation_db=10.0,
        min_overshooting_grids=30,
        percentage_overshooting_grids=0.05,
    )
    print(f"   Edge traffic threshold: {params.edge_traffic_percent*100}%")
    print(f"   Min cell distance: {params.min_cell_distance}m")
    print(f"   Min overshooting grids: {params.min_overshooting_grids}")

    # Step 3: Run detection
    print(f"\n🔍 Running overshooting detection...")
    try:
        overshooters = detect_overshooting_cells(grid_df, gis_df, params)
        print(f"   ✅ Detection complete")
        print(f"   Found {len(overshooters)} overshooting cells")
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Analyze results
    if len(overshooters) > 0:
        print(f"\n📊 Overshooting Cell Summary:")
        print(f"   Total cells analyzed: {len(gis_df)}")
        print(f"   Cells flagged: {len(overshooters)}")
        print(f"   Percentage: {len(overshooters)/len(gis_df)*100:.1f}%")

        print(f"\n📋 Top 10 Overshooters (by overshooting_grids):")
        top_10 = overshooters.nlargest(10, 'overshooting_grids')
        display_cols = [
            'cell_id',
            'overshooting_grids',
            'total_grids',
            'percentage_overshooting',
            'max_distance_m',
            'avg_edge_rsrp',
            'mechanical_tilt',
            'recommended_tilt_change'
        ]
        # Filter to columns that exist
        available_cols = [c for c in display_cols if c in top_10.columns]
        print(top_10[available_cols].to_string(index=False))

        print(f"\n📈 Statistics:")
        print(f"   Avg overshooting grids: {overshooters['overshooting_grids'].mean():.1f}")
        print(f"   Max overshooting grids: {overshooters['overshooting_grids'].max()}")
        print(f"   Avg recommended tilt change: {overshooters['recommended_tilt_change'].mean():.1f}°")
        print(f"   Max recommended tilt change: {overshooters['recommended_tilt_change'].max():.1f}°")

        # Step 5: Sample recommendations
        print(f"\n💡 Sample Tilt Recommendations:")
        sample = overshooters.head(3)
        for idx, row in sample.iterrows():
            print(f"\n   Cell: {row['cell_id']}")
            print(f"   Current mechanical tilt: {row.get('mechanical_tilt', 'N/A')}°")
            print(f"   Current electrical tilt: {row.get('electrical_tilt', 'N/A')}°")
            print(f"   Overshooting bins: {row['overshooting_grids']} ({row['percentage_overshooting']*100:.1f}%)")
            print(f"   Max serving distance: {row['max_distance_m']/1000:.1f} km")
            print(f"   ➜ Recommended tilt increase: +{row['recommended_tilt_change']:.1f}°")

    else:
        print(f"\n✅ No overshooting cells detected with current parameters")
        print(f"   This could mean:")
        print(f"   - Network is well-optimized")
        print(f"   - Parameters are too strict")
        print(f"   - Sample data too small")

    return overshooters


def test_vf_overshooting_sensitivity():
    """Test how parameter changes affect detection results."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print(f"❌ Data files not found")
        return

    # Load data (small sample for speed)
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=50000,
        decode_geohash=True,
        validate=False,
    )

    if 'latitude' not in grid_df.columns or 'longitude' not in grid_df.columns:
        print("⚠️  Grid data lacks lat/lon - skipping test")
        return

    # Test different parameter sets
    param_sets = [
        ("Strict", OvershooterParams(
            min_cell_distance=6000,
            min_overshooting_grids=50,
            percentage_overshooting_grids=0.10,
        )),
        ("Default", OvershooterParams()),
        ("Lenient", OvershooterParams(
            min_cell_distance=2000,
            min_overshooting_grids=10,
            percentage_overshooting_grids=0.02,
        )),
    ]

    print(f"\n📊 Testing {len(param_sets)} parameter configurations:")
    results = []

    for name, params in param_sets:
        try:
            overshooters = detect_overshooting_cells(grid_df, gis_df, params)
            count = len(overshooters)
            pct = count / len(gis_df) * 100 if len(gis_df) > 0 else 0

            results.append({
                'Configuration': name,
                'Cells_Flagged': count,
                'Percentage': f"{pct:.1f}%",
                'Min_Distance': params.min_cell_distance,
                'Min_Grids': params.min_overshooting_grids,
            })

            print(f"   {name:10} -> {count:3} cells ({pct:4.1f}%)")
        except Exception as e:
            print(f"   {name:10} -> Error: {e}")

    print(f"\n📋 Detailed Comparison:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    print("\n🧪 VF IRELAND OVERSHOOTING DETECTION TESTS")
    print("="*80)

    # Run tests
    overshooters = test_vf_overshooting_detection()
    test_vf_overshooting_sensitivity()

    print("\n" + "="*80)
    print("✅ TESTS COMPLETE")
    print("="*80)
