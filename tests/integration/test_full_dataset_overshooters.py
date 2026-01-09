"""
Run overshooting detection on FULL cell_coverage.csv dataset.

This will process all measurements to identify all overshooting cells.
"""
from pathlib import Path
import pandas as pd
import time
from ran_optimizer.recommendations import OvershooterParams, OvershooterDetector
from ran_optimizer.utils.logging_config import configure_logging, get_logger
from ran_optimizer.utils.dtypes import load_cell_coverage_csv

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def run_full_detection():
    """Run overshooting detection on complete dataset."""
    coverage_file = Path("data/vf-ie/input-data/cell_coverage.csv")

    print("\n" + "="*80)
    print("FULL DATASET OVERSHOOTING DETECTION - VF IRELAND")
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
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: ~{coverage_file.stat().st_size / (1024**2):.1f} MB")

    # Data overview
    print(f"\n📊 Full Dataset Overview:")
    print(f"   Total measurements: {len(df):,}")
    print(f"   Unique cells (cilac): {df['cilac'].nunique():,}")
    print(f"   Unique grid bins: {df['grid'].nunique():,}")

    # Distance statistics
    print(f"\n📊 Distance Statistics:")
    print(f"   Min: {df['distance_to_cell'].min():.1f}m")
    print(f"   Max: {df['distance_to_cell'].max():.1f}m ({df['distance_to_cell'].max()/1000:.1f}km)")
    print(f"   Mean: {df['distance_to_cell'].mean():.1f}m ({df['distance_to_cell'].mean()/1000:.1f}km)")
    print(f"   Median: {df['distance_to_cell'].median():.1f}m")

    # Bins over various thresholds
    print(f"\n📊 Distance Distribution:")
    for threshold_km in [2, 4, 6, 8, 10, 15, 20]:
        threshold_m = threshold_km * 1000
        over_threshold = df[df['distance_to_cell'] >= threshold_m]
        pct = len(over_threshold) / len(df) * 100
        cells = over_threshold['cilac'].nunique()
        print(f"   >{threshold_km:2d}km: {len(over_threshold):7,} measurements ({pct:5.1f}%) - {cells:4,} cells")

    # Prepare data for detector
    print(f"\n🔄 Preparing data for detector...")
    start_time = time.time()

    # Create grid DataFrame (including Band for frequency-aware competition)
    grid_df = df[[
        'grid',
        'cilac',
        'Band',  # CRITICAL: Include frequency band
        'avg_rsrp',
        'avg_rsrq',
        'event_count',
        'distance_to_cell',
    ]].copy()

    grid_df = grid_df.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'Band': 'Band',  # Keep as 'Band'
        'avg_rsrp': 'rsrp',
        'avg_rsrq': 'rsrq',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Create GIS DataFrame
    gis_df = df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'TiltM', 'TiltE']].drop_duplicates('cilac').copy()
    gis_df = gis_df.rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'TiltM': 'mechanical_tilt',
        'TiltE': 'electrical_tilt',
    })

    prep_time = time.time() - start_time
    print(f"   ✅ Data prepared in {prep_time:.1f}s")
    print(f"   Grid measurements: {len(grid_df):,}")
    print(f"   GIS cells: {len(gis_df):,}")
    print(f"   Frequency bands: {sorted(grid_df['Band'].unique())} ({grid_df['Band'].nunique()} bands)")

    # Run detection with DEFAULT parameters
    print(f"\n⚙️  Running detection with DEFAULT parameters...")
    params = OvershooterParams()
    print(f"   Edge traffic threshold: {params.edge_traffic_percent*100}%")
    print(f"   Min cell distance: {params.min_cell_distance}m ({params.min_cell_distance/1000}km)")
    print(f"   Min overshooting grids: {params.min_overshooting_grids}")
    print(f"   Min percentage overshooting: {params.percentage_overshooting_grids*100}%")

    detector = OvershooterDetector(params)

    # Step 1: Edge bins
    print(f"\n🔍 Step 1: Identifying edge bins...")
    start_time = time.time()
    edge_bins = detector._identify_edge_bins(grid_df)
    step1_time = time.time() - start_time
    print(f"   ✅ Edge bins identified: {len(edge_bins):,} ({len(edge_bins)/len(grid_df)*100:.1f}%)")
    print(f"   Time: {step1_time:.1f}s")

    # Step 2: Cell metrics
    print(f"\n🔍 Step 2: Calculating cell metrics...")
    start_time = time.time()
    cell_metrics = detector._calculate_cell_metrics(grid_df, edge_bins)
    step2_time = time.time() - start_time
    print(f"   ✅ Cells analyzed: {len(cell_metrics):,}")

    candidates = cell_metrics[cell_metrics['max_distance_m'] >= params.min_cell_distance]
    print(f"   Cells beyond {params.min_cell_distance}m: {len(candidates):,}")
    print(f"   Time: {step2_time:.1f}s")

    # Step 3: Apply filters
    print(f"\n🔍 Step 3: Applying overshooting filters...")
    start_time = time.time()
    overshooters, overshooting_bins = detector._apply_overshooting_filters(cell_metrics, edge_bins)
    step3_time = time.time() - start_time
    print(f"   ✅ Overshooters found: {len(overshooters):,}")
    print(f"   Time: {step3_time:.1f}s")

    if len(overshooters) == 0:
        print(f"\n⚠️  No overshooting cells detected with current parameters")
        return None

    # Step 4: Tilt recommendations
    print(f"\n🔍 Step 4: Calculating tilt recommendations...")
    start_time = time.time()
    # Calculate grid distances (needed for data-driven max distance calc)
    grid_with_distance = detector._calculate_grid_distances(grid_df, gis_df)
    overshooters = detector._calculate_tilt_recommendations(overshooters, overshooting_bins, gis_df, grid_with_distance)
    step4_time = time.time() - start_time
    print(f"   ✅ Recommendations calculated")
    print(f"   Time: {step4_time:.1f}s")

    # Step 5: Severity scores
    print(f"\n🔍 Step 5: Calculating severity scores...")
    start_time = time.time()
    overshooters = detector._calculate_severity_scores(overshooters, grid_df)
    step5_time = time.time() - start_time
    print(f"   ✅ Severity scores calculated")
    print(f"   Time: {step5_time:.1f}s")

    # Results
    print(f"\n" + "="*80)
    print(f"✅ OVERSHOOTING DETECTION RESULTS - FULL DATASET")
    print(f"="*80)

    print(f"\n📊 Summary Statistics:")
    print(f"   Total cells in network: {len(gis_df):,}")
    print(f"   Overshooting cells detected: {len(overshooters):,}")
    print(f"   Percentage overshooting: {len(overshooters)/len(gis_df)*100:.1f}%")
    print(f"   Cells optimized: {len(gis_df) - len(overshooters):,} ({(len(gis_df)-len(overshooters))/len(gis_df)*100:.1f}%)")

    print(f"\n📊 Overshooting Severity:")
    print(f"   Total overshooting grid bins: {overshooters['overshooting_grids'].sum():,}")
    print(f"   Avg overshooting grids per cell: {overshooters['overshooting_grids'].mean():.1f}")
    print(f"   Max overshooting grids: {overshooters['overshooting_grids'].max():,}")
    print(f"   Median overshooting grids: {overshooters['overshooting_grids'].median():.1f}")

    print(f"\n📊 Coverage Distance:")
    print(f"   Avg max distance: {overshooters['max_distance_m'].mean()/1000:.1f} km")
    print(f"   Max serving distance: {overshooters['max_distance_m'].max()/1000:.1f} km")
    print(f"   Median max distance: {overshooters['max_distance_m'].median()/1000:.1f} km")

    print(f"\n📊 Tilt Recommendations:")
    print(f"   Avg recommended increase: +{overshooters['recommended_tilt_change'].mean():.1f}°")
    print(f"   Max recommended increase: +{overshooters['recommended_tilt_change'].max():.1f}°")
    print(f"   Median recommended increase: +{overshooters['recommended_tilt_change'].median():.1f}°")

    print(f"\n📊 Severity Scores:")
    print(f"   Avg severity score: {overshooters['severity_score'].mean():.3f}")
    print(f"   Max severity score: {overshooters['severity_score'].max():.3f}")
    print(f"   Median severity score: {overshooters['severity_score'].median():.3f}")
    print(f"   Min severity score: {overshooters['severity_score'].min():.3f}")

    # Top 20 worst overshooters (by severity score)
    print(f"\n📋 TOP 20 WORST OVERSHOOTERS (by severity score):")
    print(f"="*80)
    top20 = overshooters.nlargest(20, 'severity_score')

    display_cols = [
        'cell_id',
        'severity_score',
        'severity_category',
        'overshooting_grids',
        'total_grids',
        'percentage_overshooting',
        'max_distance_m',
        'recommended_tilt_change',
    ]

    # Format for display
    display_df = top20[display_cols].copy()
    display_df['severity_score'] = display_df['severity_score'].round(3)
    display_df['max_distance_m'] = display_df['max_distance_m'] / 1000  # Convert to km
    display_df = display_df.rename(columns={'max_distance_m': 'max_dist_km'})
    display_df['percentage_overshooting'] = (display_df['percentage_overshooting'] * 100).round(1)
    display_df = display_df.rename(columns={'percentage_overshooting': 'overshoot_%'})

    print(display_df.to_string(index=False))

    # Distribution by severity category
    print(f"\n📊 Severity Category Distribution:")
    print(f"   (Based on multi-factor score: bins, percentage, distance, RSRP, traffic)")

    # Define order for categories
    category_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
    severity_dist = overshooters['severity_category'].value_counts()

    for category in category_order:
        count = severity_dist.get(category, 0)
        pct = count / len(overshooters) * 100 if len(overshooters) > 0 else 0

        # Get score range for this category
        category_cells = overshooters[overshooters['severity_category'] == category]
        if len(category_cells) > 0:
            avg_score = category_cells['severity_score'].mean()
            min_score = category_cells['severity_score'].min()
            max_score = category_cells['severity_score'].max()
            print(f"   {category:>8}: {count:3,} cells ({pct:5.1f}%) | Scores: {min_score:.3f}-{max_score:.3f} (avg {avg_score:.3f})")
        else:
            print(f"   {category:>8}: {count:3,} cells ({pct:5.1f}%)")

    # Breakdown by frequency band
    print(f"\n📊 Overshooting by Frequency Band:")
    # Merge with grid to get band info
    overshooters_with_band = overshooters.merge(
        grid_df[['cell_id', 'Band']].drop_duplicates('cell_id'),
        on='cell_id',
        how='left'
    )

    # Count total cells per band
    cells_per_band = grid_df[['cell_id', 'Band']].drop_duplicates().groupby('Band').size()

    # Count overshooters per band
    band_dist = overshooters_with_band.groupby('Band').size().sort_index()

    for band in sorted(cells_per_band.index):
        overshooter_count = band_dist.get(band, 0)
        total_count = cells_per_band[band]
        pct = overshooter_count / total_count * 100 if total_count > 0 else 0
        print(f"   {band:>4} MHz: {overshooter_count:3,} overshooters / {total_count:3,} cells ({pct:5.1f}%)")

    # Save results
    output_file = Path("data/vf-ie/output-data/overshooting_cells.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving results...")
    overshooters.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   Rows: {len(overshooters):,}")
    print(f"   Columns: {len(overshooters.columns)}")

    return overshooters


if __name__ == "__main__":
    print("\n🚀 FULL DATASET OVERSHOOTING DETECTION")
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
        print(f"Overshooting cells identified: {len(result):,}")
        print(f"Results saved to: data/vf-ie/output-data/overshooting_cells.csv")

    print()
