"""
Diagnostic script to debug overshooting detection.

Analyzes grid data to understand distance distributions and edge bin logic.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from ran_optimizer.data.loaders import load_grid_data, load_gis_data
from ran_optimizer.recommendations import OvershooterParams, OvershooterDetector
from ran_optimizer.utils.logging_config import configure_logging, get_logger

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def diagnose_distance_distribution():
    """Analyze distance distribution in grid data."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("DIAGNOSTIC 1: DISTANCE DISTRIBUTION ANALYSIS")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print("❌ Data files not found")
        return

    # Load data
    print("\n📂 Loading data (500K sample)...")
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=500000,
        decode_geohash=True,
        validate=False,
    )
    print(f"   Loaded {len(grid_df)} grid measurements")
    print(f"   Loaded {len(gis_df)} cells")

    # Calculate distances manually
    print("\n📏 Calculating distances...")
    grid_with_cell = grid_df.merge(
        gis_df[['cell_id', 'latitude', 'longitude']],
        on='cell_id',
        how='left',
        suffixes=('_grid', '_cell')
    )

    from ran_optimizer.core.geometry import haversine_distance
    grid_with_cell['distance_m'] = grid_with_cell.apply(
        lambda row: haversine_distance(
            row['latitude_cell'],
            row['longitude_cell'],
            row['latitude_grid'],
            row['longitude_grid']
        ) if pd.notna(row['latitude_cell']) else np.nan,
        axis=1
    )

    # Filter out NaN distances
    valid_distances = grid_with_cell[pd.notna(grid_with_cell['distance_m'])]
    print(f"   Valid measurements with distances: {len(valid_distances)}")

    # Overall distance statistics
    print(f"\n📊 Overall Distance Statistics:")
    print(f"   Min distance: {valid_distances['distance_m'].min():.1f}m")
    print(f"   Max distance: {valid_distances['distance_m'].max():.1f}m ({valid_distances['distance_m'].max()/1000:.1f}km)")
    print(f"   Mean distance: {valid_distances['distance_m'].mean():.1f}m ({valid_distances['distance_m'].mean()/1000:.1f}km)")
    print(f"   Median distance: {valid_distances['distance_m'].median():.1f}m ({valid_distances['distance_m'].median()/1000:.1f}km)")

    # Percentiles
    print(f"\n📊 Distance Percentiles:")
    for p in [10, 25, 50, 75, 85, 90, 95, 99]:
        dist = valid_distances['distance_m'].quantile(p/100)
        print(f"   {p:2d}th percentile: {dist:7.1f}m ({dist/1000:5.2f}km)")

    # Bins beyond 4km threshold
    beyond_4km = valid_distances[valid_distances['distance_m'] >= 4000]
    print(f"\n📊 Measurements Beyond 4km:")
    print(f"   Count: {len(beyond_4km):,}")
    print(f"   Percentage: {len(beyond_4km)/len(valid_distances)*100:.1f}%")
    print(f"   Unique cells serving >4km: {beyond_4km['cell_id'].nunique()}")

    # Per-cell distance analysis
    print(f"\n📊 Per-Cell Distance Analysis:")
    cell_distances = valid_distances.groupby('cell_id')['distance_m'].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max'),
        ('mean', 'mean'),
    ]).reset_index()

    cells_over_4km = cell_distances[cell_distances['max'] >= 4000]
    print(f"   Cells with max distance >= 4km: {len(cells_over_4km)}")
    print(f"   Percentage: {len(cells_over_4km)/len(cell_distances)*100:.1f}%")

    print(f"\n📋 Top 10 Cells by Max Distance:")
    top_cells = cell_distances.nlargest(10, 'max')
    print(top_cells.to_string(index=False))

    return valid_distances, cell_distances


def diagnose_edge_bin_logic():
    """Debug the edge bin identification logic."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("DIAGNOSTIC 2: EDGE BIN LOGIC ANALYSIS")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print("❌ Data files not found")
        return

    # Load data
    print("\n📂 Loading data (500K sample)...")
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=500000,
        decode_geohash=True,
        validate=False,
    )

    # Use detector to calculate distances
    print("\n🔍 Running edge bin identification...")
    detector = OvershooterDetector(OvershooterParams(edge_traffic_percent=0.15))

    grid_with_dist = detector._calculate_grid_distances(grid_df, gis_df)

    # Check for NaN distances
    nan_count = grid_with_dist['distance_m'].isna().sum()
    print(f"   Total grid measurements: {len(grid_with_dist)}")
    print(f"   NaN distances: {nan_count}")
    print(f"   Valid distances: {len(grid_with_dist) - nan_count}")

    # Filter to valid
    valid_grid = grid_with_dist[pd.notna(grid_with_dist['distance_m'])].copy()

    # Calculate edge threshold per cell
    print(f"\n📊 Calculating edge thresholds per cell...")
    edge_threshold = valid_grid.groupby('cell_id')['distance_m'].quantile(0.85).reset_index()
    edge_threshold.columns = ['cell_id', 'edge_distance_m']

    print(f"   Cells with measurements: {len(edge_threshold)}")
    print(f"   Mean edge threshold: {edge_threshold['edge_distance_m'].mean():.1f}m")
    print(f"   Median edge threshold: {edge_threshold['edge_distance_m'].median():.1f}m")

    # Show sample thresholds
    print(f"\n📋 Sample Edge Thresholds (10 random cells):")
    sample = edge_threshold.sample(min(10, len(edge_threshold)))
    print(sample.to_string(index=False))

    # Try identifying edge bins
    print(f"\n🔍 Identifying edge bins...")
    edge_bins = detector._identify_edge_bins(valid_grid)

    print(f"   Edge bins found: {len(edge_bins)}")
    print(f"   Percentage: {len(edge_bins)/len(valid_grid)*100:.1f}%")

    if len(edge_bins) > 0:
        print(f"\n📊 Edge Bin Statistics:")
        print(f"   Min distance: {edge_bins['distance_m'].min():.1f}m")
        print(f"   Max distance: {edge_bins['distance_m'].max():.1f}m")
        print(f"   Mean distance: {edge_bins['distance_m'].mean():.1f}m")
        print(f"   Unique cells: {edge_bins['cell_id'].nunique()}")

    return valid_grid, edge_bins


def diagnose_with_relaxed_params():
    """Test detection with very relaxed parameters."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("DIAGNOSTIC 3: DETECTION WITH RELAXED PARAMETERS")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print("❌ Data files not found")
        return

    # Load data
    print("\n📂 Loading data (500K sample)...")
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=500000,
        decode_geohash=True,
        validate=False,
    )

    # Filter to valid distances only
    print("\n🧹 Filtering to valid measurements...")
    grid_with_cell = grid_df.merge(
        gis_df[['cell_id', 'latitude', 'longitude']],
        on='cell_id',
        how='inner',  # Only keep measurements with matching cells
        suffixes=('_grid', '_cell')
    )
    print(f"   Measurements with matching cells: {len(grid_with_cell)}")

    # Test with very relaxed parameters
    params = OvershooterParams(
        edge_traffic_percent=0.15,
        min_cell_distance=1000,  # Only 1km
        percent_max_distance=0.5,
        min_cell_count_in_grid=2,  # Only 2 cells
        max_percentage_grid_events=0.50,  # Allow 50%
        rsrp_offset=0.5,
        min_overshooting_grids=5,  # Only 5 bins needed
        percentage_overshooting_grids=0.01,  # Only 1%
    )

    print(f"\n⚙️  Testing with VERY relaxed parameters:")
    print(f"   Min cell distance: {params.min_cell_distance}m")
    print(f"   Min overshooting grids: {params.min_overshooting_grids}")
    print(f"   Min percentage: {params.percentage_overshooting_grids*100}%")

    from ran_optimizer.recommendations import detect_overshooting_cells
    overshooters = detect_overshooting_cells(
        grid_with_cell[grid_df.columns],  # Use original grid columns
        gis_df,
        params
    )

    print(f"\n✅ Detection Results:")
    print(f"   Overshooters found: {len(overshooters)}")

    if len(overshooters) > 0:
        print(f"\n📊 Overshooting Cells:")
        display_cols = [
            'cell_id',
            'overshooting_grids',
            'total_grids',
            'percentage_overshooting',
            'max_distance_m',
        ]
        available_cols = [c for c in display_cols if c in overshooters.columns]
        print(overshooters[available_cols].head(10).to_string(index=False))
    else:
        print("\n⚠️  Still no overshooters found - there may be a logic issue")

    return overshooters


def check_sample_cell():
    """Deep dive into a single cell's data."""
    grid_file = Path("data/input-data/vf-ie/grid/grid-cell-data-150m.csv")
    gis_file = Path("data/input-data/vf-ie/gis/cell-gis.csv")

    print("\n" + "="*80)
    print("DIAGNOSTIC 4: SINGLE CELL DEEP DIVE")
    print("="*80)

    if not grid_file.exists() or not gis_file.exists():
        print("❌ Data files not found")
        return

    # Load data
    print("\n📂 Loading data...")
    gis_df = load_gis_data(gis_file, operator="Vodafone_Ireland")
    grid_df = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        sample_rows=500000,
        decode_geohash=True,
        validate=False,
    )

    # Find a cell with lots of measurements
    cell_counts = grid_df['cell_id'].value_counts()
    top_cell = cell_counts.index[0]

    print(f"\n🔬 Analyzing cell: {top_cell}")
    print(f"   Measurements: {cell_counts[top_cell]}")

    # Get cell location
    cell_info = gis_df[gis_df['cell_id'] == top_cell].iloc[0]
    print(f"   Location: ({cell_info['latitude']:.5f}, {cell_info['longitude']:.5f})")
    print(f"   Azimuth: {cell_info['azimuth_deg']}°")
    print(f"   Mechanical tilt: {cell_info['mechanical_tilt']}°")

    # Get measurements for this cell
    cell_grid = grid_df[grid_df['cell_id'] == top_cell].copy()

    # Calculate distances
    from ran_optimizer.core.geometry import haversine_distance
    cell_grid['distance_m'] = cell_grid.apply(
        lambda row: haversine_distance(
            cell_info['latitude'],
            cell_info['longitude'],
            row['latitude'],
            row['longitude']
        ),
        axis=1
    )

    print(f"\n📊 Distance Distribution for this cell:")
    print(f"   Min: {cell_grid['distance_m'].min():.1f}m")
    print(f"   Max: {cell_grid['distance_m'].max():.1f}m ({cell_grid['distance_m'].max()/1000:.1f}km)")
    print(f"   Mean: {cell_grid['distance_m'].mean():.1f}m")
    print(f"   Median: {cell_grid['distance_m'].median():.1f}m")

    # Edge threshold
    edge_threshold = cell_grid['distance_m'].quantile(0.85)
    print(f"\n📊 Edge Analysis (85th percentile = {edge_threshold:.1f}m):")
    edge = cell_grid[cell_grid['distance_m'] >= edge_threshold]
    print(f"   Edge bins: {len(edge)} ({len(edge)/len(cell_grid)*100:.1f}%)")

    if len(edge) > 0:
        print(f"   Edge distance range: {edge['distance_m'].min():.1f}m - {edge['distance_m'].max():.1f}m")

    # Show sample measurements
    print(f"\n📋 Sample Measurements:")
    sample = cell_grid.nlargest(5, 'distance_m')[['geohash7', 'distance_m', 'rsrp', 'total_traffic']]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    print("\n🔍 OVERSHOOTING DETECTION DIAGNOSTICS")
    print("="*80)

    # Run diagnostics
    print("\nRunning diagnostic suite...")

    distances, cell_stats = diagnose_distance_distribution()
    valid_grid, edge_bins = diagnose_edge_bin_logic()
    overshooters = diagnose_with_relaxed_params()
    check_sample_cell()

    print("\n" + "="*80)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*80)
