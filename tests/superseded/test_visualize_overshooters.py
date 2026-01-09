"""
Generate interactive maps for overshooting cells.

Creates:
1. Main map with severity-coded triangular sectors and clickable grid bins
2. Heatmap showing severity distribution
"""
from pathlib import Path
import pandas as pd
import time

from ran_optimizer.visualization import create_overshooting_map, create_severity_heatmap
from ran_optimizer.utils.logging_config import configure_logging, get_logger
from ran_optimizer.utils.geohash import decode

configure_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


def generate_maps():
    """Generate interactive maps for overshooting cells."""
    print("\n" + "="*80)
    print("OVERSHOOTING CELLS - MAP VISUALIZATION")
    print("="*80)

    # Load ENVIRONMENT-AWARE overshooting cells data
    overshooters_file = Path("data/output-data/vf-ie/recommendations/overshooting_cells_environment_aware.csv")

    if not overshooters_file.exists():
        print(f"\n❌ File not found: {overshooters_file}")
        print(f"   Please run test_environment_aware_detection.py first")
        return

    print(f"\n📂 Loading environment-aware overshooting cells data...")
    overshooters_df = pd.read_csv(overshooters_file)
    print(f"   ✅ Loaded {len(overshooters_df):,} overshooting cells (environment-aware detection)")

    # Show environment breakdown
    if 'environment' in overshooters_df.columns:
        env_counts = overshooters_df['environment'].value_counts()
        print(f"\n   Environment breakdown:")
        for env in ['URBAN', 'SUBURBAN', 'RURAL']:
            count = env_counts.get(env, 0)
            pct = count / len(overshooters_df) * 100
            print(f"      {env:10s}: {count:3d} cells ({pct:5.1f}%)")

    # Load full cell coverage data to get:
    # 1. Cell coordinates (latitude, longitude)
    # 2. Cell azimuth (bearing)
    # 3. Grid bins for each cell
    print(f"\n📂 Loading cell coverage data for coordinates, azimuth, and grids...")
    coverage_file = Path("data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv")

    if not coverage_file.exists():
        print(f"❌ Coverage file not found: {coverage_file}")
        return

    # Load relevant columns
    print(f"   Loading data (this may take a moment)...")
    start_time = time.time()

    coverage_df = pd.read_csv(
        coverage_file,
        usecols=['cilac', 'Latitude', 'Longitude', 'Bearing', 'grid', 'Band', 'distance_to_cell', 'event_count']
    )

    load_time = time.time() - start_time
    print(f"   ✅ Loaded {len(coverage_df):,} measurements in {load_time:.1f}s")

    # Get cell coordinates, azimuth, and band (one per cell)
    print(f"\n🔄 Extracting cell locations, azimuth, and band...")
    cell_info = coverage_df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'Band']].drop_duplicates('cilac').rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'Band': 'band',
    })

    print(f"   ✅ Got coordinates for {len(cell_info):,} cells")

    # Merge with overshooters
    overshooters_df = overshooters_df.merge(cell_info, on='cell_id', how='left')

    # Check for missing data
    missing_coords = overshooters_df[overshooters_df['latitude'].isna()]
    if len(missing_coords) > 0:
        print(f"   ⚠️  Warning: {len(missing_coords)} cells missing coordinates")
        overshooters_df = overshooters_df[overshooters_df['latitude'].notna()]

    print(f"   ✅ Merged coordinates and azimuth for {len(overshooters_df):,} cells")

    # Calculate intersite distance and classify environment (SITE-LEVEL)
    print(f"\n🔄 Calculating intersite distances (site-level) and environment classification...")
    from scipy.spatial import cKDTree
    import numpy as np

    # Get unique physical sites (all bands)
    cell_info['site_key'] = cell_info['latitude'].round(6).astype(str) + '_' + cell_info['longitude'].round(6).astype(str)
    unique_sites = cell_info[['latitude', 'longitude', 'site_key']].drop_duplicates('site_key')

    print(f"   Total cells: {len(cell_info):,}")
    print(f"   Unique sites: {len(unique_sites):,}")

    # Build KD-tree for ALL sites (band-agnostic)
    coords = unique_sites[['latitude', 'longitude']].values
    tree = cKDTree(coords)

    # Calculate intersite distance for each site
    min_distance_km = 0.1  # 100m minimum distance to be considered a different site
    site_env_map = {}

    for idx, site in unique_sites.iterrows():
        site_coords = [[site['latitude'], site['longitude']]]

        # Query more sites than we need to filter out co-located ones
        k = min(20, len(unique_sites))  # Get up to 20 nearest sites
        distances, indices = tree.query(site_coords, k=k)
        distances_km = distances[0] * 111  # Convert to km

        # Filter to sites that are at least 100m away (exclude co-located cells)
        valid_distances = []
        for i in range(1, len(distances_km)):  # Skip index 0 (self)
            if distances_km[i] >= min_distance_km:
                valid_distances.append(distances_km[i])
                if len(valid_distances) == 3:  # We only need 3 neighbors
                    break

        # Calculate mean of valid neighbors
        if len(valid_distances) >= 3:
            mean_intersite_km = np.mean(valid_distances[:3])
        elif len(valid_distances) >= 1:
            mean_intersite_km = np.mean(valid_distances)
        else:
            mean_intersite_km = 2.0  # Default to suburban if isolated

        # Classify: URBAN ≤1km, RURAL ≥3km, else SUBURBAN
        if mean_intersite_km <= 1.0:
            environment = 'URBAN'
        elif mean_intersite_km >= 3.0:
            environment = 'RURAL'
        else:
            environment = 'SUBURBAN'

        site_env_map[site['site_key']] = {
            'intersite_distance_km': mean_intersite_km,
            'environment': environment
        }

    # Apply site-level classification to all cells at each site
    env_results = []
    for idx, cell in cell_info.iterrows():
        site_info = site_env_map.get(cell['site_key'], {'intersite_distance_km': 2.0, 'environment': 'SUBURBAN'})
        env_results.append({
            'cell_id': cell['cell_id'],
            'intersite_distance_km': site_info['intersite_distance_km'],
            'environment': site_info['environment']
        })

    env_df = pd.DataFrame(env_results)

    # Show site-level breakdown
    site_env_counts = pd.DataFrame(site_env_map.values())['environment'].value_counts()
    print(f"   ✅ Site-level environment classification:")
    print(f"      URBAN:     {site_env_counts.get('URBAN', 0):3d} sites")
    print(f"      SUBURBAN:  {site_env_counts.get('SUBURBAN', 0):3d} sites")
    print(f"      RURAL:     {site_env_counts.get('RURAL', 0):3d} sites")

    # Merge environment classification with overshooters (only if not already present)
    if 'environment' not in overshooters_df.columns:
        overshooters_df = overshooters_df.merge(env_df[['cell_id', 'environment', 'intersite_distance_km']],
                                                on='cell_id', how='left')

        # Fill any missing with SUBURBAN as default
        overshooters_df['environment'] = overshooters_df['environment'].fillna('SUBURBAN')
        overshooters_df['intersite_distance_km'] = overshooters_df['intersite_distance_km'].fillna(2.0)
    else:
        print(f"   ℹ️  Environment column already present in data (using environment-aware detection results)")

    env_counts = overshooters_df['environment'].value_counts()
    print(f"   ✅ Environment classification complete:")
    print(f"      URBAN:     {env_counts.get('URBAN', 0):3d} cells")
    print(f"      SUBURBAN:  {env_counts.get('SUBURBAN', 0):3d} cells")
    print(f"      RURAL:     {env_counts.get('RURAL', 0):3d} cells")

    # Prepare grid data with overshooting flag
    print(f"\n🔄 Preparing grid bins for visualization...")

    # Load the original grid data which has overshooting information
    # We need to identify which grid bins are overshooting for each cell

    # For now, let's create a simpler version - load all grids for overshooting cells
    # and mark them based on the detector's results

    # Get unique overshooting cell IDs
    overshooting_cell_ids = set(overshooters_df['cell_id'].unique())

    # Filter coverage data to only overshooting cells
    overshooting_coverage = coverage_df[coverage_df['cilac'].isin(overshooting_cell_ids)].copy()

    # Decode geohash to get grid coordinates
    print(f"   Decoding geohash coordinates...")
    grid_coords = overshooting_coverage['grid'].apply(lambda gh: decode(gh))
    overshooting_coverage['grid_lat'] = grid_coords.apply(lambda x: x[0])
    overshooting_coverage['grid_lon'] = grid_coords.apply(lambda x: x[1])

    # Rename columns for grid data
    grid_df = overshooting_coverage[['cilac', 'grid', 'grid_lat', 'grid_lon']].rename(columns={
        'cilac': 'cell_id',
        'grid': 'geohash7',
        'grid_lat': 'latitude',
        'grid_lon': 'longitude',
    })

    # Add is_overshooting flag by running the detector's logic
    print(f"   Running overshooting bin detection...")

    from ran_optimizer.recommendations import OvershooterParams

    params = OvershooterParams()

    # Add distance and band info
    grid_df = grid_df.merge(
        overshooting_coverage[['cilac', 'grid', 'distance_to_cell', 'Band']],
        left_on=['cell_id', 'geohash7'],
        right_on=['cilac', 'grid'],
        how='left'
    ).drop(columns=['cilac', 'grid'])

    # Mark edge bins per cell
    def mark_edge_bins(cell_grids):
        edge_distance = cell_grids['distance_to_cell'].quantile(params.edge_traffic_percent)
        cell_grids['is_edge'] = cell_grids['distance_to_cell'] >= edge_distance
        return cell_grids

    grid_df = grid_df.groupby('cell_id', group_keys=False).apply(mark_edge_bins)

    # Competition filter: count cells per grid per band
    grid_cell_counts = coverage_df.groupby(['grid', 'Band']).agg({
        'cilac': 'nunique',
        'event_count': 'sum',
    }).reset_index()
    grid_cell_counts.columns = ['grid', 'Band', 'cells_in_grid', 'total_grid_traffic']

    grid_df = grid_df.merge(
        grid_cell_counts,
        left_on=['geohash7', 'Band'],
        right_on=['grid', 'Band'],
        how='left'
    ).drop(columns=['grid'])

    # Get traffic per cell-grid
    cell_grid_traffic = coverage_df[['cilac', 'grid', 'event_count']].rename(columns={
        'cilac': 'cell_id',
        'grid': 'geohash7',
        'event_count': 'cell_traffic'
    })

    grid_df = grid_df.merge(cell_grid_traffic, on=['cell_id', 'geohash7'], how='left')
    grid_df['cell_traffic_pct'] = grid_df['cell_traffic'] / grid_df['total_grid_traffic']

    # Mark overshooting (OLD algorithm): edge + competition only
    grid_df['is_overshooting_old'] = (
        (grid_df['is_edge'] == True) &
        (grid_df['cells_in_grid'] >= params.min_cell_count_in_grid) &
        (grid_df['cell_traffic_pct'] <= params.max_percentage_grid_events)
    )

    # NEW: Add relative distance criterion
    # Calculate max distance any cell reaches to each grid bin
    print(f"   Calculating relative distance criterion...")
    grid_max_distances = coverage_df.groupby(['grid', 'Band'])['distance_to_cell'].max().reset_index()
    grid_max_distances.columns = ['grid', 'Band', 'grid_max_distance']

    grid_df = grid_df.merge(
        grid_max_distances,
        left_on=['geohash7', 'Band'],
        right_on=['grid', 'Band'],
        how='left'
    ).drop(columns=['grid'])

    # Calculate relative reach: how far is THIS cell reaching compared to the furthest?
    grid_df['relative_reach'] = grid_df['distance_to_cell'] / grid_df['grid_max_distance']

    # Mark overshooting (NEW algorithm): edge + competition + relative distance
    grid_df['is_overshooting'] = (
        (grid_df['is_edge'] == True) &
        (grid_df['cells_in_grid'] >= params.min_cell_count_in_grid) &
        (grid_df['cell_traffic_pct'] <= params.max_percentage_grid_events) &
        (grid_df['relative_reach'] >= params.min_relative_reach)  # NEW criterion
    )

    # Compare OLD vs NEW
    overshooting_old = grid_df['is_overshooting_old'].sum()
    overshooting_new = grid_df['is_overshooting'].sum()
    reduction = overshooting_old - overshooting_new
    reduction_pct = (reduction / overshooting_old * 100) if overshooting_old > 0 else 0

    print(f"   ✅ Prepared {len(grid_df):,} grid bins")
    print(f"\n   📊 Algorithm Comparison:")
    print(f"      OLD (no relative distance):")
    print(f"         Overshooting: {overshooting_old:,} bins ({overshooting_old/len(grid_df)*100:.1f}%)")
    print(f"         Normal: {len(grid_df) - overshooting_old:,} bins ({(len(grid_df)-overshooting_old)/len(grid_df)*100:.1f}%)")
    print(f"\n      NEW (with relative_reach ≥ {params.min_relative_reach}):")
    print(f"         Overshooting: {overshooting_new:,} bins ({overshooting_new/len(grid_df)*100:.1f}%)")
    print(f"         Normal: {len(grid_df) - overshooting_new:,} bins ({(len(grid_df)-overshooting_new)/len(grid_df)*100:.1f}%)")
    print(f"\n      📉 Reduction: {reduction:,} bins ({reduction_pct:.1f}% fewer false positives)")

    # Keep only needed columns
    grid_df = grid_df[['cell_id', 'geohash7', 'latitude', 'longitude', 'is_overshooting', 'is_overshooting_old', 'relative_reach', 'distance_to_cell']].copy()

    # Show data overview
    print(f"\n📊 Data Overview:")
    print(f"   Cells with coordinates: {len(overshooters_df):,}")
    print(f"   Latitude range: {overshooters_df['latitude'].min():.4f} to {overshooters_df['latitude'].max():.4f}")
    print(f"   Longitude range: {overshooters_df['longitude'].min():.4f} to {overshooters_df['longitude'].max():.4f}")
    print(f"   Azimuth range: {overshooters_df['azimuth_deg'].min():.1f}° to {overshooters_df['azimuth_deg'].max():.1f}°")

    severity_dist = overshooters_df['severity_category'].value_counts()
    print(f"\n   Severity breakdown:")
    for category in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        count = severity_dist.get(category, 0)
        print(f"      {category:>8}: {count:3,} cells")

    # Create output directory
    output_dir = Path("data/output-data/vf-ie/recommendations/maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main map with triangular sectors and grids
    print(f"\n🗺️  Generating main interactive map with sectors and grids...")
    start_time = time.time()

    main_map_file = output_dir / "overshooting_cells_map.html"

    try:
        map_obj = create_overshooting_map(
            overshooters_df=overshooters_df,
            grid_df=grid_df,  # Pass grid data
            show_sector_shapes=True,  # Show triangular sectors
            show_optimized_cells=False,
            output_file=main_map_file,
        )

        gen_time = time.time() - start_time
        file_size_mb = main_map_file.stat().st_size / (1024**2)

        print(f"   ✅ Map generated in {gen_time:.1f}s")
        print(f"   📁 Saved to: {main_map_file}")
        print(f"   📊 File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"   ❌ Failed to generate map: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate severity heatmap
    print(f"\n🗺️  Generating severity heatmap...")
    start_time = time.time()

    heatmap_file = output_dir / "overshooting_severity_heatmap.html"

    try:
        heatmap_obj = create_severity_heatmap(
            overshooters_df=overshooters_df,
            output_file=heatmap_file,
        )

        gen_time = time.time() - start_time
        file_size_mb = heatmap_file.stat().st_size / (1024**2)

        print(f"   ✅ Heatmap generated in {gen_time:.1f}s")
        print(f"   📁 Saved to: {heatmap_file}")
        print(f"   📊 File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"   ❌ Failed to generate heatmap: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n" + "="*80)
    print(f"✅ MAP GENERATION COMPLETE")
    print(f"="*80)
    print(f"\nGenerated files:")
    print(f"   1. Main map: {main_map_file}")
    print(f"   2. Heatmap:  {heatmap_file}")
    print(f"\nNew Features:")
    print(f"   ✓ Cells shown as triangular sectors (bearing ±20°)")
    print(f"   ✓ Sectors color-coded by severity")
    print(f"   ✓ Click cell markers to show/hide grid bins")
    print(f"   ✓ Grid bins: Red = overshooting, Gray = normal")
    print(f"\nTo view:")
    print(f"   Open the HTML files in your web browser")
    print(f"   • Click markers to see cell details and show/hide grids")
    print(f"   • Use layer controls to toggle sectors/markers")
    print(f"   • Zoom and pan to explore the network")
    print()


if __name__ == "__main__":
    print("\n🗺️  OVERSHOOTING CELLS MAP VISUALIZATION")
    print("="*80)

    start = time.time()
    generate_maps()
    total_time = time.time() - start

    print(f"Total processing time: {total_time:.1f}s")
    print()
