"""
Test undershooting detection on VF Ireland dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

from ran_optimizer.recommendations import UndershooterDetector, UndershooterParams

print("\n" + "="*80)
print("UNDERSHOOTING DETECTION TEST - VF IRELAND")
print("="*80)

# Load data
print("\n📂 Loading data...")
coverage_df = pd.read_csv(
    './data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv',
    low_memory=False
)
print(f"   ✅ Loaded {len(coverage_df):,} measurements")

# Extract cell info
print("\n🔄 Extracting cell information...")
cell_info = coverage_df.groupby('cilac').agg({
    'Latitude': 'first',
    'Longitude': 'first',
    'Bearing': 'first',
    'Band': 'first',
    'TiltM': 'first',
    'TiltE': 'first',
    'Height': 'first',
}).reset_index()
cell_info.columns = ['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band',
                     'mechanical_tilt', 'electrical_tilt', 'height']
print(f"   ✅ Extracted {len(cell_info)} cells")

# Calculate environment classification
print("\n🏙️  Calculating environment classification...")
cell_info['site_key'] = cell_info['latitude'].round(6).astype(str) + '_' + cell_info['longitude'].round(6).astype(str)
unique_sites = cell_info[['latitude', 'longitude', 'site_key']].drop_duplicates('site_key')

# Build KD-tree for ALL sites
coords = unique_sites[['latitude', 'longitude']].values
tree = cKDTree(coords)

min_distance_km = 0.1  # 100m minimum
site_env_map = {}

for idx, site in unique_sites.iterrows():
    site_coords = [[site['latitude'], site['longitude']]]
    k = min(20, len(unique_sites))
    distances, indices = tree.query(site_coords, k=k)
    distances_km = distances[0] * 111

    # Filter to sites ≥100m away
    valid_distances = []
    for i in range(1, len(distances_km)):
        if distances_km[i] >= min_distance_km:
            valid_distances.append(distances_km[i])
            if len(valid_distances) == 3:
                break

    mean_intersite_km = np.mean(valid_distances[:3]) if len(valid_distances) >= 3 else (np.mean(valid_distances) if len(valid_distances) >= 1 else 2.0)

    # Classify
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

# Apply site-level classification to all cells
env_results = []
for idx, cell in cell_info.iterrows():
    site_info = site_env_map.get(cell['site_key'], {'intersite_distance_km': 2.0, 'environment': 'SUBURBAN'})
    env_results.append({
        'cell_id': cell['cell_id'],
        'intersite_distance_km': site_info['intersite_distance_km'],
        'environment': site_info['environment']
    })

environment_df = pd.DataFrame(env_results)

env_counts = environment_df['environment'].value_counts()
print(f"   ✅ Classified {len(environment_df)} cells:")
print(f"      URBAN:    {env_counts.get('URBAN', 0):3d} cells ({env_counts.get('URBAN', 0)/len(environment_df)*100:5.1f}%)")
print(f"      SUBURBAN: {env_counts.get('SUBURBAN', 0):3d} cells ({env_counts.get('SUBURBAN', 0)/len(environment_df)*100:5.1f}%)")
print(f"      RURAL:    {env_counts.get('RURAL', 0):3d} cells ({env_counts.get('RURAL', 0)/len(environment_df)*100:5.1f}%)")

# Prepare GIS DataFrame
gis_df = cell_info[['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'mechanical_tilt', 'electrical_tilt', 'height']].copy()

# Prepare grid DataFrame with correct column names
print("\n🔄 Preparing grid data...")
grid_df = coverage_df[[
    'grid',
    'cilac',
    'Band',
    'avg_rsrp',
    'event_count',
    'distance_to_cell',
]].copy()

grid_df = grid_df.rename(columns={
    'grid': 'geohash7',
    'cilac': 'cell_id',
    'Band': 'Band',
    'avg_rsrp': 'rsrp',
    'event_count': 'total_traffic',
    'distance_to_cell': 'distance_m',
})

# Calculate cell_count per grid (for interference detection)
print("\n🔄 Calculating interference metrics...")
grid_cell_counts = grid_df.groupby('geohash7')['cell_id'].nunique().reset_index()
grid_cell_counts.columns = ['geohash7', 'cell_count']
grid_df = grid_df.merge(grid_cell_counts, on='geohash7', how='left')

print(f"   ✅ Grid data prepared: {len(grid_df):,} measurements")

# Run detection with default parameters
print("\n" + "="*80)
print("RUNNING UNDERSHOOTING DETECTION")
print("="*80)

params = UndershooterParams.from_config()
detector = UndershooterDetector(params)

print("\nDetection parameters:")
print(f"   Max cell distance: {params.max_cell_distance}m ({params.max_cell_distance/1000}km)")
print(f"   Min traffic: {params.min_cell_event_count} events")
print(f"   Max interference: {params.max_interference_percentage*100}%")
print(f"   Min coverage increase (1°): {params.min_coverage_increase_1deg*100}%")
print(f"   Min coverage increase (2°): {params.min_coverage_increase_2deg*100}%")

undershooters = detector.detect(grid_df, gis_df)

# Display results
print("\n" + "="*80)
print("📊 DETECTION RESULTS")
print("="*80)

if len(undershooters) > 0:
    print(f"\nTotal undershooters: {len(undershooters)}")

    # Merge with environment
    undershooters = undershooters.merge(environment_df, on='cell_id', how='left')

    # By environment
    if 'environment' in undershooters.columns:
        env_breakdown = undershooters['environment'].value_counts()
        print("\nBy environment:")
        for env in ['URBAN', 'SUBURBAN', 'RURAL']:
            count = env_breakdown.get(env, 0)
            pct = count / len(undershooters) * 100
            total_cells = env_counts.get(env, 1)
            detection_rate = count / total_cells * 100
            print(f"   {env:10s}: {count:3d} cells ({pct:5.1f}% of undershooters, {detection_rate:4.1f}% detection rate)")

    # By recommended uptilt
    uptilt_breakdown = undershooters['recommended_uptilt_deg'].value_counts()
    print("\nBy recommended uptilt:")
    for deg in sorted(uptilt_breakdown.index):
        count = uptilt_breakdown[deg]
        pct = count / len(undershooters) * 100
        print(f"   {deg}° uptilt: {count:3d} cells ({pct:5.1f}%)")

    # Statistics
    print("\nCoverage statistics:")
    print(f"   Current max distance:")
    print(f"      Mean:   {undershooters['max_distance_m'].mean():,.0f}m")
    print(f"      Median: {undershooters['max_distance_m'].median():,.0f}m")
    print(f"      Min:    {undershooters['max_distance_m'].min():,.0f}m")
    print(f"      Max:    {undershooters['max_distance_m'].max():,.0f}m")

    print(f"\n   Coverage increase:")
    print(f"      Mean:   {undershooters['coverage_increase_percentage'].mean()*100:.1f}%")
    print(f"      Median: {undershooters['coverage_increase_percentage'].median()*100:.1f}%")
    print(f"      Min:    {undershooters['coverage_increase_percentage'].min()*100:.1f}%")
    print(f"      Max:    {undershooters['coverage_increase_percentage'].max()*100:.1f}%")

    # Sample cells
    print("\nTop 10 candidates by coverage increase:")
    top10 = undershooters.nlargest(10, 'coverage_increase_percentage')[
        ['cell_id', 'environment', 'max_distance_m', 'recommended_uptilt_deg',
         'new_max_distance_m', 'coverage_increase_percentage']
    ]
    for idx, row in top10.iterrows():
        print(f"   Cell {row['cell_id']} ({row['environment']}): " +
              f"{row['max_distance_m']:.0f}m → {row['new_max_distance_m']:.0f}m " +
              f"(+{row['coverage_increase_percentage']*100:.1f}%, {row['recommended_uptilt_deg']:.0f}° uptilt)")

    # Save results
    output_dir = Path('./data/output-data/vf-ie/recommendations')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'undershooting_cells.csv'
    undershooters.to_csv(output_file, index=False)
    print(f"\n💾 Saved results to: {output_file}")

else:
    print("\nNo undershooting cells detected")

print("\n" + "="*80)
print("✅ UNDERSHOOTING DETECTION TEST COMPLETE")
print("="*80)
