"""
Create unified visualization with enhanced metrics for both overshooting and undershooting.

Uses the full dataset results with all enhanced metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Import existing visualization
from ran_optimizer.visualization.map_overshooters import (
    create_overshooting_map,
)

print("\n" + "="*80)
print("🗺️  UNIFIED MAP WITH ENHANCED METRICS")
print("="*80)

# Load data with enhanced metrics
print("\n📂 Loading data with enhanced metrics...")
overshooters_file = Path('./data/output-data/vf-ie/recommendations/overshooting_cells_full_dataset.csv')
undershooters_file = Path('./data/output-data/vf-ie/recommendations/undershooting_cells_full_dataset.csv')

if not overshooters_file.exists() or not undershooters_file.exists():
    print(f"❌ Missing files! Please run:")
    print(f"   python tests/integration/test_full_dataset_overshooters.py")
    print(f"   python tests/integration/test_undershooting_full_dataset.py")
    exit(1)

overshooters_df = pd.read_csv(overshooters_file)
undershooters_df = pd.read_csv(undershooters_file)

print(f"   ✅ Overshooting cells: {len(overshooters_df)}")
print(f"   ✅ Undershooting cells: {len(undershooters_df)}")

# Load environment info from environment-aware files
print("\n📂 Loading environment classification data...")
env_over_file = Path('./data/output-data/vf-ie/recommendations/overshooting_cells_environment_aware.csv')
env_under_file = Path('./data/output-data/vf-ie/recommendations/undershooting_cells_environment_aware.csv')

if env_over_file.exists():
    env_over_df = pd.read_csv(env_over_file)
    if 'environment' in env_over_df.columns and 'intersite_distance_km' in env_over_df.columns:
        overshooters_df = overshooters_df.merge(
            env_over_df[['cell_id', 'environment', 'intersite_distance_km']],
            on='cell_id',
            how='left'
        )
        print(f"   ✅ Merged environment info for overshooting cells")
    else:
        print(f"   ⚠️  Environment file missing required columns")
else:
    print(f"   ⚠️  No environment classification file for overshooting")

if env_under_file.exists():
    env_under_df = pd.read_csv(env_under_file)
    if 'environment' in env_under_df.columns and 'intersite_distance_km' in env_under_df.columns:
        undershooters_df = undershooters_df.merge(
            env_under_df[['cell_id', 'environment', 'intersite_distance_km']],
            on='cell_id',
            how='left'
        )
        print(f"   ✅ Merged environment info for undershooting cells")
    else:
        print(f"   ⚠️  Environment file missing required columns")
else:
    print(f"   ⚠️  No environment classification file for undershooting")

# Verify enhanced metrics are present
print("\n🔍 Verifying enhanced metrics...")
over_metrics = ['current_interference_grids', 'removed_interference_grids', 'new_interference_grids', 'interference_reduction_pct']
under_metrics = ['current_coverage_grids', 'new_coverage_grids', 'distance_increase_m', 'total_coverage_after_uptilt']

over_missing = [m for m in over_metrics if m not in overshooters_df.columns]
under_missing = [m for m in under_metrics if m not in undershooters_df.columns]

if over_missing or under_missing:
    print(f"❌ Missing enhanced metrics!")
    if over_missing:
        print(f"   Overshooting missing: {over_missing}")
    if under_missing:
        print(f"   Undershooting missing: {under_missing}")
    exit(1)

print(f"   ✅ All enhanced metrics present!")

# Load cell coverage data
print("\n📂 Loading cell coverage data...")
start_time = time.time()
coverage_df = pd.read_csv(
    './data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv',
    low_memory=False
)
elapsed = time.time() - start_time
print(f"   ✅ Loaded {len(coverage_df):,} measurements in {elapsed:.1f}s")

# Extract cell info
print("\n🔄 Extracting cell information...")
cell_info = coverage_df.groupby('cilac').agg({
    'Latitude': 'first',
    'Longitude': 'first',
    'Bearing': 'first',
    'Band': 'first',
}).reset_index()
cell_info.columns = ['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']

# Merge with both datasets (include band, lat, lon, azimuth)
overshooters_df = overshooters_df.merge(
    cell_info[['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']],
    on='cell_id',
    how='inner',
    suffixes=('', '_from_coverage')
)

undershooters_df = undershooters_df.merge(
    cell_info[['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']],
    on='cell_id',
    how='inner',
    suffixes=('', '_from_coverage')
)

print(f"   ✅ Merged coordinates for {len(overshooters_df)} overshooting cells")
print(f"   ✅ Merged coordinates for {len(undershooters_df)} undershooting cells")

# Prepare grid data for both
print("\n🔄 Preparing grid data...")

# Overshooting grids - mark far edge grids as overshooting (simplified heuristic)
over_grid_df = coverage_df[coverage_df['cilac'].isin(overshooters_df['cell_id'])].copy()
over_grid_df = over_grid_df.rename(columns={
    'cilac': 'cell_id',
    'grid': 'geohash7',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

# Mark grids in the far 20% of each cell's range as "overshooting"
cell_max_dist = over_grid_df.groupby('cell_id')['distance_to_cell'].quantile(0.80).reset_index()
cell_max_dist.columns = ['cell_id', 'edge_threshold']
over_grid_df = over_grid_df.merge(cell_max_dist, on='cell_id', how='left')
over_grid_df['is_overshooting'] = over_grid_df['distance_to_cell'] >= over_grid_df['edge_threshold']
print(f"   ✅ Overshooting: {len(over_grid_df):,} grid measurements ({over_grid_df['is_overshooting'].sum():,} in far edge)")

# Undershooting grids - mark interference grids (multiple cells)
under_grid_df = coverage_df[coverage_df['cilac'].isin(undershooters_df['cell_id'])].copy()
under_grid_df = under_grid_df.rename(columns={
    'cilac': 'cell_id',
    'grid': 'geohash7',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

# Mark grids that appear in multiple cells as having interference
grid_cell_counts = under_grid_df.groupby('geohash7')['cell_id'].nunique().reset_index()
grid_cell_counts.columns = ['geohash7', 'num_cells']
under_grid_df = under_grid_df.merge(grid_cell_counts, on='geohash7', how='left')
under_grid_df['is_overshooting'] = under_grid_df['num_cells'] > 1  # Use same column name for consistency
print(f"   ✅ Undershooting: {len(under_grid_df):,} grid measurements ({under_grid_df['is_overshooting'].sum():,} with interference)")

# Create output directory
output_dir = Path('./data/output-data/vf-ie/recommendations/maps')
output_dir.mkdir(parents=True, exist_ok=True)

# Create separate subdirectories for each map's grids
overshooting_grids_dir = output_dir / 'grids_overshooting'
undershooting_grids_dir = output_dir / 'grids_undershooting'
overshooting_grids_dir.mkdir(exist_ok=True)
undershooting_grids_dir.mkdir(exist_ok=True)

# Generate overshooting map
print("\n🗺️  Generating overshooting map with enhanced metrics...")
start_time = time.time()

overshooting_output = output_dir / 'overshooting_enhanced.html'
overshooting_map = create_overshooting_map(
    overshooters_df=overshooters_df,
    grid_df=over_grid_df,
    gis_df=cell_info,
    show_sector_shapes=True,
    show_optimized_cells=True,
    output_file=overshooting_output
)
overshooting_map.save(str(overshooting_output))

elapsed = time.time() - start_time
file_size_mb = overshooting_output.stat().st_size / (1024 * 1024)
print(f"   ✅ Generated in {elapsed:.1f}s")
print(f"   📊 File size: {file_size_mb:.2f} MB")

# For undershooting, we need to adapt the data to match the overshooting format
print("\n🗺️  Preparing undershooting map with enhanced metrics...")

# Add required columns for undershooting (mimic overshooting structure)
undershooters_adapted = undershooters_df.copy()

# Map undershooting fields to overshooting-like fields for visualization
undershooters_adapted['severity_score'] = undershooters_adapted['coverage_increase_percentage']
undershooters_adapted['severity_category'] = undershooters_adapted['recommended_uptilt_deg'].apply(
    lambda x: 'HIGH' if x == 2 else 'MEDIUM'
)
undershooters_adapted['overshooting_grids'] = 0  # Not applicable
undershooters_adapted['percentage_overshooting'] = 0.0
undershooters_adapted['avg_edge_rsrp'] = -100  # Placeholder
undershooters_adapted['recommended_tilt_change'] = -undershooters_adapted['recommended_uptilt_deg']  # Negative for uptilt
undershooters_adapted['max_distance_m'] = undershooters_adapted['current_distance_m']

# Generate undershooting map
print("\n🗺️  Generating undershooting map with enhanced metrics...")
start_time = time.time()

undershooting_output = output_dir / 'undershooting_enhanced.html'
undershooting_map = create_overshooting_map(
    overshooters_df=undershooters_adapted,
    grid_df=under_grid_df,
    gis_df=cell_info,
    show_sector_shapes=True,
    show_optimized_cells=True,
    output_file=undershooting_output,
    map_type='undershooting'
)
undershooting_map.save(str(undershooting_output))

elapsed = time.time() - start_time
file_size_mb = undershooting_output.stat().st_size / (1024 * 1024)
print(f"   ✅ Generated in {elapsed:.1f}s")
print(f"   📊 File size: {file_size_mb:.2f} MB")

# Collect band and severity statistics
print("\n📊 Collecting filter statistics...")
over_bands = sorted(overshooters_df['band'].dropna().unique())
under_bands = sorted(undershooters_df['band'].dropna().unique())
all_bands = sorted(set(over_bands) | set(under_bands))

over_band_counts = overshooters_df['band'].value_counts().to_dict()
under_band_counts = undershooters_df['band'].value_counts().to_dict()

over_severity_counts = overshooters_df['severity_category'].value_counts().to_dict()
under_uptilt_counts = undershooters_df['recommended_uptilt_deg'].value_counts().to_dict()

print(f"   Bands found: {all_bands}")
print(f"   Overshooting severities: {list(over_severity_counts.keys())}")
print(f"   Undershooting uptilts: {list(under_uptilt_counts.keys())}")

# Now create unified HTML that embeds both maps
print("\n📝 Creating unified interface with enhanced metrics display...")

unified_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <title>RAN Optimization - Enhanced Metrics View</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }}
        #control-panel {{
            position: fixed;
            top: 20px;
            left: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            z-index: 10000;
            width: 320px;
            max-height: 90vh;
            overflow-y: auto;
        }}
        #control-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}
        .control-section {{
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .control-section:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
        .section-title {{
            font-size: 14px;
            font-weight: bold;
            color: #555;
            margin-bottom: 10px;
        }}
        .radio-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .radio-option {{
            display: flex;
            align-items: center;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            background: white;
        }}
        .radio-option:hover {{
            background: #f5f5f5;
            border-color: #2196F3;
        }}
        .radio-option input[type="radio"] {{
            margin-right: 10px;
            width: 18px;
            height: 18px;
            cursor: pointer;
            flex-shrink: 0;
        }}
        .radio-option.selected {{
            background: #E3F2FD;
            border-color: #2196F3;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
        }}
        .radio-label {{
            flex: 1;
            font-size: 14px;
            line-height: 1.3;
        }}
        .radio-count {{
            color: #666;
            font-size: 12px;
            display: block;
            margin-top: 2px;
            font-weight: normal;
        }}
        .info-box {{
            padding: 12px;
            background: #f0f8ff;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.6;
        }}
        .enhanced-badge {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 5px;
        }}
        .notice-box {{
            padding: 12px;
            background: #fff9e6;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            font-size: 11px;
            line-height: 1.4;
            color: #856404;
        }}
        .iframe-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .iframe-container iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .iframe-container.hidden {{
            display: none;
        }}
        .loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px 50px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            z-index: 20000;
            display: none;
            text-align: center;
        }}
        .loading.show {{
            display: block;
        }}
        .loading-spinner {{
            font-size: 32px;
            margin-bottom: 10px;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <!-- Overshooting map iframe -->
    <div id="overshooting-container" class="iframe-container">
        <iframe src="overshooting_enhanced.html"></iframe>
    </div>

    <!-- Undershooting map iframe -->
    <div id="undershooting-container" class="iframe-container hidden">
        <iframe src="undershooting_enhanced.html"></iframe>
    </div>

    <!-- Control panel -->
    <div id="control-panel">
        <h3>🗺️  RAN Optimization <span class="enhanced-badge">ENHANCED</span></h3>

        <!-- View Mode Selection -->
        <div class="control-section">
            <div class="section-title">View Mode</div>
            <div class="radio-group">
                <label class="radio-option selected" for="radio-overshooting">
                    <input type="radio" id="radio-overshooting" name="view-mode" value="overshooting" checked>
                    <span class="radio-label">
                        Overshooting Cells
                        <span class="radio-count">{len(overshooters_df)} cells • Enhanced metrics</span>
                    </span>
                </label>
                <label class="radio-option" for="radio-undershooting">
                    <input type="radio" id="radio-undershooting" name="view-mode" value="undershooting">
                    <span class="radio-label">
                        Undershooting Cells
                        <span class="radio-count">{len(undershooters_df)} cells • Enhanced metrics</span>
                    </span>
                </label>
            </div>
        </div>

        <!-- Current View Info -->
        <div class="control-section">
            <div class="info-box" id="info-box">
                <strong>📊 Overshooting Cells</strong><br>
                Cells serving too far beyond intended coverage.<br>
                <strong>Recommendation:</strong> Downtilt to reduce interference.<br>
                <br>
                <strong>✨ Enhanced Metrics:</strong><br>
                • Current vs. resolved interference grids<br>
                • Interference reduction percentage<br>
                • Before/after comparison
            </div>
        </div>

        <!-- Filter Instructions Notice -->
        <div class="control-section">
            <div class="notice-box">
                🗺️ <strong>Interactive Features:</strong><br>
                • <strong>Click cell markers</strong> to see detailed enhanced metrics<br>
                • Use <strong>Layers</strong> control (top-right) to filter by band/severity<br>
                • <strong>Load grids</strong> button in popups to show coverage bins<br>
                • All metrics calculated using RF propagation models
            </div>
        </div>
    </div>

    <!-- Loading indicator -->
    <div class="loading" id="loading">
        <div class="loading-spinner">⏳</div>
        <div>Switching view...</div>
    </div>

    <script>
        const overshootingContainer = document.getElementById('overshooting-container');
        const undershootingContainer = document.getElementById('undershooting-container');
        const infoBox = document.getElementById('info-box');
        const loading = document.getElementById('loading');

        function switchToOvershooting() {{
            loading.classList.add('show');

            setTimeout(() => {{
                overshootingContainer.classList.remove('hidden');
                undershootingContainer.classList.add('hidden');

                infoBox.innerHTML = `
                    <strong>📊 Overshooting Cells</strong><br>
                    Cells serving too far beyond intended coverage.<br>
                    <strong>Recommendation:</strong> Downtilt to reduce interference.<br>
                    <br>
                    <strong>✨ Enhanced Metrics:</strong><br>
                    • Current vs. resolved interference grids<br>
                    • Interference reduction percentage<br>
                    • Before/after comparison
                `;

                loading.classList.remove('show');
            }}, 300);
        }}

        function switchToUndershooting() {{
            loading.classList.add('show');

            setTimeout(() => {{
                undershootingContainer.classList.remove('hidden');
                overshootingContainer.classList.add('hidden');

                infoBox.innerHTML = `
                    <strong>📡 Undershooting Cells</strong><br>
                    Cells with insufficient coverage in low-interference areas.<br>
                    <strong>Recommendation:</strong> Uptilt to extend reach.<br>
                    <br>
                    <strong>✨ Enhanced Metrics:</strong><br>
                    • Current vs. new coverage grids<br>
                    • Coverage increase percentage<br>
                    • Distance gain in meters
                `;

                loading.classList.remove('show');
            }}, 300);
        }}

        function updateRadioStyles(groupName) {{
            document.querySelectorAll('input[name="' + groupName + '"]').forEach(radio => {{
                const option = radio.closest('.radio-option');
                if (option) {{
                    if (radio.checked) {{
                        option.classList.add('selected');
                    }} else {{
                        option.classList.remove('selected');
                    }}
                }}
            }});
        }}

        // View mode radio handlers
        document.querySelectorAll('input[name="view-mode"]').forEach(radio => {{
            radio.addEventListener('change', function() {{
                updateRadioStyles('view-mode');
                if (this.value === 'overshooting') {{
                    switchToOvershooting();
                }} else {{
                    switchToUndershooting();
                }}
            }});
        }});
    </script>
</body>
</html>
"""

# Save unified HTML
unified_file = output_dir / 'enhanced_metrics_map.html'
with open(unified_file, 'w') as f:
    f.write(unified_html)

file_size_kb = unified_file.stat().st_size / 1024
print(f"   ✅ Unified interface created")
print(f"   📁 Saved to: {unified_file}")
print(f"   📊 File size: {file_size_kb:.1f} KB")

print("\n" + "="*80)
print("✅ ENHANCED METRICS VISUALIZATION COMPLETE")
print("="*80)

print(f"\nGenerated files:")
print(f"   1. Unified interface: {unified_file}")
print(f"   2. Overshooting map: {overshooting_output}")
print(f"   3. Undershooting map: {undershooting_output}")

print(f"\nEnhanced Metrics Features:")
print(f"   ✓ Overshooting:")
print(f"     - Current/resolved/after interference grids")
print(f"     - Interference reduction percentage")
print(f"   ✓ Undershooting:")
print(f"     - Current/new/total coverage grids")
print(f"     - Coverage increase percentage")
print(f"     - Distance gain in meters")

print(f"\nTo view:")
print(f"   The web server is already running on port 8888")
print(f"   Access at: http://localhost:8888/enhanced_metrics_map.html")
