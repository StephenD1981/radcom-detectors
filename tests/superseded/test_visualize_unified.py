"""
Unified visualization for both overshooting and undershooting cells with radio button toggle.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import time
import json

# Import existing visualization
from ran_optimizer.visualization.map_overshooters import (
    create_overshooting_map,
    calculate_sector_points
)

print("\n" + "="*80)
print("🗺️  UNIFIED CELL OPTIMIZATION MAP VISUALIZATION")
print("="*80)

# Load overshooting cells
print("\n📂 Loading overshooting cells data...")
overshooters_file = Path('./data/output-data/vf-ie/recommendations/overshooting_cells_environment_aware.csv')
overshooters_df = pd.read_csv(overshooters_file)
print(f"   ✅ Loaded {len(overshooters_df):,} overshooting cells (environment-aware detection)")

env_breakdown_over = overshooters_df['environment'].value_counts()
print(f"\n   Environment breakdown:")
print(f"      URBAN     : {env_breakdown_over.get('URBAN', 0):3d} cells ({env_breakdown_over.get('URBAN', 0)/len(overshooters_df)*100:5.1f}%)")
print(f"      SUBURBAN  : {env_breakdown_over.get('SUBURBAN', 0):3d} cells ({env_breakdown_over.get('SUBURBAN', 0)/len(overshooters_df)*100:5.1f}%)")
print(f"      RURAL     : {env_breakdown_over.get('RURAL', 0):3d} cells ({env_breakdown_over.get('RURAL', 0)/len(overshooters_df)*100:5.1f}%)")

# Load undershooting cells
print("\n📂 Loading undershooting cells data...")
undershooters_file = Path('./data/output-data/vf-ie/recommendations/undershooting_cells_environment_aware.csv')
undershooters_df = pd.read_csv(undershooters_file)
print(f"   ✅ Loaded {len(undershooters_df):,} undershooting cells (environment-aware detection)")

env_breakdown_under = undershooters_df['environment'].value_counts()
print(f"\n   Environment breakdown:")
print(f"      URBAN     : {env_breakdown_under.get('URBAN', 0):3d} cells ({env_breakdown_under.get('URBAN', 0)/len(undershooters_df)*100:5.1f}%)")
print(f"      SUBURBAN  : {env_breakdown_under.get('SUBURBAN', 0):3d} cells ({env_breakdown_under.get('SUBURBAN', 0)/len(undershooters_df)*100:5.1f}%)")
print(f"      RURAL     : {env_breakdown_under.get('RURAL', 0):3d} cells ({env_breakdown_under.get('RURAL', 0)/len(undershooters_df)*100:5.1f}%)")

# Load cell coverage data for coordinates, azimuth, and grids
print("\n📂 Loading cell coverage data for coordinates, azimuth, and grids...")
start_time = time.time()
print("   Loading data (this may take a moment)...")
coverage_df = pd.read_csv(
    './data/output-data/vf-ie/recommendations/created_datasets/cell_coverage.csv',
    low_memory=False
)
elapsed = time.time() - start_time
print(f"   ✅ Loaded {len(coverage_df):,} measurements in {elapsed:.1f}s")

# Extract cell locations and azimuth
print("\n🔄 Extracting cell locations, azimuth, and band...")
cell_info = coverage_df.groupby('cilac').agg({
    'Latitude': 'first',
    'Longitude': 'first',
    'Bearing': 'first',
    'Band': 'first',
}).reset_index()
cell_info.columns = ['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']
print(f"   ✅ Got coordinates for {len(cell_info):,} cells")

# Merge coordinates with overshooting cells
overshooters_df = overshooters_df.merge(
    cell_info[['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']],
    on='cell_id',
    how='inner'
)
print(f"   ✅ Merged coordinates and azimuth for {len(overshooters_df)} overshooting cells")

# Merge coordinates with undershooting cells
undershooters_df = undershooters_df.merge(
    cell_info[['cell_id', 'latitude', 'longitude', 'azimuth_deg', 'band']],
    on='cell_id',
    how='inner'
)
print(f"   ✅ Merged coordinates and azimuth for {len(undershooters_df)} undershooting cells")

# Prepare grid data
print("\n🔄 Preparing grid bins for visualization...")
print("   Decoding geohash coordinates...")

from ran_optimizer.utils.geohash import get_box_bounds

# Get grid data for overshooting cells
over_grid_df = coverage_df[coverage_df['cilac'].isin(overshooters_df['cell_id'])].copy()
over_grid_df = over_grid_df.rename(columns={'cilac': 'cell_id', 'grid': 'geohash7'})

# Decode geohash to lat/lon
# get_box_bounds returns: (min_lat, max_lat, min_lon, max_lon)
grid_coords = []
for gh in over_grid_df['geohash7'].unique():
    min_lat, max_lat, min_lon, max_lon = get_box_bounds(gh)
    grid_coords.append({
        'geohash7': gh,
        'latitude': (min_lat + max_lat) / 2,
        'longitude': (min_lon + max_lon) / 2,
    })
grid_coords_df = pd.DataFrame(grid_coords)

# Merge with grid data
over_grid_df = over_grid_df.merge(grid_coords_df, on='geohash7', how='left')

print(f"   ✅ Prepared {len(over_grid_df):,} grid bins for overshooting cells")

# Get grid data for undershooting cells (we don't have overshooting classification for these)
# So we'll just show all grids for undershooters as "normal" (gray)
under_grid_df = coverage_df[coverage_df['cilac'].isin(undershooters_df['cell_id'])].copy()
under_grid_df = under_grid_df.rename(columns={'cilac': 'cell_id', 'grid': 'geohash7'})
under_grid_df = under_grid_df.merge(grid_coords_df, on='geohash7', how='left')

print(f"   ✅ Prepared {len(under_grid_df):,} grid bins for undershooting cells")

# Create output directory
output_dir = Path('./data/output-data/vf-ie/recommendations/maps')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("📊 DATA OVERVIEW")
print("="*80)

print(f"\nOvershooting cells with coordinates: {len(overshooters_df)}")
print(f"   Latitude range: {overshooters_df['latitude'].min():.4f} to {overshooters_df['latitude'].max():.4f}")
print(f"   Longitude range: {overshooters_df['longitude'].min():.4f} to {overshooters_df['longitude'].max():.4f}")

severity_counts = overshooters_df['severity_category'].value_counts()
print(f"\n   Severity breakdown:")
for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
    count = severity_counts.get(sev, 0)
    print(f"      {sev:8s}: {count:3d} cells")

print(f"\nUndershooting cells with coordinates: {len(undershooters_df)}")
print(f"   Latitude range: {undershooters_df['latitude'].min():.4f} to {undershooters_df['latitude'].max():.4f}")
print(f"   Longitude range: {undershooters_df['longitude'].min():.4f} to {undershooters_df['longitude'].max():.4f}")

uptilt_counts = undershooters_df['recommended_uptilt_deg'].value_counts()
print(f"\n   Recommended uptilt breakdown:")
for deg in sorted(uptilt_counts.index):
    count = uptilt_counts[deg]
    print(f"      {deg:.0f}° uptilt: {count:3d} cells")

# Generate overshooting map
print("\n🗺️  Generating overshooting cells map...")
start_time = time.time()

overshooting_map = create_overshooting_map(
    overshooters_df=overshooters_df,
    grid_df=over_grid_df,
    gis_df=None,
    show_sector_shapes=True,
    output_file=output_dir / 'temp_overshooting_map.html'
)

elapsed = time.time() - start_time
overshooting_html_file = output_dir / 'temp_overshooting_map.html'
overshooting_map.save(str(overshooting_html_file))

file_size_mb = overshooting_html_file.stat().st_size / (1024 * 1024)
print(f"   ✅ Overshooting map generated in {elapsed:.1f}s")
print(f"   📁 Saved to: {overshooting_html_file}")
print(f"   📊 File size: {file_size_mb:.2f} MB")

print("\n📝 Creating unified map with radio button toggle...")
print("   This will combine both datasets into a single interactive map...")

# Read the overshooting HTML to extract the map div and leaflet code
with open(overshooting_html_file, 'r') as f:
    overshooting_html = f.read()

# Create unified HTML with radio button toggle
unified_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <title>RAN Optimization Map - Overshooting & Undershooting Cells</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #control-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 250px;
        }}
        #control-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 16px;
            color: #333;
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
        }}
        .radio-option.selected {{
            background: #E3F2FD;
            border-color: #2196F3;
            font-weight: bold;
        }}
        .radio-label {{
            flex: 1;
        }}
        .radio-count {{
            color: #666;
            font-size: 13px;
            margin-left: 8px;
        }}
        #map {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .info-box {{
            margin-top: 15px;
            padding: 10px;
            background: #f0f8ff;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.6;
        }}
        .loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 2000;
            display: none;
        }}
        .loading.show {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div id="control-panel">
        <h3>🗺️  View Selection</h3>
        <div class="radio-group">
            <label class="radio-option selected" for="radio-overshooting">
                <input type="radio" id="radio-overshooting" name="view-mode" value="overshooting" checked>
                <span class="radio-label">
                    Overshooting Cells
                    <span class="radio-count">({len(overshooters_df)} cells)</span>
                </span>
            </label>
            <label class="radio-option" for="radio-undershooting">
                <input type="radio" id="radio-undershooting" name="view-mode" value="undershooting">
                <span class="radio-label">
                    Undershooting Cells
                    <span class="radio-count">({len(undershooters_df)} cells)</span>
                </span>
            </label>
        </div>

        <div class="info-box" id="info-box">
            <strong>Overshooting Cells</strong><br>
            Cells serving too far beyond intended coverage area.<br>
            <strong>Recommendation:</strong> Downtilt to reduce interference.
        </div>
    </div>

    <div class="loading" id="loading">
        <div style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <div>Switching view...</div>
        </div>
    </div>

    <script>
        // Store data for both modes
        const overshootingData = {json.dumps([row.to_dict() for _, row in overshooters_df.iterrows()])};
        const undershootingData = {json.dumps([row.to_dict() for _, row in undershooters_df.iterrows()])};

        // Calculate center
        const centerLat = {overshooters_df['latitude'].mean()};
        const centerLon = {overshooters_df['longitude'].mean()};

        // Create map
        const map = L.map('map').setView([centerLat, centerLon], 10);

        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Layer groups
        let currentMarkers = L.layerGroup().addTo(map);

        // Severity colors
        const severityColors = {{
            'CRITICAL': '#8B0000',
            'HIGH': '#FF0000',
            'MEDIUM': '#FF6600',
            'LOW': '#FFA500',
            'MINIMAL': '#FFD700'
        }};

        function createOvershootingMarker(cell) {{
            const color = severityColors[cell.severity_category] || '#999';

            const popupHtml = `
                <div style="font-family: Arial; width: 300px;">
                    <h3 style="margin: 0 0 10px 0;">Cell ${{cell.cell_id}}</h3>
                    <div style="background: #f0f0f0; padding: 8px; border-radius: 4px;">
                        <strong>Severity:</strong> ${{cell.severity_category}}<br>
                        <strong>Environment:</strong> ${{cell.environment}}<br>
                        <strong>Max Distance:</strong> ${{Math.round(cell.max_distance_m)}}m<br>
                        <strong>Recommended:</strong> ${{cell.recommended_tilt_change}}° downtilt
                    </div>
                </div>
            `;

            return L.circleMarker([cell.latitude, cell.longitude], {{
                radius: 8,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).bindPopup(popupHtml);
        }}

        function createUndershootingMarker(cell) {{
            const color = cell.recommended_uptilt_deg === 2 ? '#0066CC' : '#4D94FF';

            const popupHtml = `
                <div style="font-family: Arial; width: 300px;">
                    <h3 style="margin: 0 0 10px 0;">Cell ${{cell.cell_id}}</h3>
                    <div style="background: #f0f0f0; padding: 8px; border-radius: 4px;">
                        <strong>Environment:</strong> ${{cell.environment}}<br>
                        <strong>Max Distance:</strong> ${{Math.round(cell.max_distance_m)}}m<br>
                        <strong>Coverage Increase:</strong> ${{(cell.coverage_increase_percentage * 100).toFixed(1)}}%<br>
                        <strong>Recommended:</strong> ${{cell.recommended_uptilt_deg}}° uptilt
                    </div>
                </div>
            `;

            return L.circleMarker([cell.latitude, cell.longitude], {{
                radius: 8,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).bindPopup(popupHtml);
        }}

        function switchToOvershooting() {{
            currentMarkers.clearLayers();
            overshootingData.forEach(cell => {{
                if (cell.latitude && cell.longitude) {{
                    const marker = createOvershootingMarker(cell);
                    currentMarkers.addLayer(marker);
                }}
            }});

            document.getElementById('info-box').innerHTML = `
                <strong>Overshooting Cells</strong><br>
                Cells serving too far beyond intended coverage area.<br>
                <strong>Recommendation:</strong> Downtilt to reduce interference.
            `;
        }}

        function switchToUndershooting() {{
            currentMarkers.clearLayers();
            undershootingData.forEach(cell => {{
                if (cell.latitude && cell.longitude) {{
                    const marker = createUndershootingMarker(cell);
                    currentMarkers.addLayer(marker);
                }}
            }});

            document.getElementById('info-box').innerHTML = `
                <strong>Undershooting Cells</strong><br>
                Cells with insufficient coverage in low-interference areas.<br>
                <strong>Recommendation:</strong> Uptilt to extend reach.
            `;
        }}

        // Initialize with overshooting view
        switchToOvershooting();

        // Radio button handlers
        document.querySelectorAll('input[name="view-mode"]').forEach(radio => {{
            radio.addEventListener('change', function() {{
                // Update selected styling
                document.querySelectorAll('.radio-option').forEach(opt => {{
                    opt.classList.remove('selected');
                }});
                this.closest('.radio-option').classList.add('selected');

                // Show loading
                document.getElementById('loading').classList.add('show');

                // Switch view
                setTimeout(() => {{
                    if (this.value === 'overshooting') {{
                        switchToOvershooting();
                    }} else {{
                        switchToUndershooting();
                    }}
                    document.getElementById('loading').classList.remove('show');
                }}, 100);
            }});
        }});
    </script>
</body>
</html>
"""

# Save unified map
unified_file = output_dir / 'unified_optimization_map.html'
with open(unified_file, 'w') as f:
    f.write(unified_html)

file_size_mb = unified_file.stat().st_size / (1024 * 1024)
print(f"   ✅ Unified map generated")
print(f"   📁 Saved to: {unified_file}")
print(f"   📊 File size: {file_size_mb:.2f} MB")

# Clean up temp file
overshooting_html_file.unlink()

print("\n" + "="*80)
print("✅ VISUALIZATION GENERATION COMPLETE")
print("="*80)

print(f"\nGenerated file:")
print(f"   1. Unified map: {unified_file}")

print(f"\nFeatures:")
print(f"   ✓ Radio button toggle between Overshooting and Undershooting views")
print(f"   ✓ {len(overshooters_df)} overshooting cells (color-coded by severity)")
print(f"   ✓ {len(undershooters_df)} undershooting cells (color-coded by uptilt recommendation)")
print(f"   ✓ Interactive popups with cell details")
print(f"   ✓ Environment-aware detection results")

print(f"\nTo view:")
print(f"   Open the HTML file in your web browser")
print(f"   • Use radio buttons to switch between views")
print(f"   • Click markers to see cell details")
print(f"   • Zoom and pan to explore the network")
