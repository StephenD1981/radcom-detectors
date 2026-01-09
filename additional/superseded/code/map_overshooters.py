"""
Interactive map visualization for overshooting cells using Folium.

Creates HTML maps with:
- Triangular sector markers showing cell bearing
- Lazy-loaded grid bins (loaded on-demand when cell is clicked)
"""
import folium
from folium import plugins
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.geohash import get_box_bounds

logger = get_logger(__name__)


def calculate_sector_points(
    lat: float,
    lon: float,
    bearing: float,
    radius_m: float,
    sector_width: float = 40.0
) -> List[Tuple[float, float]]:
    """
    Calculate points for a triangular sector (cell antenna pattern).

    Args:
        lat: Cell latitude
        lon: Cell longitude
        bearing: Antenna azimuth in degrees (0=North, 90=East)
        radius_m: Sector radius in meters
        sector_width: Total sector width in degrees (default 40 = bearing ±20°)

    Returns:
        List of (lat, lon) tuples forming the triangle
    """
    # Earth radius in meters
    earth_radius = 6371000

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Half sector width
    half_width = sector_width / 2.0

    # Three angles: center, left edge, right edge
    angles = [
        bearing,                # Center (apex of triangle)
        bearing - half_width,   # Left edge
        bearing + half_width,   # Right edge
    ]

    points = [(lat, lon)]  # Start with cell location (apex)

    # Calculate points at the arc (left and right edges)
    for angle in angles[1:]:  # Skip center, just do left and right
        angle_rad = np.radians(angle)

        # Calculate destination point
        lat2 = np.arcsin(
            np.sin(lat_rad) * np.cos(radius_m / earth_radius) +
            np.cos(lat_rad) * np.sin(radius_m / earth_radius) * np.cos(angle_rad)
        )

        lon2 = lon_rad + np.arctan2(
            np.sin(angle_rad) * np.sin(radius_m / earth_radius) * np.cos(lat_rad),
            np.cos(radius_m / earth_radius) - np.sin(lat_rad) * np.sin(lat2)
        )

        points.append((np.degrees(lat2), np.degrees(lon2)))

    return points


def create_popup_html(cell: pd.Series, grid_data_path: Optional[str] = None) -> str:
    """
    Create rich HTML popup for cell marker.

    Args:
        cell: Row from overshooters DataFrame
        grid_data_path: Relative path to grid data JSON file

    Returns:
        HTML string for popup
    """
    cell_name = cell['cell_name']

    html = f"""
    <div style="font-family: Arial; width: 300px;">
        <h3 style="margin: 0 0 10px 0; color: #333;">Cell {cell_name}</h3>

        <div style="background: #f0f0f0; padding: 8px; margin-bottom: 8px; border-radius: 4px;">
            <strong>Severity:</strong> {cell['severity_category']}<br>
            <strong>Score:</strong> {cell['severity_score']:.3f}
        </div>

        <div style="margin-bottom: 8px;">
            <strong>Overshooting:</strong><br>
            &nbsp;&nbsp;• Bins: {cell['overshooting_grids']:,} ({cell['percentage_overshooting']*100:.1f}%)<br>
            &nbsp;&nbsp;• Total grids: {cell['total_grids']:,}<br>
            &nbsp;&nbsp;• Max distance: {cell['max_distance_m']/1000:.1f} km
        </div>

        <div style="margin-bottom: 8px;">
            <strong>Signal Quality:</strong><br>
            &nbsp;&nbsp;• Avg edge RSRP: {cell['avg_edge_rsrp']:.1f} dBm
        </div>

        <div style="margin-bottom: 8px;">
            <strong>Environment:</strong><br>
            &nbsp;&nbsp;• Type: {cell.get('environment', 'N/A')}<br>
            &nbsp;&nbsp;• Intersite distance: {cell.get('intersite_distance_km', 0):.2f} km
        </div>

        <div style="margin-bottom: 8px;">
            <strong>Current Config:</strong><br>
            &nbsp;&nbsp;• Band: {cell.get('band', 'N/A')} MHz<br>
            &nbsp;&nbsp;• Azimuth: {cell.get('azimuth_deg', 'N/A')}°<br>
            &nbsp;&nbsp;• Mechanical tilt: {cell['mechanical_tilt']:.1f}°<br>
            &nbsp;&nbsp;• Electrical tilt: {cell['electrical_tilt']:.1f}°
        </div>

        <div style="background: #fff3cd; padding: 8px; border-radius: 4px; margin-bottom: 8px;">
            <strong>Recommendation:</strong> +{cell['recommended_tilt_change']:.1f}° tilt increase
        </div>
    """

    # Add enhanced metrics section for overshooting
    if 'current_interference_grids' in cell and cell['recommended_tilt_change'] > 0:
        # Get the new max distance for the recommended tilt
        if cell['recommended_tilt_change'] == 1:
            new_max_dist = cell.get('new_max_distance_1deg_m', cell['max_distance_m'])
        else:
            new_max_dist = cell.get('new_max_distance_2deg_m', cell['max_distance_m'])

        html += f"""
        <div style="background: #e7f3ff; padding: 8px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid #2196F3;">
            <strong>📊 Interference Reduction Impact:</strong><br>
            &nbsp;&nbsp;• Current: {cell['current_interference_grids']:.0f} grids ({cell['current_interference_pct']*100:.1f}%)<br>
            &nbsp;&nbsp;• Will resolve: {cell['removed_interference_grids']:.0f} grids<br>
            &nbsp;&nbsp;• After downtilt: {cell['new_interference_grids']:.0f} grids ({cell['new_interference_pct']*100:.1f}%)<br>
            &nbsp;&nbsp;• <strong>Reduction: {cell['interference_reduction_pct']:.1f}%</strong><br>
            <br>
            <strong>📏 Coverage Distance Change:</strong><br>
            &nbsp;&nbsp;• Current max: {cell['max_distance_m']/1000:.2f} km<br>
            &nbsp;&nbsp;• New max: {new_max_dist/1000:.2f} km<br>
            &nbsp;&nbsp;• Reduction: {(cell['max_distance_m'] - new_max_dist):.0f} m
        </div>
        """

    # Add enhanced metrics section for undershooting
    elif 'current_coverage_grids' in cell and cell['recommended_tilt_change'] < 0:
        uptilt = abs(cell['recommended_tilt_change'])
        current_dist = cell.get('current_distance_m', cell['max_distance_m'])
        new_dist = cell.get('new_max_distance_m', cell['max_distance_m'])

        html += f"""
        <div style="background: #e8f5e9; padding: 8px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid #4CAF50;">
            <strong>📡 Coverage Expansion Impact:</strong><br>
            &nbsp;&nbsp;• Current: {cell['current_coverage_grids']:.0f} grids<br>
            &nbsp;&nbsp;• Will gain: {cell['new_coverage_grids']:.0f} grids<br>
            &nbsp;&nbsp;• After uptilt: {cell['total_coverage_after_uptilt']:.0f} grids<br>
            &nbsp;&nbsp;• <strong>Increase: {cell.get('coverage_increase_percentage', 0)*100:.1f}%</strong><br>
            <br>
            <strong>📏 Coverage Distance Change:</strong><br>
            &nbsp;&nbsp;• Current max: {current_dist/1000:.2f} km<br>
            &nbsp;&nbsp;• New max: {new_dist/1000:.2f} km<br>
            &nbsp;&nbsp;• Extension: +{cell['distance_increase_m']:.0f} m
        </div>
        """

    html += """
    """

    if grid_data_path:
        html += f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="gridBtn_{cell_name}" onclick="toggleGrids_{cell_name}()"
                    style="background: #007bff; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Load & Show Grids
            </button>
            <div id="gridStatus_{cell_name}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
        </div>
        """

    html += "</div>"
    return html


def save_grid_data_files(
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Save grid data as separate JSON files per cell for lazy loading.

    Args:
        grid_df: DataFrame with grid bins
        output_dir: Directory to save JSON files

    Returns:
        Dict mapping cell_name to relative JSON file path
    """
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_paths = {}

    # OPTIMIZATION: Precompute geohash bounds for all unique geohashes at once
    unique_geohashes = grid_df['geohash7'].unique()
    geohash_bounds = {}
    for gh in unique_geohashes:
        min_lat, max_lat, min_lon, max_lon = get_box_bounds(gh)
        geohash_bounds[gh] = [[min_lat, min_lon], [max_lat, max_lon]]

    # OPTIMIZATION: Pre-process columns to avoid repeated .get() calls
    # Check for both overshooting and interference flags (for undershooting grids)
    has_is_overshooting = 'is_overshooting' in grid_df.columns
    has_is_interfering = 'is_interfering' in grid_df.columns  # Used by undershooting grids
    has_band = 'Band' in grid_df.columns

    # Group by cell_name and process each group
    for cell_name, cell_grids in grid_df.groupby('cell_name'):
        # OPTIMIZATION: Use vectorized operations instead of iterrows
        grids_list = []

        # Extract arrays for faster iteration
        geohashes = cell_grids['geohash7'].values
        latitudes = cell_grids['latitude'].values
        longitudes = cell_grids['longitude'].values
        # Use is_overshooting for overshooters, is_interfering for undershooters
        if has_is_overshooting:
            highlighted = cell_grids['is_overshooting'].values
        elif has_is_interfering:
            highlighted = cell_grids['is_interfering'].values
        else:
            highlighted = [False] * len(cell_grids)
        bands = cell_grids['Band'].values if has_band else [0] * len(cell_grids)

        for i in range(len(cell_grids)):
            geohash = geohashes[i]
            band_val = bands[i]

            grids_list.append({
                'lat': float(latitudes[i]),
                'lon': float(longitudes[i]),
                'hash': geohash,
                'overshoot': bool(highlighted[i]),
                'band': int(band_val) if pd.notna(band_val) else 0,
                'bounds': geohash_bounds[geohash]
            })

        grid_data = {
            'cell_name': int(cell_name),
            'grids': grids_list
        }

        # Save to JSON file
        json_file = grid_dir / f"cell_{cell_name}_grids.json"
        with open(json_file, 'w') as f:
            json.dump(grid_data, f, separators=(',', ':'))  # Compact format

        # Store relative path
        grid_paths[cell_name] = f"grids/cell_{cell_name}_grids.json"

    logger.info(
        "Saved grid data files",
        num_cells=len(grid_paths),
        output_dir=str(grid_dir),
    )

    return grid_paths


def create_overshooting_map(
    overshooters_df: pd.DataFrame,
    grid_df: Optional[pd.DataFrame] = None,
    gis_df: Optional[pd.DataFrame] = None,
    show_sector_shapes: bool = True,
    show_optimized_cells: bool = False,
    output_file: Optional[Path] = None,
    map_type: str = 'overshooting',
) -> folium.Map:
    """
    Create interactive Folium map of overshooting/undershooting cells with lazy-loaded grids.

    Args:
        overshooters_df: DataFrame with overshooting cells and severity scores
        grid_df: Optional DataFrame with grid bins (will be saved as separate JSON files)
        gis_df: Optional DataFrame with all cells
        show_sector_shapes: If True, show triangular sectors for cells
        show_optimized_cells: If True, show optimized cells in green
        output_file: Optional path to save HTML file
        map_type: Type of map ('overshooting' or 'undershooting') for appropriate labeling

    Returns:
        folium.Map object
    """
    logger.info(
        "Creating overshooting map with lazy-loaded grids",
        overshooters=len(overshooters_df),
        show_sectors=show_sector_shapes,
        has_grid_data=grid_df is not None,
    )

    # Validate required columns
    required_cols = {
        'cell_name', 'latitude', 'longitude', 'severity_score', 'severity_category',
        'overshooting_grids', 'percentage_overshooting', 'max_distance_m',
        'avg_edge_rsrp', 'mechanical_tilt', 'electrical_tilt',
        'recommended_tilt_change', 'total_grids'
    }
    missing_cols = required_cols - set(overshooters_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if 'azimuth_deg' not in overshooters_df.columns:
        logger.warning("azimuth_deg column missing - sectors will point North")
        overshooters_df = overshooters_df.copy()
        overshooters_df['azimuth_deg'] = 0.0

    # Save grid data to separate JSON files
    grid_paths = {}
    if grid_df is not None and output_file:
        output_dir = Path(output_file).parent
        grid_paths = save_grid_data_files(grid_df, output_dir)
        logger.info("Grid data saved for lazy loading", cells_with_grids=len(grid_paths))

    # Calculate map center
    center_lat = overshooters_df['latitude'].mean()
    center_lon = overshooters_df['longitude'].mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap',
        control_scale=True,
    )

    # Color mapping for severity (sectors)
    severity_colors = {
        'CRITICAL': '#dc3545',
        'HIGH': '#fd7e14',
        'MEDIUM': '#ffc107',
        'LOW': '#0dcaf0',
        'MINIMAL': '#6c757d',
    }

    # Color mapping for environment (markers)
    environment_colors = {
        'URBAN': '#0066ff',     # Blue
        'SUBURBAN': '#ff9900',  # Orange
        'RURAL': '#ff0000',     # Red
    }

    # Feature groups
    sectors_layer = folium.FeatureGroup(name='Cell Sectors (Triangles)', show=True)
    markers_layer = folium.FeatureGroup(name='Cell Markers', show=True)
    optimized_layer = folium.FeatureGroup(name='Background Cells', show=True)

    severity_counts = overshooters_df['severity_category'].value_counts()

    # Plot overshooting cells
    for idx, cell in overshooters_df.iterrows():
        # Sector color = severity, Marker color = environment
        sector_color = severity_colors.get(cell['severity_category'], '#6c757d')
        marker_color = environment_colors.get(cell.get('environment', 'SUBURBAN'), '#ff9900')
        cell_name = cell['cell_name']
        radius = 5 + (cell['severity_score'] * 12)
        cell_band = cell.get('band', 0)
        cell_env = cell.get('environment', 'SUBURBAN')

        # Create popup HTML (shared between sector and marker)
        grid_path = grid_paths.get(cell_name)
        popup_html = create_popup_html(cell, grid_data_path=grid_path)

        # 1. Draw triangular sector (fixed 500m radius)
        if show_sector_shapes:
            sector_points = calculate_sector_points(
                lat=cell['latitude'],
                lon=cell['longitude'],
                bearing=cell['azimuth_deg'],
                radius_m=500,  # Fixed 500m sector size
                sector_width=40.0
            )

            sector = folium.Polygon(
                locations=sector_points,
                color=sector_color,
                weight=2,
                fill=True,
                fillColor=sector_color,
                fillOpacity=0.2,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Cell {cell_name} - {cell['severity_category']} ({cell['severity_score']:.3f})",
                class_name=f"cell-sector band-{cell_band} severity-{cell['severity_category']}",
            )
            sector.add_to(sectors_layer)

        # 2. Draw center marker (colored by environment)
        marker = folium.CircleMarker(
            location=[cell['latitude'], cell['longitude']],
            radius=radius,
            color=marker_color,
            fill=True,
            fillColor=marker_color,
            fillOpacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Cell {cell_name} - {cell_env} - {cell['severity_category']} ({cell['severity_score']:.3f})",
            class_name=f"cell-marker band-{cell_band} severity-{cell['severity_category']}",
        )
        marker.add_to(markers_layer)

    logger.info("Plotted overshooting cells", markers=len(overshooters_df))

    # Plot all other cells as grey background for context
    if show_optimized_cells and gis_df is not None:
        problem_cell_names = set(overshooters_df['cell_name'].unique())
        other_cells = gis_df[~gis_df['cell_name'].isin(problem_cell_names)]

        logger.info(f"Adding {len(other_cells)} background cells for context")

        for idx, cell in other_cells.iterrows():
            folium.CircleMarker(
                location=[cell['latitude'], cell['longitude']],
                radius=2,
                color='#808080',  # Grey
                fill=True,
                fillColor='#808080',  # Grey
                fillOpacity=0.3,  # Transparent
                weight=0.5,
                popup=f"Cell {cell['cell_name']}<br>Status: Normal",
            ).add_to(optimized_layer)

    # Add layers
    sectors_layer.add_to(m)
    markers_layer.add_to(m)
    if show_optimized_cells:
        optimized_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Add JavaScript for lazy loading grids
    if grid_paths:
        # Get the Folium map variable name
        map_var_name = m.get_name()

        # Set label text based on map type
        if map_type == 'undershooting':
            highlighted_label = 'HIGH INTERFERENCE'
            normal_label = 'Low Interference'
        else:  # overshooting
            highlighted_label = 'OVERSHOOTING'
            normal_label = 'Normal'

        grid_js = f"""
        <script>
        // Store loaded grids by cell_name
        var loadedGrids = {{}};
        var gridLayers = {{}};

        // Function to load and display grids for a cell
        function loadGridsForCell(cellId, gridPath) {{
            var statusDiv = document.getElementById('gridStatus_' + cellId);
            var btn = document.getElementById('gridBtn_' + cellId);

            if (loadedGrids[cellId]) {{
                // Already loaded, just toggle visibility
                var layer = gridLayers[cellId];
                if (layer) {{
                    if (layer._map) {{
                        {map_var_name}.removeLayer(layer);
                        btn.textContent = 'Show Grids';
                    }} else {{
                        {map_var_name}.addLayer(layer);
                        btn.textContent = 'Hide Grids';
                    }}
                }}
                return;
            }}

            // Load from JSON
            statusDiv.textContent = 'Loading...';
            btn.disabled = true;

            fetch(gridPath)
                .then(response => response.json())
                .then(data => {{
                    // Create layer group for this cell's grids
                    var gridLayer = L.layerGroup();

                    data.grids.forEach(function(grid) {{
                        var color = grid.overshoot ? '#dc3545' : '#6c757d';

                        // Draw as rectangle using geohash bounds
                        var bounds = grid.bounds;
                        var rectangle = L.rectangle(bounds, {{
                            color: color,
                            fillColor: color,
                            fillOpacity: 0.4,
                            weight: 1
                        }});
                        rectangle.bindPopup(
                            'Grid: ' + grid.hash + '<br>' +
                            (grid.overshoot ? '<b>{highlighted_label}</b>' : '{normal_label}')
                        );
                        rectangle.addTo(gridLayer);
                    }});

                    {map_var_name}.addLayer(gridLayer);
                    loadedGrids[cellId] = true;
                    gridLayers[cellId] = gridLayer;

                    statusDiv.textContent = data.grids.length + ' grids loaded';
                    btn.textContent = 'Hide Grids';
                    btn.disabled = false;
                }})
                .catch(error => {{
                    statusDiv.textContent = 'Error loading grids';
                    btn.disabled = false;
                    console.error('Error loading grids:', error);
                }});
        }}
        """

        # Add toggle functions for each cell
        for cell_name, grid_path in grid_paths.items():
            grid_js += f"""
        function toggleGrids_{cell_name}() {{
            loadGridsForCell({cell_name}, '{grid_path}');
        }}
        """

        grid_js += "</script>"
        m.get_root().html.add_child(folium.Element(grid_js))

    # Get unique bands and severities
    unique_bands = sorted(overshooters_df['band'].dropna().unique()) if 'band' in overshooters_df.columns else []
    unique_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']  # Ordered list

    # Frequency Band Filter UI
    band_filter_html = """
    <div style="
        position: fixed; top: 80px; right: 50px; width: 220px;
        background-color: white; border: 2px solid grey; border-radius: 5px;
        z-index: 9999; font-size: 12px; padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 8px 0;">Frequency Band Filter</h4>
        <div style="margin-bottom: 5px;">
            <label style="display: block; cursor: pointer;">
                <input type="radio" name="bandFilter" value="all" checked onchange="applyFilters()">
                <strong>All Bands</strong>
            </label>
        </div>
    """

    # Add radio buttons for each band
    for band in unique_bands:
        band_count = len(overshooters_df[overshooters_df['band'] == band])
        band_filter_html += f"""
        <div style="margin-bottom: 5px;">
            <label style="display: block; cursor: pointer;">
                <input type="radio" name="bandFilter" value="{band}" onchange="applyFilters()">
                {band} MHz ({band_count} cells)
            </label>
        </div>
        """

    band_filter_html += """
    </div>
    """

    # Severity Filter UI
    severity_filter_html = """
    <div style="
        position: fixed; top: 320px; right: 50px; width: 220px;
        background-color: white; border: 2px solid grey; border-radius: 5px;
        z-index: 9999; font-size: 12px; padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 8px 0;">Severity Filter</h4>
        <div style="margin-bottom: 5px;">
            <label style="display: block; cursor: pointer;">
                <input type="radio" name="severityFilter" value="all" checked onchange="applyFilters()">
                <strong>All Severities</strong>
            </label>
        </div>
    """

    # Add radio buttons for each severity
    for severity in unique_severities:
        severity_count = severity_counts.get(severity, 0)
        if severity_count > 0:  # Only show severities that exist
            severity_filter_html += f"""
            <div style="margin-bottom: 5px;">
                <label style="display: block; cursor: pointer;">
                    <input type="radio" name="severityFilter" value="{severity}" onchange="applyFilters()">
                    {severity} ({severity_count} cells)
                </label>
            </div>
            """

    severity_filter_html += """
    </div>
    """

    # Combined filter JavaScript
    filter_js = """
    <script>
    function applyFilters() {
        // Get selected band
        var selectedBand = document.querySelector('input[name="bandFilter"]:checked').value;

        // Get selected severity
        var selectedSeverity = document.querySelector('input[name="severityFilter"]:checked').value;

        // Get all cell elements (both sectors and markers)
        var allCells = document.querySelectorAll('.cell-sector, .cell-marker');

        allCells.forEach(function(cell) {
            var showCell = true;

            // Band filter
            if (selectedBand !== 'all') {
                if (!cell.classList.contains('band-' + selectedBand)) {
                    showCell = false;
                }
            }

            // Severity filter
            if (selectedSeverity !== 'all') {
                if (!cell.classList.contains('severity-' + selectedSeverity)) {
                    showCell = false;
                }
            }

            // Apply visibility
            cell.style.display = showCell ? '' : 'none';
        });
    }
    </script>
    """

    # Get environment counts
    env_counts = overshooters_df.get('environment', pd.Series()).value_counts() if 'environment' in overshooters_df.columns else {}

    # Legend
    legend_html = f"""
    <div style="
        position: fixed; bottom: 50px; left: 50px; width: 280px;
        background-color: white; border: 2px solid grey; border-radius: 5px;
        z-index: 9999; font-size: 12px; padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 8px 0;">Legend</h4>

        <p style="margin: 4px 0; font-size: 11px;"><strong>Markers (Environment):</strong></p>
        <p style="margin: 2px 0;"><span style="color: {environment_colors['URBAN']};">●</span> URBAN ({env_counts.get('URBAN', 0)}) - ≤1km intersite</p>
        <p style="margin: 2px 0;"><span style="color: {environment_colors['SUBURBAN']};">●</span> SUBURBAN ({env_counts.get('SUBURBAN', 0)}) - 1-3km intersite</p>
        <p style="margin: 2px 0;"><span style="color: {environment_colors['RURAL']};">●</span> RURAL ({env_counts.get('RURAL', 0)}) - ≥3km intersite</p>

        <hr style="margin: 8px 0;">

        <p style="margin: 4px 0; font-size: 11px;"><strong>Triangles (Severity):</strong></p>
        <p style="margin: 2px 0;"><span style="color: {severity_colors['CRITICAL']};">▲</span> CRITICAL ({severity_counts.get('CRITICAL', 0)})</p>
        <p style="margin: 2px 0;"><span style="color: {severity_colors['HIGH']};">▲</span> HIGH ({severity_counts.get('HIGH', 0)})</p>
        <p style="margin: 2px 0;"><span style="color: {severity_colors['MEDIUM']};">▲</span> MEDIUM ({severity_counts.get('MEDIUM', 0)})</p>
        <p style="margin: 2px 0;"><span style="color: {severity_colors['LOW']};">▲</span> LOW ({severity_counts.get('LOW', 0)})</p>
        <p style="margin: 2px 0;"><span style="color: {severity_colors['MINIMAL']};">▲</span> MINIMAL ({severity_counts.get('MINIMAL', 0)})</p>

        <hr style="margin: 8px 0;">
        <p style="margin: 4px 0; font-size: 10px; color: #666;">
            Click cell → Load grids on-demand<br>
            <span style="color: #dc3545;">●</span> Red grid = Overshooting<br>
            <span style="color: #6c757d;">●</span> Gray grid = Normal
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add band filter
    if unique_bands:
        m.get_root().html.add_child(folium.Element(band_filter_html))

    # Add severity filter
    m.get_root().html.add_child(folium.Element(severity_filter_html))

    # Add combined filter JavaScript
    m.get_root().html.add_child(folium.Element(filter_js))

    # Title
    title_html = """
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        background-color: white; border: 2px solid grey; border-radius: 5px;
        z-index: 9999; padding: 10px 20px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <h3 style="margin: 0; color: #333;">Overshooting Cells - VF Ireland (Lazy-Loaded Grids)</h3>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    plugins.Fullscreen().add_to(m)

    # Save
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_file))
        logger.info(
            "Map saved",
            output_file=str(output_file),
            file_size_mb=output_file.stat().st_size / (1024**2),
        )

    return m


def create_severity_heatmap(
    overshooters_df: pd.DataFrame,
    output_file: Optional[Path] = None,
) -> folium.Map:
    """Create heatmap visualization of severity scores."""
    logger.info("Creating severity heatmap", overshooters=len(overshooters_df))

    center_lat = overshooters_df['latitude'].mean()
    center_lon = overshooters_df['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='CartoDB dark_matter',
    )

    heat_data = [
        [row['latitude'], row['longitude'], row['severity_score']]
        for idx, row in overshooters_df.iterrows()
    ]

    plugins.HeatMap(
        heat_data,
        name='Severity Heatmap',
        min_opacity=0.3,
        max_zoom=13,
        radius=15,
        blur=20,
        gradient={
            0.0: 'blue',
            0.2: 'cyan',
            0.4: 'lime',
            0.6: 'yellow',
            0.8: 'orange',
            1.0: 'red'
        }
    ).add_to(m)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_file))
        logger.info("Heatmap saved", output_file=str(output_file))

    return m
