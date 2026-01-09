"""
Enhanced unified map visualization for RAN optimization results.

Creates a comprehensive interactive dashboard with:
- All 4 algorithm outputs on a single map
- Filtering by issue type, severity, band, and environment
- Summary statistics panel
- Cell detail popups with recommendations
- Exportable data
"""
import folium
from folium import plugins
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime

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
    earth_radius = 6371000

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    half_width = sector_width / 2.0

    angles = [
        bearing,
        bearing - half_width,
        bearing + half_width,
    ]

    points = [(lat, lon)]

    for angle in angles[1:]:
        angle_rad = np.radians(angle)

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

    unique_geohashes = grid_df['geohash7'].unique()
    geohash_bounds = {}
    for gh in unique_geohashes:
        min_lat, max_lat, min_lon, max_lon = get_box_bounds(gh)
        geohash_bounds[gh] = [[min_lat, min_lon], [max_lat, max_lon]]

    has_is_overshooting = 'is_overshooting' in grid_df.columns
    has_is_interfering = 'is_interfering' in grid_df.columns
    has_band = 'Band' in grid_df.columns

    for cell_name, cell_grids in grid_df.groupby('cell_name'):
        grids_list = []

        geohashes = cell_grids['geohash7'].values
        latitudes = cell_grids['latitude'].values
        longitudes = cell_grids['longitude'].values

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
            'cell_name': int(cell_name) if isinstance(cell_name, (int, np.integer)) else str(cell_name),
            'grids': grids_list
        }

        json_file = grid_dir / f"cell_{cell_name}_grids.json"
        with open(json_file, 'w') as f:
            json.dump(grid_data, f, separators=(',', ':'))

        grid_paths[cell_name] = f"grids/cell_{cell_name}_grids.json"

    logger.info(
        "Saved grid data files",
        num_cells=len(grid_paths),
        output_dir=str(grid_dir),
    )

    return grid_paths


# Color schemes
SEVERITY_COLORS = {
    'CRITICAL': '#dc3545',  # Red
    'HIGH': '#fd7e14',      # Orange
    'MEDIUM': '#ffc107',    # Yellow
    'LOW': '#0dcaf0',       # Cyan
    'MINIMAL': '#6c757d',   # Gray
}

ENVIRONMENT_COLORS = {
    'URBAN': '#0d6efd',     # Blue
    'SUBURBAN': '#198754',  # Green
    'RURAL': '#6f42c1',     # Purple
}

ISSUE_COLORS = {
    'overshooting': '#dc3545',   # Red
    'undershooting': '#0d6efd',  # Blue
    'no_coverage': '#ffc107',    # Yellow
    'low_coverage': '#fd7e14',   # Orange
}


def create_enhanced_map(
    overshooting_df: Optional[pd.DataFrame] = None,
    undershooting_df: Optional[pd.DataFrame] = None,
    gis_df: Optional[pd.DataFrame] = None,
    no_coverage_gdf: Optional[gpd.GeoDataFrame] = None,
    low_coverage_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    overshooting_grid_df: Optional[pd.DataFrame] = None,
    undershooting_grid_df: Optional[pd.DataFrame] = None,
    cell_hulls_gdf: Optional[gpd.GeoDataFrame] = None,
    output_file: Optional[Path] = None,
    title: str = "RAN Optimizer - Network Issues Dashboard",
) -> folium.Map:
    """
    Create an enhanced unified map with all RAN optimization layers.

    Args:
        overshooting_df: DataFrame with overshooting cell results
        undershooting_df: DataFrame with undershooting cell results
        gis_df: DataFrame with cell GIS data (for coordinate lookup)
        no_coverage_gdf: GeoDataFrame with no coverage gap polygons
        low_coverage_gdfs: Dict of band -> GeoDataFrame with low coverage areas
        overshooting_grid_df: DataFrame with overshooting grid bins (for lazy loading)
        undershooting_grid_df: DataFrame with undershooting grid bins (for lazy loading)
        cell_hulls_gdf: GeoDataFrame with cell coverage hull polygons
        output_file: Path to save HTML output
        title: Map title

    Returns:
        folium.Map object
    """
    logger.info("Creating enhanced unified map")

    # Initialize counts
    overshooting_count = len(overshooting_df) if overshooting_df is not None else 0
    undershooting_count = len(undershooting_df) if undershooting_df is not None else 0
    no_coverage_count = len(no_coverage_gdf) if no_coverage_gdf is not None else 0
    low_coverage_count = sum(len(gdf) for gdf in (low_coverage_gdfs or {}).values())

    # Calculate map center
    center_lat, center_lon = _calculate_map_center(
        overshooting_df, undershooting_df, gis_df
    )

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap',
        control_scale=True,
    )

    # Add tile layer options
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)

    # Build statistics for summary panel
    stats = _calculate_statistics(
        overshooting_df, undershooting_df, no_coverage_gdf, low_coverage_gdfs
    )

    # Create feature groups for each layer
    layers = {}

    # Get cell IDs that have grid data
    over_grid_cell_names = set(overshooting_grid_df['cell_name'].astype(str).unique()) if overshooting_grid_df is not None and len(overshooting_grid_df) > 0 else None
    under_grid_cell_names = set(undershooting_grid_df['cell_name'].astype(str).unique()) if undershooting_grid_df is not None and len(undershooting_grid_df) > 0 else None

    # Base Layer: Coverage hulls per band (show=False by default)
    if cell_hulls_gdf is not None and len(cell_hulls_gdf) > 0:
        hull_layers = _add_coverage_hulls_layers(m, cell_hulls_gdf, gis_df)
        layers.update(hull_layers)

    # Base Layer: All cells (show=False by default)
    if gis_df is not None and len(gis_df) > 0:
        layers['all_cells'] = _add_all_cells_layer(m, gis_df)

    # Layer 1: Overshooting cells
    if overshooting_df is not None and len(overshooting_df) > 0:
        layers['overshooting'] = _add_overshooting_layer(m, overshooting_df, gis_df, grid_cell_names=over_grid_cell_names)

    # Layer 2: Undershooting cells
    if undershooting_df is not None and len(undershooting_df) > 0:
        layers['undershooting'] = _add_undershooting_layer(m, undershooting_df, gis_df, grid_cell_names=under_grid_cell_names)

    # Layer 3: No coverage gaps
    if no_coverage_gdf is not None and len(no_coverage_gdf) > 0:
        layers['no_coverage'] = _add_no_coverage_layer(m, no_coverage_gdf)

    # Layer 4: Low coverage areas (per band)
    if low_coverage_gdfs:
        for band, gdf in low_coverage_gdfs.items():
            if len(gdf) > 0:
                layers[f'low_coverage_{band}'] = _add_low_coverage_layer(m, gdf, band)

    # Save grid data for lazy loading and add JavaScript
    grid_paths = {}
    if output_file:
        output_dir = Path(output_file).parent

        # Save overshooting grid data
        if overshooting_grid_df is not None and len(overshooting_grid_df) > 0:
            over_grid_paths = save_grid_data_files(overshooting_grid_df, output_dir / 'overshooting_grids')
            for cell_name, path in over_grid_paths.items():
                # path is like "grids/cell_xxx_grids.json", prepend the overshooting_grids dir
                grid_paths[f'over_{cell_name}'] = f'overshooting_grids/{path}'
            logger.info("Saved overshooting grid data", cells=len(over_grid_paths))

        # Save undershooting grid data
        if undershooting_grid_df is not None and len(undershooting_grid_df) > 0:
            under_grid_paths = save_grid_data_files(undershooting_grid_df, output_dir / 'undershooting_grids')
            for cell_name, path in under_grid_paths.items():
                # path is like "grids/cell_xxx_grids.json", prepend the undershooting_grids dir
                grid_paths[f'under_{cell_name}'] = f'undershooting_grids/{path}'
            logger.info("Saved undershooting grid data", cells=len(under_grid_paths))

    # Add grid loading JavaScript if we have grids
    if grid_paths:
        _add_grid_loading_javascript(m, grid_paths)

    # Add layer control (topright to avoid overlap with summary/filter panels on left)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # Add fullscreen button
    plugins.Fullscreen(position='topleft').add_to(m)

    # Add minimap
    plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)

    # Add search box (if cells have names)
    if gis_df is not None and 'site_name' in gis_df.columns:
        plugins.Search(
            layer=layers.get('overshooting'),
            search_label='site_name',
            placeholder='Search cells...',
            collapsed=True,
        ).add_to(m)

    # Add custom HTML elements
    _add_title_panel(m, title)
    _add_summary_panel(m, stats)
    _add_legend_panel(m, stats)
    _add_filter_panel(m, stats)

    # Add filter JavaScript
    _add_filter_javascript(m)

    # Add export functionality
    _add_export_functionality(m, stats)

    # Save if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_file))
        logger.info("Enhanced map saved", path=str(output_file))

    return m


def _calculate_map_center(
    overshooting_df: Optional[pd.DataFrame],
    undershooting_df: Optional[pd.DataFrame],
    gis_df: Optional[pd.DataFrame],
) -> tuple:
    """Calculate map center from available data."""
    lats, lons = [], []

    if overshooting_df is not None and 'latitude' in overshooting_df.columns:
        lats.extend(overshooting_df['latitude'].dropna().tolist())
        lons.extend(overshooting_df['longitude'].dropna().tolist())

    if undershooting_df is not None and 'latitude' in undershooting_df.columns:
        lats.extend(undershooting_df['latitude'].dropna().tolist())
        lons.extend(undershooting_df['longitude'].dropna().tolist())

    if gis_df is not None:
        lat_col = 'latitude' if 'latitude' in gis_df.columns else 'Latitude'
        lon_col = 'longitude' if 'longitude' in gis_df.columns else 'Longitude'
        if lat_col in gis_df.columns:
            lats.extend(gis_df[lat_col].dropna().tolist())
            lons.extend(gis_df[lon_col].dropna().tolist())

    if lats and lons:
        return sum(lats) / len(lats), sum(lons) / len(lons)
    return 53.0, -8.0  # Default to Ireland center


def _calculate_statistics(
    overshooting_df: Optional[pd.DataFrame],
    undershooting_df: Optional[pd.DataFrame],
    no_coverage_gdf: Optional[gpd.GeoDataFrame],
    low_coverage_gdfs: Optional[Dict[str, gpd.GeoDataFrame]],
) -> dict:
    """Calculate summary statistics for the dashboard."""
    stats = {
        'overshooting': {
            'total': 0,
            'by_severity': {},
            'by_environment': {},
            'by_band': {},
            'avg_tilt_change': 0,
        },
        'undershooting': {
            'total': 0,
            'by_severity': {},
            'by_environment': {},
            'by_band': {},
            'avg_uptilt': 0,
        },
        'no_coverage': {
            'total': 0,
            'total_area_km2': 0,
        },
        'low_coverage': {
            'total': 0,
            'by_band': {},
            'total_area_km2': 0,
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Overshooting stats
    if overshooting_df is not None and len(overshooting_df) > 0:
        stats['overshooting']['total'] = len(overshooting_df)
        if 'severity_category' in overshooting_df.columns:
            stats['overshooting']['by_severity'] = overshooting_df['severity_category'].value_counts().to_dict()
        if 'environment' in overshooting_df.columns:
            stats['overshooting']['by_environment'] = overshooting_df['environment'].value_counts().to_dict()
        if 'band' in overshooting_df.columns:
            stats['overshooting']['by_band'] = overshooting_df['band'].value_counts().to_dict()
        if 'recommended_tilt_change' in overshooting_df.columns:
            stats['overshooting']['avg_tilt_change'] = overshooting_df['recommended_tilt_change'].mean()

    # Undershooting stats
    if undershooting_df is not None and len(undershooting_df) > 0:
        stats['undershooting']['total'] = len(undershooting_df)
        if 'severity_category' in undershooting_df.columns:
            stats['undershooting']['by_severity'] = undershooting_df['severity_category'].value_counts().to_dict()
        if 'environment' in undershooting_df.columns:
            stats['undershooting']['by_environment'] = undershooting_df['environment'].value_counts().to_dict()
        if 'band' in undershooting_df.columns:
            stats['undershooting']['by_band'] = undershooting_df['band'].value_counts().to_dict()
        if 'recommended_uptilt_deg' in undershooting_df.columns:
            stats['undershooting']['avg_uptilt'] = undershooting_df['recommended_uptilt_deg'].mean()

    # No coverage stats
    if no_coverage_gdf is not None and len(no_coverage_gdf) > 0:
        stats['no_coverage']['total'] = len(no_coverage_gdf)
        if 'area_km2' in no_coverage_gdf.columns:
            stats['no_coverage']['total_area_km2'] = no_coverage_gdf['area_km2'].sum()

    # Low coverage stats
    if low_coverage_gdfs:
        for band, gdf in low_coverage_gdfs.items():
            if len(gdf) > 0:
                stats['low_coverage']['total'] += len(gdf)
                stats['low_coverage']['by_band'][str(band)] = len(gdf)
                if 'area_km2' in gdf.columns:
                    stats['low_coverage']['total_area_km2'] += gdf['area_km2'].sum()

    return stats


def _add_all_cells_layer(
    m: folium.Map,
    gis_df: pd.DataFrame,
) -> folium.FeatureGroup:
    """Add a layer showing all cells as grey sector triangles."""
    layer = folium.FeatureGroup(name='All Cells', show=False)

    # Find the right columns
    cell_name_col = 'CILAC' if 'CILAC' in gis_df.columns else 'cell_name'
    lat_col = 'Latitude' if 'Latitude' in gis_df.columns else 'latitude'
    lon_col = 'Longitude' if 'Longitude' in gis_df.columns else 'longitude'
    az_col = 'Bearing' if 'Bearing' in gis_df.columns else 'azimuth_deg'
    name_col = 'Name' if 'Name' in gis_df.columns else 'cell_name'
    band_col = 'Band' if 'Band' in gis_df.columns else 'band'

    for _, cell in gis_df.iterrows():
        lat = cell.get(lat_col)
        lon = cell.get(lon_col)

        if pd.isna(lat) or pd.isna(lon):
            continue

        cell_name = cell.get(cell_name_col, 'N/A')
        cell_name = cell.get(name_col, '') if name_col in gis_df.columns else ''
        azimuth = cell.get(az_col, 0) if az_col in gis_df.columns else 0
        band = cell.get(band_col, 'N/A') if band_col in gis_df.columns else 'N/A'

        # Format display
        if pd.notna(cell_name) and cell_name:
            display_name = f"{cell_name} ({cell_name})"
        else:
            display_name = str(cell_name)

        tooltip = f"{display_name} - {band}"

        # Draw sector triangle if azimuth available
        if pd.notna(azimuth):
            sector_points = calculate_sector_points(lat, lon, azimuth, 400, 40.0)
            folium.Polygon(
                locations=sector_points,
                color='#6c757d',  # Grey
                weight=1,
                fill=True,
                fillColor='#6c757d',
                fillOpacity=0.4,
                tooltip=tooltip,
            ).add_to(layer)

        # Draw small center marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='#6c757d',
            fill=True,
            fillColor='#495057',
            fillOpacity=0.6,
            weight=1,
            tooltip=tooltip,
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _add_coverage_hulls_layers(
    m: folium.Map,
    hulls_gdf: gpd.GeoDataFrame,
    gis_df: Optional[pd.DataFrame] = None,
) -> Dict[str, folium.FeatureGroup]:
    """Add coverage hull layers split by band."""
    # Find columns
    name_col = 'cell_name' if 'cell_name' in hulls_gdf.columns else 'Name'
    cilac_col = 'cilac' if 'cilac' in hulls_gdf.columns else 'CILAC'
    area_col = 'area_km2' if 'area_km2' in hulls_gdf.columns else None

    # Merge band info from GIS data
    hulls_with_band = hulls_gdf.copy()
    if gis_df is not None:
        gis_cell_name_col = 'CILAC' if 'CILAC' in gis_df.columns else 'cell_name'
        gis_band_col = 'Band' if 'Band' in gis_df.columns else 'band'

        if gis_band_col in gis_df.columns:
            band_map = gis_df[[gis_cell_name_col, gis_band_col]].drop_duplicates(gis_cell_name_col)
            band_map[gis_cell_name_col] = band_map[gis_cell_name_col].astype(str)
            hulls_with_band[cilac_col] = hulls_with_band[cilac_col].astype(str)
            hulls_with_band = hulls_with_band.merge(
                band_map, left_on=cilac_col, right_on=gis_cell_name_col, how='left'
            )
            hulls_with_band['band'] = hulls_with_band[gis_band_col]

    # Get unique bands
    if 'band' in hulls_with_band.columns:
        bands = hulls_with_band['band'].dropna().unique()
        # Extract numeric band values
        band_values = []
        for b in bands:
            if pd.notna(b):
                # Handle "L800", "L700" format or numeric
                if isinstance(b, str) and b.startswith('L'):
                    try:
                        band_values.append(int(b[1:]))
                    except ValueError:
                        band_values.append(b)
                else:
                    try:
                        band_values.append(int(b))
                    except (ValueError, TypeError):
                        band_values.append(b)
        band_values = sorted(set(band_values))
    else:
        band_values = ['All']

    # Generate band colors dynamically from a palette
    color_palette = [
        '#2ecc71',  # Green
        '#3498db',  # Blue
        '#9b59b6',  # Purple
        '#e74c3c',  # Red
        '#f39c12',  # Orange
        '#1abc9c',  # Teal
        '#e91e63',  # Pink
        '#00bcd4',  # Cyan
        '#ff5722',  # Deep Orange
        '#795548',  # Brown
    ]
    band_colors = {band: color_palette[i % len(color_palette)] for i, band in enumerate(band_values) if band != 'All'}

    layers = {}
    for band in band_values:
        layer = folium.FeatureGroup(name=f'Coverage Hulls {band}MHz', show=False)

        # Filter hulls for this band
        if 'band' in hulls_with_band.columns and band != 'All':
            # Match both numeric and string formats
            band_filter = hulls_with_band['band'].apply(
                lambda x: (isinstance(x, str) and x == f'L{band}') or
                          (pd.notna(x) and str(x) == str(band))
            )
            band_hulls = hulls_with_band[band_filter]
        else:
            band_hulls = hulls_with_band

        color = band_colors.get(band, '#adb5bd')

        for _, row in band_hulls.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            cell_name = row.get(name_col, 'N/A') if name_col in band_hulls.columns else 'N/A'
            cilac = row.get(cilac_col, 'N/A') if cilac_col in band_hulls.columns else 'N/A'
            area = row.get(area_col, 0) if area_col and area_col in band_hulls.columns else 0

            tooltip = f"{cell_name} ({cilac}) - {band}MHz"
            if area > 0:
                tooltip += f" - {area:.2f} km²"

            # Create style function with closure to capture color
            def style_func(x, c=color):
                return {
                    'fillColor': c,
                    'color': c,
                    'weight': 1,
                    'fillOpacity': 0.2,
                }

            folium.GeoJson(
                geom,
                style_function=style_func,
                tooltip=tooltip,
            ).add_to(layer)

        layer.add_to(m)
        layers[f'coverage_hulls_{band}'] = layer

    return layers


def _add_overshooting_layer(
    m: folium.Map,
    df: pd.DataFrame,
    gis_df: Optional[pd.DataFrame],
    grid_cell_names: Optional[set] = None,
) -> folium.FeatureGroup:
    """Add overshooting cells layer with sectors and markers."""
    layer = folium.FeatureGroup(name='Overshooting Cells', show=True)

    # Merge cell_name, band and azimuth from GIS data if available
    if gis_df is not None and len(df) > 0:
        df = df.copy()
        # Find the right columns in GIS data
        gis_cell_name_col = 'CILAC' if 'CILAC' in gis_df.columns else 'cell_name'
        gis_name_col = 'Name' if 'Name' in gis_df.columns else None
        gis_band_col = 'Band' if 'Band' in gis_df.columns else None
        gis_azimuth_col = 'Bearing' if 'Bearing' in gis_df.columns else ('azimuth_deg' if 'azimuth_deg' in gis_df.columns else None)

        if gis_name_col or gis_band_col or gis_azimuth_col:
            gis_cols = [gis_cell_name_col]
            if gis_name_col:
                gis_cols.append(gis_name_col)
            if gis_band_col:
                gis_cols.append(gis_band_col)
            if gis_azimuth_col:
                gis_cols.append(gis_azimuth_col)

            gis_subset = gis_df[gis_cols].drop_duplicates(gis_cell_name_col).copy()
            gis_subset[gis_cell_name_col] = gis_subset[gis_cell_name_col].astype(str)
            df['cell_name'] = df['cell_name'].astype(str)

            df = df.merge(gis_subset, left_on='cell_name', right_on=gis_cell_name_col, how='left')
            if gis_name_col and gis_name_col not in ['cell_name']:
                df = df.rename(columns={gis_name_col: 'cell_name'})
            if gis_band_col and gis_band_col not in ['band']:
                df = df.rename(columns={gis_band_col: 'band'})
            if gis_azimuth_col and gis_azimuth_col not in ['azimuth_deg']:
                df = df.rename(columns={gis_azimuth_col: 'azimuth_deg'})

    for _, cell in df.iterrows():
        lat = cell.get('latitude')
        lon = cell.get('longitude')

        # Skip if no coordinates
        if pd.isna(lat) or pd.isna(lon):
            continue

        cell_name = cell['cell_name']
        severity = cell.get('severity_category', 'MEDIUM')
        severity_color = SEVERITY_COLORS.get(severity, '#6c757d')
        env = cell.get('environment', 'SUBURBAN')
        env_color = ENVIRONMENT_COLORS.get(env, '#198754')
        band = cell.get('band', 0)
        azimuth = cell.get('azimuth_deg', 0)

        # Create popup content
        has_grid = grid_cell_names is not None and str(cell_name) in grid_cell_names
        popup_html = _create_overshooting_popup(cell, has_grid_data=has_grid)

        # Draw sector triangle
        if not pd.isna(azimuth):
            sector_points = calculate_sector_points(lat, lon, azimuth, 500, 40.0)
            folium.Polygon(
                locations=sector_points,
                color=severity_color,
                weight=2,
                fill=True,
                fillColor=severity_color,
                fillOpacity=0.3,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"Cell {cell_name} - {severity}",
                className=f"issue-marker issue-overshooting severity-{severity} env-{env} band-{band}",
            ).add_to(layer)

        # Draw center marker
        radius = 6 + (cell.get('severity_score', 0.5) * 8)
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=env_color,
            fill=True,
            fillColor=env_color,
            fillOpacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Cell {cell_name} - {env} - {severity}",
            className=f"issue-marker issue-overshooting severity-{severity} env-{env} band-{band}",
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _add_undershooting_layer(
    m: folium.Map,
    df: pd.DataFrame,
    gis_df: Optional[pd.DataFrame],
    grid_cell_names: Optional[set] = None,
) -> folium.FeatureGroup:
    """Add undershooting cells layer."""
    layer = folium.FeatureGroup(name='Undershooting Cells', show=True)

    # Merge from GIS data (lat/lon, cell_name, band, azimuth)
    if gis_df is not None:
        cell_name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'
        lat_col = 'latitude' if 'latitude' in gis_df.columns else 'Latitude'
        lon_col = 'longitude' if 'longitude' in gis_df.columns else 'Longitude'
        az_col = 'azimuth_deg' if 'azimuth_deg' in gis_df.columns else 'Bearing'
        name_col = 'Name' if 'Name' in gis_df.columns else None
        band_col = 'Band' if 'Band' in gis_df.columns else None

        gis_cols = [cell_name_col]
        if lat_col in gis_df.columns:
            gis_cols.extend([lat_col, lon_col])
        if az_col in gis_df.columns:
            gis_cols.append(az_col)
        if name_col:
            gis_cols.append(name_col)
        if band_col:
            gis_cols.append(band_col)

        gis_subset = gis_df[gis_cols].drop_duplicates(cell_name_col).copy()

        # Rename columns to standard names
        rename_map = {}
        if cell_name_col != 'cell_name':
            rename_map[cell_name_col] = 'cell_name'
        if lat_col != 'latitude' and lat_col in gis_subset.columns:
            rename_map[lat_col] = 'latitude'
            rename_map[lon_col] = 'longitude'
        if az_col != 'azimuth_deg' and az_col in gis_subset.columns:
            rename_map[az_col] = 'azimuth_deg'
        if name_col and name_col != 'cell_name':
            rename_map[name_col] = 'cell_name'
        if band_col and band_col != 'band':
            rename_map[band_col] = 'band'

        if rename_map:
            gis_subset = gis_subset.rename(columns=rename_map)

        # Ensure cell_name types match for merge
        df = df.copy()
        df['cell_name'] = df['cell_name'].astype(str)
        gis_subset['cell_name'] = gis_subset['cell_name'].astype(str)

        # Merge only columns not already in df
        merge_cols = ['cell_name'] + [c for c in gis_subset.columns if c != 'cell_name' and c not in df.columns]
        df = df.merge(gis_subset[merge_cols], on='cell_name', how='left')

    for _, cell in df.iterrows():
        lat = cell.get('latitude')
        lon = cell.get('longitude')

        if pd.isna(lat) or pd.isna(lon):
            continue

        cell_name = cell['cell_name']
        env = cell.get('environment', 'SUBURBAN')
        env_color = ENVIRONMENT_COLORS.get(env, '#198754')
        band = cell.get('band', 0)
        azimuth = cell.get('azimuth_deg', 0)
        severity = cell.get('severity_category', 'N/A')

        # Create popup
        has_grid = grid_cell_names is not None and str(cell_name) in grid_cell_names
        popup_html = _create_undershooting_popup(cell, has_grid_data=has_grid)

        # Draw sector (if azimuth available)
        if not pd.isna(azimuth):
            sector_points = calculate_sector_points(lat, lon, azimuth, 400, 40.0)
            folium.Polygon(
                locations=sector_points,
                color='#0d6efd',
                weight=2,
                fill=True,
                fillColor='#0d6efd',
                fillOpacity=0.2,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"Cell {cell_name} - Undershooting",
                className=f"issue-marker issue-undershooting severity-{severity} env-{env} band-{band}",
            ).add_to(layer)

        # Draw center marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color='#0d6efd',
            fill=True,
            fillColor=env_color,
            fillOpacity=0.8,
            weight=2,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Cell {cell_name} - {env}",
            className=f"issue-marker issue-undershooting severity-{severity} env-{env} band-{band}",
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _add_no_coverage_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
) -> folium.FeatureGroup:
    """Add no coverage gaps layer."""
    layer = folium.FeatureGroup(name='No Coverage Gaps', show=True)

    for idx, row in gdf.iterrows():
        geom = row['geometry']
        cluster_id = row.get('cluster_id', idx)
        area = row.get('area_km2', 0)
        n_points = row.get('n_points', 0)

        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #ffc107;">No Coverage Gap</h4>
            <div style="background: #fff3cd; padding: 8px; border-radius: 4px;">
                <strong>Cluster ID:</strong> {cluster_id}<br>
                <strong>Area:</strong> {area:.2f} km²<br>
                <strong>Grid points:</strong> {n_points}
            </div>
        </div>
        """

        folium.GeoJson(
            geom,
            style_function=lambda x: {
                'fillColor': '#ffc107',
                'color': '#856404',
                'weight': 2,
                'fillOpacity': 0.4,
            },
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"No Coverage - {area:.2f} km²",
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _add_low_coverage_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    band: str,
) -> folium.FeatureGroup:
    """Add low coverage layer for a specific band."""
    layer = folium.FeatureGroup(name=f'Low Coverage ({band} MHz)', show=True)

    for idx, row in gdf.iterrows():
        geom = row['geometry']
        cluster_id = row.get('cluster_id', idx)
        area = row.get('area_km2', 0)
        n_points = row.get('n_points', 0)
        serving_cells = row.get('serving_cells', '')
        serving_cell_names = row.get('serving_cell_names', '')
        n_serving_cells = row.get('n_serving_cells', 0)

        # Build serving cells HTML with both names and IDs
        serving_cells_html = ""
        if serving_cells:
            cells_list = serving_cells.split(',')[:5]  # Show max 5 cells
            names_list = serving_cell_names.split(',')[:5] if serving_cell_names else []

            # Combine names and IDs
            cells_display_parts = []
            for i, cell_name in enumerate(cells_list):
                cell_name = names_list[i] if i < len(names_list) and names_list[i] else None
                if cell_name:
                    cells_display_parts.append(f"{cell_name} ({cell_name})")
                else:
                    cells_display_parts.append(cell_name)

            cells_display = ', '.join(cells_display_parts)
            if n_serving_cells > 5:
                cells_display += f' (+{n_serving_cells - 5} more)'
            serving_cells_html = f"""
                <strong>Serving Cells:</strong> {cells_display}<br>
                <strong>Total Cells:</strong> {n_serving_cells}
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 0 0 8px 0; color: #fd7e14;">Low Coverage Area</h4>
            <div style="background: #ffe5d0; padding: 8px; border-radius: 4px;">
                <strong>Band:</strong> {band} MHz<br>
                <strong>Cluster ID:</strong> {cluster_id}<br>
                <strong>Area:</strong> {area:.2f} km²<br>
                <strong>Grid points:</strong> {n_points}<br>
                {serving_cells_html}
            </div>
        </div>
        """

        folium.GeoJson(
            geom,
            style_function=lambda x: {
                'fillColor': '#fd7e14',
                'color': '#c45a00',
                'weight': 2,
                'fillOpacity': 0.35,
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Low Coverage {band}MHz - {area:.2f} km²",
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _create_overshooting_popup(cell: pd.Series, has_grid_data: bool = False) -> str:
    """Create detailed popup for overshooting cell."""
    cell_name = cell['cell_name']
    cell_name = cell.get('cell_name', '')
    severity = cell.get('severity_category', 'N/A')
    severity_score = cell.get('severity_score', 0)
    env = cell.get('environment', 'N/A')
    band = cell.get('band', 'N/A')
    tilt_change = cell.get('recommended_tilt_change', 0)

    # Format band display
    if pd.notna(band) and band != 'N/A':
        band_display = f"{int(band)} MHz" if isinstance(band, (int, float)) else f"{band} MHz"
    else:
        band_display = 'N/A'

    # Format cell display with name if available
    if pd.notna(cell_name) and cell_name:
        cell_display = f"{cell_name} ({cell_name})"
    else:
        cell_display = str(cell_name)

    html = f"""
    <div style="font-family: Arial; width: 320px;">
        <h3 style="margin: 0 0 10px 0; color: #dc3545;">
            {cell_display}
            <span style="float: right; font-size: 12px; background: {SEVERITY_COLORS.get(severity, '#6c757d')};
                         color: white; padding: 2px 8px; border-radius: 4px;">{severity}</span>
        </h3>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Severity Score</div>
                <div style="font-size: 16px; font-weight: bold;">{severity_score:.3f}</div>
            </div>
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Environment</div>
                <div style="font-size: 14px; font-weight: bold; color: {ENVIRONMENT_COLORS.get(env, '#198754')};">{env}</div>
            </div>
        </div>

        <table style="width: 100%; font-size: 12px; margin-bottom: 10px;">
            <tr><td style="color: #6c757d;">Band:</td><td><strong>{band_display}</strong></td></tr>
            <tr><td style="color: #6c757d;">Overshooting grids:</td><td><strong>{cell.get('overshooting_grids', 'N/A'):,}</strong> ({cell.get('percentage_overshooting', 0)*100:.1f}%)</td></tr>
            <tr><td style="color: #6c757d;">Max distance:</td><td><strong>{cell.get('max_distance_m', 0)/1000:.1f} km</strong></td></tr>
            <tr><td style="color: #6c757d;">Avg edge RSRP:</td><td><strong>{cell.get('avg_edge_rsrp', 0):.1f} dBm</strong></td></tr>
            <tr><td style="color: #6c757d;">Current tilt:</td><td>M: {cell.get('mechanical_tilt', 0):.1f}° / E: {cell.get('electrical_tilt', 0):.1f}°</td></tr>
        </table>

        <div style="background: #fff3cd; padding: 10px; border-radius: 4px; border-left: 4px solid #ffc107;">
            <strong>Recommendation:</strong><br>
            Increase downtilt by <strong>+{tilt_change:.1f}°</strong>
        </div>
    """

    # Add expected impact info
    impact_lines = []
    if 'interference_reduction_pct' in cell and cell['interference_reduction_pct'] > 0:
        impact_lines.append(f"Interference reduction: {cell['interference_reduction_pct']:.1f}%")
        impact_lines.append(f"Grids resolved: {cell.get('removed_interference_grids', 0):.0f}")
    if 'new_max_distance_1deg_m' in cell and cell['new_max_distance_1deg_m'] > 0:
        new_dist_1deg = cell['new_max_distance_1deg_m'] / 1000
        new_dist_2deg = cell.get('new_max_distance_2deg_m', 0) / 1000
        impact_lines.append(f"New max distance: {new_dist_1deg:.1f} km (+1°) / {new_dist_2deg:.1f} km (+2°)")

    if impact_lines:
        html += f"""
        <div style="background: #d1ecf1; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
            <strong>Expected Impact:</strong><br>
            {'<br>'.join(impact_lines)}
        </div>
        """

    # Add grid loading button if grid data is available
    if has_grid_data:
        html += f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="gridBtn_overshooting_{cell_name}" onclick="toggleOvershootingGrids_{cell_name}()"
                    style="background: #dc3545; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Load & Show Grids
            </button>
            <div id="gridStatus_overshooting_{cell_name}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
        </div>
        """

    html += "</div>"
    return html


def _create_undershooting_popup(cell: pd.Series, has_grid_data: bool = False) -> str:
    """Create detailed popup for undershooting cell."""
    cell_name = cell['cell_name']
    cell_name = cell.get('cell_name', '')
    severity = cell.get('severity_category', 'N/A')
    severity_score = cell.get('severity_score', 0)
    env = cell.get('environment', 'N/A')
    band = cell.get('band', 'N/A')
    uptilt = cell.get('recommended_uptilt_deg', 0)

    # Format band display
    if pd.notna(band) and band != 'N/A':
        band_display = f"{int(band)} MHz" if isinstance(band, (int, float)) else f"{band} MHz"
    else:
        band_display = 'N/A'

    # Format cell display with name if available
    if pd.notna(cell_name) and cell_name:
        cell_display = f"{cell_name} ({cell_name})"
    else:
        cell_display = str(cell_name)

    # Get current and new distances
    current_dist = cell.get('max_distance_m', cell.get('current_distance_m', 0)) / 1000
    new_dist = cell.get('new_max_distance_m', 0) / 1000

    html = f"""
    <div style="font-family: Arial; width: 320px;">
        <h3 style="margin: 0 0 10px 0; color: #0d6efd;">
            {cell_display}
            <span style="float: right; font-size: 12px; background: {SEVERITY_COLORS.get(severity, '#0d6efd')};
                         color: white; padding: 2px 8px; border-radius: 4px;">{severity}</span>
        </h3>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Severity Score</div>
                <div style="font-size: 16px; font-weight: bold;">{severity_score:.3f}</div>
            </div>
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Environment</div>
                <div style="font-size: 14px; font-weight: bold; color: {ENVIRONMENT_COLORS.get(env, '#198754')};">{env}</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Band</div>
                <div style="font-size: 14px; font-weight: bold;">{band_display}</div>
            </div>
            <div style="background: #f8f9fa; padding: 6px; border-radius: 4px; text-align: center;">
                <div style="font-size: 11px; color: #6c757d;">Type</div>
                <div style="font-size: 14px; font-weight: bold; color: #0d6efd;">UNDERSHOOTING</div>
            </div>
        </div>

        <table style="width: 100%; font-size: 12px; margin-bottom: 10px;">
            <tr><td style="color: #6c757d;">Current max distance:</td><td><strong>{current_dist:.2f} km</strong></td></tr>
            <tr><td style="color: #6c757d;">Interference:</td><td><strong>{cell.get('interference_percentage', 0)*100:.1f}%</strong></td></tr>
            <tr><td style="color: #6c757d;">Current tilt:</td><td>M: {cell.get('mechanical_tilt', 0):.1f}° / E: {cell.get('electrical_tilt', 0):.1f}°</td></tr>
        </table>

        <div style="background: #cfe2ff; padding: 10px; border-radius: 4px; border-left: 4px solid #0d6efd;">
            <strong>Recommendation:</strong><br>
            Reduce downtilt by <strong>-{uptilt:.1f}°</strong> (uptilt)
        </div>
    """

    # Add coverage expansion info if available
    impact_lines = []
    if new_dist > 0:
        impact_lines.append(f"New max distance: {new_dist:.2f} km")
    if 'distance_increase_m' in cell and cell['distance_increase_m'] > 0:
        impact_lines.append(f"Coverage extension: +{cell['distance_increase_m']:.0f} m")
    if 'new_coverage_grids' in cell and cell['new_coverage_grids'] > 0:
        impact_lines.append(f"New coverage grids: +{cell['new_coverage_grids']:.0f}")
    if 'coverage_increase_percentage' in cell and cell['coverage_increase_percentage'] > 0:
        impact_lines.append(f"Coverage increase: +{cell['coverage_increase_percentage']*100:.1f}%")

    if impact_lines:
        html += f"""
        <div style="background: #d1ecf1; padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
            <strong>Expected Impact:</strong><br>
            {'<br>'.join(impact_lines)}
        </div>
        """

    # Add grid loading button if grid data is available
    if has_grid_data:
        html += f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="gridBtn_undershooting_{cell_name}" onclick="toggleUndershootingGrids_{cell_name}()"
                    style="background: #0d6efd; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Load & Show Grids
            </button>
            <div id="gridStatus_undershooting_{cell_name}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
        </div>
        """

    html += "</div>"
    return html


def _add_title_panel(m: folium.Map, title: str):
    """Add title panel to map."""
    html = f"""
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9999; padding: 12px 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h2 style="margin: 0; color: #333; font-family: Arial;">{title}</h2>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_summary_panel(m: folium.Map, stats: dict):
    """Add summary statistics panel."""
    over_total = stats['overshooting']['total']
    under_total = stats['undershooting']['total']
    no_cov_total = stats['no_coverage']['total']
    low_cov_total = stats['low_coverage']['total']
    total_issues = over_total + under_total + no_cov_total + low_cov_total

    html = f"""
    <div id="summaryPanel" style="
        position: fixed; top: 80px; left: 10px; width: 240px;
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9999; padding: 12px; font-family: Arial;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h4 style="margin: 0 0 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">
            Summary Statistics
        </h4>

        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 28px; font-weight: bold; color: #333;">{total_issues}</span>
            <span style="font-size: 12px; color: #6c757d; align-self: flex-end;">Total Issues</span>
        </div>

        <div style="margin-top: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 12px; height: 12px; background: {ISSUE_COLORS['overshooting']}; border-radius: 2px; margin-right: 6px;"></span>
                    Overshooting
                </span>
                <strong>{over_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 12px; height: 12px; background: {ISSUE_COLORS['undershooting']}; border-radius: 2px; margin-right: 6px;"></span>
                    Undershooting
                </span>
                <strong>{under_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 12px; height: 12px; background: {ISSUE_COLORS['no_coverage']}; border-radius: 2px; margin-right: 6px;"></span>
                    No Coverage
                </span>
                <strong>{no_cov_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 12px; height: 12px; background: {ISSUE_COLORS['low_coverage']}; border-radius: 2px; margin-right: 6px;"></span>
                    Low Coverage
                </span>
                <strong>{low_cov_total}</strong>
            </div>
        </div>

        <hr style="margin: 12px 0; border: 0; border-top: 1px solid #eee;">

        <div style="font-size: 11px; color: #6c757d;">
            <div>Avg downtilt: +{stats['overshooting']['avg_tilt_change']:.1f}°</div>
            <div>Avg uptilt: -{stats['undershooting']['avg_uptilt']:.1f}°</div>
            <div>No cov area: {stats['no_coverage']['total_area_km2']:.1f} km²</div>
            <div>Low cov area: {stats['low_coverage']['total_area_km2']:.1f} km²</div>
        </div>

        <div style="margin-top: 10px; font-size: 10px; color: #aaa;">
            Generated: {stats['timestamp'][:19]}
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_legend_panel(m: folium.Map, stats: dict):
    """Add legend panel."""
    html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 10px; width: 200px;
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9999; padding: 12px; font-family: Arial; font-size: 11px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h4 style="margin: 0 0 10px 0;">Legend</h4>

        <p style="margin: 4px 0; font-weight: bold;">Severity (Sectors)</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['CRITICAL']};">&#9650;</span> CRITICAL</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['HIGH']};">&#9650;</span> HIGH</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['MEDIUM']};">&#9650;</span> MEDIUM</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['LOW']};">&#9650;</span> LOW</p>

        <p style="margin: 10px 0 4px 0; font-weight: bold;">Environment (Markers)</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['URBAN']};">&#9679;</span> URBAN</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['SUBURBAN']};">&#9679;</span> SUBURBAN</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['RURAL']};">&#9679;</span> RURAL</p>

        <p style="margin: 10px 0 4px 0; font-weight: bold;">Coverage Areas</p>
        <p style="margin: 2px 0;"><span style="color: {ISSUE_COLORS['no_coverage']};">&#9632;</span> No Coverage</p>
        <p style="margin: 2px 0;"><span style="color: {ISSUE_COLORS['low_coverage']};">&#9632;</span> Low Coverage</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_filter_panel(m: folium.Map, stats: dict):
    """Add filter panel."""
    # Get unique values for filters
    severities = list(stats['overshooting'].get('by_severity', {}).keys())
    environments = list(stats['overshooting'].get('by_environment', {}).keys())
    environments.extend([e for e in stats['undershooting'].get('by_environment', {}).keys() if e not in environments])
    bands = list(stats['overshooting'].get('by_band', {}).keys())
    bands.extend([b for b in stats['undershooting'].get('by_band', {}).keys() if b not in bands])
    bands = sorted(set(bands))

    html = """
    <div id="filterPanel" style="
        position: fixed; top: 420px; right: 10px; width: 220px;
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9999; padding: 12px; font-family: Arial; font-size: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h4 style="margin: 0 0 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">
            Filters
        </h4>

        <div style="margin-bottom: 12px;">
            <label style="font-weight: bold; display: block; margin-bottom: 4px;">Issue Type</label>
            <select id="issueFilter" onchange="applyFilters()" style="width: 100%; padding: 4px;">
                <option value="all">All Issues</option>
                <option value="overshooting">Overshooting</option>
                <option value="undershooting">Undershooting</option>
            </select>
        </div>

        <div style="margin-bottom: 12px;">
            <label style="font-weight: bold; display: block; margin-bottom: 4px;">Severity</label>
            <select id="severityFilter" onchange="applyFilters()" style="width: 100%; padding: 4px;">
                <option value="all">All Severities</option>
    """

    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        if sev in severities:
            html += f'<option value="{sev}">{sev}</option>'

    html += """
            </select>
        </div>

        <div style="margin-bottom: 12px;">
            <label style="font-weight: bold; display: block; margin-bottom: 4px;">Environment</label>
            <select id="envFilter" onchange="applyFilters()" style="width: 100%; padding: 4px;">
                <option value="all">All Environments</option>
    """

    for env in ['URBAN', 'SUBURBAN', 'RURAL']:
        if env in environments:
            html += f'<option value="{env}">{env}</option>'

    html += """
            </select>
        </div>

        <div style="margin-bottom: 12px;">
            <label style="font-weight: bold; display: block; margin-bottom: 4px;">Frequency Band</label>
            <select id="bandFilter" onchange="applyFilters()" style="width: 100%; padding: 4px;">
                <option value="all">All Bands</option>
    """

    for band in bands:
        html += f'<option value="{band}">{band} MHz</option>'

    html += """
            </select>
        </div>

        <button onclick="resetFilters()" style="
            width: 100%; padding: 8px; background: #6c757d; color: white;
            border: none; border-radius: 4px; cursor: pointer;
        ">Reset Filters</button>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_filter_javascript(m: folium.Map):
    """Add JavaScript for filtering markers."""
    js = """
    <script>
    function applyFilters() {
        var issueType = document.getElementById('issueFilter').value;
        var severity = document.getElementById('severityFilter').value;
        var env = document.getElementById('envFilter').value;
        var band = document.getElementById('bandFilter').value;

        var markers = document.querySelectorAll('.issue-marker');
        markers.forEach(function(marker) {
            var show = true;

            // Issue type filter
            if (issueType !== 'all') {
                if (!marker.classList.contains('issue-' + issueType)) {
                    show = false;
                }
            }

            // Severity filter
            if (severity !== 'all') {
                if (!marker.classList.contains('severity-' + severity)) {
                    show = false;
                }
            }

            // Environment filter
            if (env !== 'all') {
                if (!marker.classList.contains('env-' + env)) {
                    show = false;
                }
            }

            // Band filter
            if (band !== 'all') {
                if (!marker.classList.contains('band-' + band)) {
                    show = false;
                }
            }

            marker.style.display = show ? '' : 'none';
        });
    }

    function resetFilters() {
        document.getElementById('issueFilter').value = 'all';
        document.getElementById('severityFilter').value = 'all';
        document.getElementById('envFilter').value = 'all';
        document.getElementById('bandFilter').value = 'all';
        applyFilters();
    }
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


def _add_grid_loading_javascript(m: folium.Map, grid_paths: dict):
    """Add JavaScript for lazy-loading grid data from JSON files."""
    map_var_name = m.get_name()

    # Separate overshooting and undershooting paths
    over_paths = {k.replace('over_', ''): v for k, v in grid_paths.items() if k.startswith('over_')}
    under_paths = {k.replace('under_', ''): v for k, v in grid_paths.items() if k.startswith('under_')}

    js = f"""
    <script>
    // Store loaded grids by cell_name and type
    var loadedGrids = {{}};
    var gridLayers = {{}};

    // Grid paths for each cell
    var overshootingGridPaths = {json.dumps(over_paths)};
    var undershootingGridPaths = {json.dumps(under_paths)};

    // Function to load and display grids for a cell
    function loadGridsForCell(cellId, gridType) {{
        var gridPath = gridType === 'overshooting' ? overshootingGridPaths[cellId] : undershootingGridPaths[cellId];
        if (!gridPath) {{
            console.log('No grid path for cell ' + cellId + ' type ' + gridType);
            return;
        }}

        var key = gridType + '_' + cellId;
        var statusDiv = document.getElementById('gridStatus_' + key);
        var btn = document.getElementById('gridBtn_' + key);

        if (loadedGrids[key]) {{
            // Already loaded, just toggle visibility
            var layer = gridLayers[key];
            if (layer) {{
                if (layer._map) {{
                    {map_var_name}.removeLayer(layer);
                    if (btn) btn.textContent = 'Show Grids';
                }} else {{
                    {map_var_name}.addLayer(layer);
                    if (btn) btn.textContent = 'Hide Grids';
                }}
            }}
            return;
        }}

        // Load from JSON
        if (statusDiv) statusDiv.textContent = 'Loading...';
        if (btn) btn.disabled = true;

        fetch(gridPath)
            .then(response => response.json())
            .then(data => {{
                // Create layer group for this cell's grids
                var gridLayer = L.layerGroup();

                // Color based on type - red for problem grids (overshooting or interference)
                var highlightColor = '#dc3545';  // Red for both overshooting and interference grids
                var highlightLabel = gridType === 'overshooting' ? 'OVERSHOOTING' : 'HIGH INTERFERENCE';
                var normalLabel = gridType === 'overshooting' ? 'Normal' : 'Low Interference';

                data.grids.forEach(function(grid) {{
                    // Color red if flagged as overshooting/interfering, gray otherwise
                    var isHighlighted = grid.overshoot;
                    var color = isHighlighted ? highlightColor : '#6c757d';

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
                        (isHighlighted ? '<b>' + highlightLabel + '</b>' : normalLabel)
                    );
                    rectangle.addTo(gridLayer);
                }});

                {map_var_name}.addLayer(gridLayer);
                loadedGrids[key] = true;
                gridLayers[key] = gridLayer;

                if (statusDiv) statusDiv.textContent = data.grids.length + ' grids loaded';
                if (btn) {{
                    btn.textContent = 'Hide Grids';
                    btn.disabled = false;
                }}
            }})
            .catch(error => {{
                if (statusDiv) statusDiv.textContent = 'Error loading grids';
                if (btn) btn.disabled = false;
                console.error('Error loading grids:', error);
            }});
    }}

    // Create toggle functions for each cell
    """

    # Add toggle functions for overshooting cells
    for cell_name in over_paths.keys():
        js += f"""
    function toggleOvershootingGrids_{cell_name}() {{
        loadGridsForCell('{cell_name}', 'overshooting');
    }}
    """

    # Add toggle functions for undershooting cells
    for cell_name in under_paths.keys():
        js += f"""
    function toggleUndershootingGrids_{cell_name}() {{
        loadGridsForCell('{cell_name}', 'undershooting');
    }}
    """

    js += "</script>"
    m.get_root().html.add_child(folium.Element(js))


def _add_export_functionality(m: folium.Map, stats: dict):
    """Add export button and functionality."""
    stats_json = json.dumps(stats, indent=2)

    html = f"""
    <div style="
        position: fixed; bottom: 30px; right: 10px;
        z-index: 9999;
    ">
        <button onclick="exportData()" style="
            padding: 10px 16px; background: #198754; color: white;
            border: none; border-radius: 4px; cursor: pointer;
            font-family: Arial; font-weight: bold;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        ">
            Export Statistics
        </button>
    </div>

    <script>
    var statsData = {stats_json};

    function exportData() {{
        var blob = new Blob([JSON.stringify(statsData, null, 2)], {{type: 'application/json'}});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'ran_optimizer_stats.json';
        a.click();
        URL.revokeObjectURL(url);
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(html))


def generate_enhanced_map_from_files(
    output_dir: Path,
    gis_file: Path,
    output_file: Optional[Path] = None,
) -> folium.Map:
    """
    Generate enhanced map from output files.

    Args:
        output_dir: Directory containing algorithm output files
        gis_file: Path to GIS data file
        output_file: Path to save HTML map

    Returns:
        folium.Map object
    """
    output_dir = Path(output_dir)

    # Load GIS data
    gis_df = pd.read_csv(gis_file)

    # Load overshooting results
    over_file = output_dir / 'overshooting_cells.csv'
    overshooting_df = pd.read_csv(over_file) if over_file.exists() else None

    # Load undershooting results
    under_file = output_dir / 'undershooting_cells.csv'
    undershooting_df = pd.read_csv(under_file) if under_file.exists() else None

    # Load no coverage results
    no_cov_file = output_dir / 'no_coverage_clusters.geojson'
    no_coverage_gdf = gpd.read_file(no_cov_file) if no_cov_file.exists() else None

    # Load low coverage results (single file with all bands)
    low_coverage_gdfs = {}
    low_cov_file = output_dir / 'low_coverage.geojson'
    if low_cov_file.exists():
        all_low_cov = gpd.read_file(low_cov_file)
        # Split by band for compatibility with existing visualization
        for band in all_low_cov['band'].unique():
            low_coverage_gdfs[band] = all_low_cov[all_low_cov['band'] == band]

    # Generate map
    if output_file is None:
        output_file = output_dir / 'maps' / 'enhanced_dashboard.html'

    return create_enhanced_map(
        overshooting_df=overshooting_df,
        undershooting_df=undershooting_df,
        gis_df=gis_df,
        no_coverage_gdf=no_coverage_gdf,
        low_coverage_gdfs=low_coverage_gdfs,
        output_file=output_file,
    )
