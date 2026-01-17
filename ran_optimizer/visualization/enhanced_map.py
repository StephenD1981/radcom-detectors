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
import re
from datetime import datetime

from ran_optimizer.utils.logging_config import get_logger


def _sanitize_js_name(name: str) -> str:
    """Sanitize a string for use as JavaScript function/variable name.

    Replaces any non-alphanumeric characters (except underscore) with underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
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


def prepare_grid_data_inline(grid_df: pd.DataFrame) -> dict:
    """
    Prepare grid data as inline JavaScript data structure.

    This embeds grid data directly in the HTML to avoid CORS issues when
    opening the HTML file directly via file:// protocol.

    Args:
        grid_df: DataFrame with grid bins

    Returns:
        Dict mapping cell_name to grid data dict (for inline embedding)
    """
    if grid_df is None or len(grid_df) == 0:
        return {}

    grid_data_map = {}

    # Find geohash column (different datasets use different names)
    geohash_col = next((c for c in ['geohash7', 'grid', 'geohash'] if c in grid_df.columns), 'geohash7')
    unique_geohashes = grid_df[geohash_col].unique()
    geohash_bounds = {}
    for gh in unique_geohashes:
        min_lat, max_lat, min_lon, max_lon = get_box_bounds(gh)
        geohash_bounds[gh] = [[min_lat, min_lon], [max_lat, max_lon]]

    has_is_overshooting = 'is_overshooting' in grid_df.columns
    has_is_interfering = 'is_interfering' in grid_df.columns
    band_col = 'band' if 'band' in grid_df.columns else ('Band' if 'Band' in grid_df.columns else None)
    has_band = band_col is not None

    for cell_name, cell_grids in grid_df.groupby('cell_name'):
        grids_list = []

        geohashes = cell_grids[geohash_col].values
        latitudes = cell_grids['latitude'].values
        longitudes = cell_grids['longitude'].values

        if has_is_overshooting:
            highlighted = cell_grids['is_overshooting'].values
        elif has_is_interfering:
            highlighted = cell_grids['is_interfering'].values
        else:
            highlighted = [False] * len(cell_grids)
        bands = cell_grids[band_col].values if has_band else [0] * len(cell_grids)

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

        # Use string key for JavaScript compatibility
        cell_key = str(cell_name)
        grid_data_map[cell_key] = grid_data

    logger.info(
        "Prepared inline grid data",
        num_cells=len(grid_data_map),
        total_grids=sum(len(d['grids']) for d in grid_data_map.values()),
    )

    return grid_data_map


def save_grid_data_files(
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Save grid data as separate JSON files per cell (for debugging/external tools).

    Note: The enhanced map now uses inline data (prepare_grid_data_inline) to avoid
    CORS issues. This function is kept for compatibility with external tools that
    may want to access the grid data programmatically.

    Args:
        grid_df: DataFrame with grid bins
        output_dir: Directory to save JSON files

    Returns:
        Dict mapping cell_name to relative JSON file path
    """
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_paths = {}

    # Find geohash column (different datasets use different names)
    geohash_col = next((c for c in ['geohash7', 'grid', 'geohash'] if c in grid_df.columns), 'geohash7')
    unique_geohashes = grid_df[geohash_col].unique()
    geohash_bounds = {}
    for gh in unique_geohashes:
        min_lat, max_lat, min_lon, max_lon = get_box_bounds(gh)
        geohash_bounds[gh] = [[min_lat, min_lon], [max_lat, max_lon]]

    has_is_overshooting = 'is_overshooting' in grid_df.columns
    has_is_interfering = 'is_interfering' in grid_df.columns
    band_col = 'band' if 'band' in grid_df.columns else ('Band' if 'Band' in grid_df.columns else None)
    has_band = band_col is not None

    for cell_name, cell_grids in grid_df.groupby('cell_name'):
        grids_list = []

        geohashes = cell_grids[geohash_col].values
        latitudes = cell_grids['latitude'].values
        longitudes = cell_grids['longitude'].values

        if has_is_overshooting:
            highlighted = cell_grids['is_overshooting'].values
        elif has_is_interfering:
            highlighted = cell_grids['is_interfering'].values
        else:
            highlighted = [False] * len(cell_grids)
        bands = cell_grids[band_col].values if has_band else [0] * len(cell_grids)

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
    'pci_confusion': '#9b59b6',  # Purple
    'pci_collision': '#9b59b6',  # Purple
    'pci_blacklist': '#fd7e14',  # Orange
    'ca_imbalance': '#17a2b8',   # Teal
    'crossed_feeder': '#dc3545', # Red
    'interference': '#dc3545',   # Red
}

# Line colors for relationship visualization
LINE_COLORS = {
    'pci': '#9b59b6',              # Purple - PCI relationships
    'blacklist': '#fd7e14',        # Orange - Blacklist suggestions
    'crossed_offender': '#dc3545', # Red - Suspicious crossed feeder relations
    'crossed_normal': '#198754',   # Green - Normal crossed feeder relations
    'low_coverage': '#fd7e14',     # Orange - Low coverage serving cells
    'interference': '#dc3545',     # Red - Interference cluster cells
    'ca_coverage': '#0d6efd',      # Blue - CA imbalance coverage hull
    'ca_capacity': '#198754',      # Green - CA imbalance capacity hull
}


def create_enhanced_map(
    overshooting_df: Optional[pd.DataFrame] = None,
    undershooting_df: Optional[pd.DataFrame] = None,
    gis_df: Optional[pd.DataFrame] = None,
    no_coverage_gdf: Optional[gpd.GeoDataFrame] = None,
    no_coverage_per_band_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    low_coverage_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    overshooting_grid_df: Optional[pd.DataFrame] = None,
    undershooting_grid_df: Optional[pd.DataFrame] = None,
    cell_hulls_gdf: Optional[gpd.GeoDataFrame] = None,
    # New detector DataFrames
    pci_confusions_df: Optional[pd.DataFrame] = None,
    pci_collisions_df: Optional[pd.DataFrame] = None,
    pci_blacklist_df: Optional[pd.DataFrame] = None,
    ca_imbalance_df: Optional[pd.DataFrame] = None,
    crossed_feeder_df: Optional[pd.DataFrame] = None,
    interference_gdf: Optional[gpd.GeoDataFrame] = None,
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
        no_coverage_per_band_gdfs: Dict of band -> GeoDataFrame with no coverage per band
        low_coverage_gdfs: Dict of band -> GeoDataFrame with low coverage areas
        overshooting_grid_df: DataFrame with overshooting grid bins (for lazy loading)
        undershooting_grid_df: DataFrame with undershooting grid bins (for lazy loading)
        cell_hulls_gdf: GeoDataFrame with cell coverage hull polygons
        pci_confusions_df: DataFrame with PCI confusion issues
        pci_collisions_df: DataFrame with PCI collision issues
        pci_blacklist_df: DataFrame with PCI blacklist suggestions
        ca_imbalance_df: DataFrame with CA imbalance issues
        crossed_feeder_df: DataFrame with crossed feeder issues
        interference_gdf: GeoDataFrame with interference cluster polygons
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
        overshooting_df, undershooting_df, no_coverage_gdf, low_coverage_gdfs,
        no_coverage_per_band_gdfs=no_coverage_per_band_gdfs,
        pci_confusions_df=pci_confusions_df,
        pci_collisions_df=pci_collisions_df,
        pci_blacklist_df=pci_blacklist_df,
        ca_imbalance_df=ca_imbalance_df,
        crossed_feeder_df=crossed_feeder_df,
        interference_gdf=interference_gdf,
    )

    # Create feature groups for each layer
    layers = {}

    # Build cell coordinates lookup for line drawing
    cell_coords = _build_cell_coords_lookup(gis_df)

    # Build cell geometries lookup for displaying cell sectors
    cell_geometries = _build_cell_geometries_lookup(gis_df)

    # Build cell-to-band mapping for filtering
    cell_band_map = {}
    if gis_df is not None and len(gis_df) > 0:
        name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'
        band_col = 'band' if 'band' in gis_df.columns else 'Band'
        if band_col in gis_df.columns:
            for _, row in gis_df.iterrows():
                cell_name = str(row.get(name_col, ''))
                band = row.get(band_col, '')
                if cell_name and band:
                    cell_band_map[cell_name] = str(band)

    # Extract unique bands from GIS data for filter dropdown
    all_bands = []
    if gis_df is not None and 'band' in gis_df.columns:
        all_bands = gis_df['band'].dropna().unique().tolist()
    elif gis_df is not None and 'Band' in gis_df.columns:
        all_bands = gis_df['Band'].dropna().unique().tolist()

    # Collect all line data for JavaScript
    all_line_data = {}

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

    # Layer 3b: No coverage per band
    if no_coverage_per_band_gdfs:
        combined_no_cov_pb = []
        for band, gdf in no_coverage_per_band_gdfs.items():
            if len(gdf) > 0:
                gdf_copy = gdf.copy()
                if 'band' not in gdf_copy.columns:
                    gdf_copy['band'] = band
                combined_no_cov_pb.append(gdf_copy)
        if combined_no_cov_pb:
            combined_gdf = gpd.GeoDataFrame(pd.concat(combined_no_cov_pb, ignore_index=True))
            layers['no_coverage_per_band'] = _add_no_coverage_per_band_layer(m, combined_gdf)

    # Layer 4: Low coverage areas (all bands combined) with serving cell lines
    if low_coverage_gdfs:
        # Combine all bands into one GeoDataFrame
        combined_low_cov = []
        for band, gdf in low_coverage_gdfs.items():
            if len(gdf) > 0:
                gdf_copy = gdf.copy()
                if 'band' not in gdf_copy.columns:
                    gdf_copy['band'] = band
                combined_low_cov.append(gdf_copy)
        if combined_low_cov:
            combined_gdf = gpd.GeoDataFrame(pd.concat(combined_low_cov, ignore_index=True))
            layer, line_data = _add_low_coverage_layer(m, combined_gdf, 'all', cell_coords)
            layers['low_coverage'] = layer
            all_line_data.update(line_data)

    # Layer 5: Interference clusters with offending cell lines
    if interference_gdf is not None and len(interference_gdf) > 0:
        layer, line_data = _add_interference_layer(m, interference_gdf, cell_coords)
        layers['interference'] = layer
        all_line_data.update(line_data)

    # Layer 6: PCI Confusions
    if pci_confusions_df is not None and len(pci_confusions_df) > 0:
        layer, line_data = _add_pci_confusions_layer(m, pci_confusions_df, cell_coords, cell_geometries)
        layers['pci_confusions'] = layer
        all_line_data.update(line_data)

    # Layer 7: PCI Collisions (Exact only)
    if pci_collisions_df is not None and len(pci_collisions_df) > 0:
        layer, line_data = _add_pci_collisions_layer(m, pci_collisions_df, cell_coords, cell_geometries)
        layers['pci_collisions'] = layer
        all_line_data.update(line_data)

    # Layer 7b: PCI MOD3 Collisions
    if pci_collisions_df is not None and len(pci_collisions_df) > 0:
        layer, line_data = _add_pci_mod3_collisions_layer(m, pci_collisions_df, cell_coords, cell_geometries)
        layers['pci_mod3_collisions'] = layer
        all_line_data.update(line_data)

    # Layer 8: PCI Blacklist Suggestions
    if pci_blacklist_df is not None and len(pci_blacklist_df) > 0:
        layer, line_data = _add_pci_blacklist_layer(m, pci_blacklist_df, cell_coords, cell_geometries, cell_band_map)
        layers['pci_blacklist'] = layer
        all_line_data.update(line_data)

    # Layer 9: CA Imbalance with hull toggle
    all_hull_data = {}
    if ca_imbalance_df is not None and len(ca_imbalance_df) > 0:
        layer, hull_data = _add_ca_imbalance_layer(m, ca_imbalance_df, cell_coords, cell_hulls_gdf, cell_geometries)
        layers['ca_imbalance'] = layer
        all_hull_data.update(hull_data)

    # Layer 10: Crossed Feeders
    if crossed_feeder_df is not None and len(crossed_feeder_df) > 0:
        layer, line_data = _add_crossed_feeder_layer(m, crossed_feeder_df, cell_coords, cell_geometries)
        layers['crossed_feeders'] = layer
        all_line_data.update(line_data)

    # Prepare inline grid data (embedded in HTML to avoid CORS issues with file:// protocol)
    overshooting_inline_data = {}
    undershooting_inline_data = {}

    if overshooting_grid_df is not None and len(overshooting_grid_df) > 0:
        overshooting_inline_data = prepare_grid_data_inline(overshooting_grid_df)
        logger.info("Prepared overshooting grid data for inline embedding", cells=len(overshooting_inline_data))

    if undershooting_grid_df is not None and len(undershooting_grid_df) > 0:
        undershooting_inline_data = prepare_grid_data_inline(undershooting_grid_df)
        logger.info("Prepared undershooting grid data for inline embedding", cells=len(undershooting_inline_data))

    # Add grid loading JavaScript with embedded data (no fetch needed, no CORS issues)
    if overshooting_inline_data or undershooting_inline_data:
        _add_grid_loading_javascript(m, overshooting_inline_data, undershooting_inline_data)

    # Add line toggle JavaScript if we have any line data or hull data
    if all_line_data or all_hull_data:
        _add_line_toggle_javascript(m, all_line_data, cell_coords, all_hull_data)

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
    _add_filter_panel(m, stats, all_bands=all_bands)

    # Add recommendations table panel
    _add_recommendations_table_panel(
        m, cell_coords,
        overshooting_df=overshooting_df,
        undershooting_df=undershooting_df,
        pci_confusions_df=pci_confusions_df,
        pci_collisions_df=pci_collisions_df,
        pci_blacklist_df=pci_blacklist_df,
        ca_imbalance_df=ca_imbalance_df,
        crossed_feeder_df=crossed_feeder_df,
    )

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
    no_coverage_per_band_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    pci_confusions_df: Optional[pd.DataFrame] = None,
    pci_collisions_df: Optional[pd.DataFrame] = None,
    pci_blacklist_df: Optional[pd.DataFrame] = None,
    ca_imbalance_df: Optional[pd.DataFrame] = None,
    crossed_feeder_df: Optional[pd.DataFrame] = None,
    interference_gdf: Optional[gpd.GeoDataFrame] = None,
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
        'no_coverage_per_band': {
            'total': 0,
            'by_band': {},
            'total_area_km2': 0,
        },
        'low_coverage': {
            'total': 0,
            'by_band': {},
            'total_area_km2': 0,
        },
        'pci_confusions': {'total': 0, 'by_severity': {}, 'by_band': {}},
        'pci_collisions': {'total': 0, 'by_severity': {}, 'by_band': {}},
        'pci_blacklist': {'total': 0},
        'ca_imbalance': {'total': 0, 'by_severity': {}, 'by_band': {}},
        'crossed_feeders': {'total': 0, 'by_severity': {}, 'by_band': {}},
        'interference': {'total': 0, 'by_severity': {}, 'by_band': {}},
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

    # No coverage per band stats
    if no_coverage_per_band_gdfs:
        for band, gdf in no_coverage_per_band_gdfs.items():
            if len(gdf) > 0:
                stats['no_coverage_per_band']['total'] += len(gdf)
                stats['no_coverage_per_band']['by_band'][str(band)] = len(gdf)
                if 'area_km2' in gdf.columns:
                    stats['no_coverage_per_band']['total_area_km2'] += gdf['area_km2'].sum()

    # Low coverage stats
    if low_coverage_gdfs:
        for band, gdf in low_coverage_gdfs.items():
            if len(gdf) > 0:
                stats['low_coverage']['total'] += len(gdf)
                stats['low_coverage']['by_band'][str(band)] = len(gdf)
                if 'area_km2' in gdf.columns:
                    stats['low_coverage']['total_area_km2'] += gdf['area_km2'].sum()

    # PCI stats
    if pci_confusions_df is not None and len(pci_confusions_df) > 0:
        stats['pci_confusions']['total'] = len(pci_confusions_df)
        if 'severity_category' in pci_confusions_df.columns:
            stats['pci_confusions']['by_severity'] = pci_confusions_df['severity_category'].value_counts().to_dict()
        if 'band' in pci_confusions_df.columns:
            stats['pci_confusions']['by_band'] = pci_confusions_df['band'].value_counts().to_dict()
    if pci_collisions_df is not None and len(pci_collisions_df) > 0:
        stats['pci_collisions']['total'] = len(pci_collisions_df)
        if 'severity_category' in pci_collisions_df.columns:
            stats['pci_collisions']['by_severity'] = pci_collisions_df['severity_category'].value_counts().to_dict()
        if 'band' in pci_collisions_df.columns:
            stats['pci_collisions']['by_band'] = pci_collisions_df['band'].value_counts().to_dict()
    if pci_blacklist_df is not None:
        stats['pci_blacklist']['total'] = len(pci_blacklist_df)

    # CA Imbalance stats
    if ca_imbalance_df is not None and len(ca_imbalance_df) > 0:
        stats['ca_imbalance']['total'] = len(ca_imbalance_df)
        if 'severity_category' in ca_imbalance_df.columns:
            stats['ca_imbalance']['by_severity'] = ca_imbalance_df['severity_category'].value_counts().to_dict()
        if 'coverage_band' in ca_imbalance_df.columns:
            stats['ca_imbalance']['by_band'] = ca_imbalance_df['coverage_band'].value_counts().to_dict()

    # Crossed Feeder stats
    if crossed_feeder_df is not None and len(crossed_feeder_df) > 0:
        flagged = crossed_feeder_df[crossed_feeder_df.get('flagged', False) == True] if 'flagged' in crossed_feeder_df.columns else crossed_feeder_df
        stats['crossed_feeders']['total'] = len(flagged)
        if 'severity_category' in flagged.columns:
            stats['crossed_feeders']['by_severity'] = flagged['severity_category'].value_counts().to_dict()
        if 'band' in flagged.columns:
            stats['crossed_feeders']['by_band'] = flagged['band'].value_counts().to_dict()

    # Interference stats
    if interference_gdf is not None and len(interference_gdf) > 0:
        stats['interference']['total'] = len(interference_gdf)
        if 'severity_category' in interference_gdf.columns:
            stats['interference']['by_severity'] = interference_gdf['severity_category'].value_counts().to_dict()
        if 'band' in interference_gdf.columns:
            stats['interference']['by_band'] = interference_gdf['band'].value_counts().to_dict()

    return stats


def _add_all_cells_layer(
    m: folium.Map,
    gis_df: pd.DataFrame,
) -> folium.FeatureGroup:
    """Add a layer showing all cells with their actual sector geometry."""
    from shapely import wkt
    from shapely.geometry import mapping

    layer = folium.FeatureGroup(name='All Cells', show=False)

    # Standard column names (lowercase)
    cell_name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'
    lat_col = 'latitude' if 'latitude' in gis_df.columns else 'Latitude'
    lon_col = 'longitude' if 'longitude' in gis_df.columns else 'Longitude'
    band_col = 'band' if 'band' in gis_df.columns else 'Band'
    site_col = 'site' if 'site' in gis_df.columns else 'Site'
    geom_col = 'geometry' if 'geometry' in gis_df.columns else None
    bearing_col = 'bearing' if 'bearing' in gis_df.columns else 'Bearing'

    for _, cell in gis_df.iterrows():
        cell_name = str(cell.get(cell_name_col, 'N/A'))
        lat = cell.get(lat_col)
        lon = cell.get(lon_col)
        band = cell.get(band_col, 'N/A')
        site = cell.get(site_col, '')
        bearing = cell.get(bearing_col, 0) if bearing_col in gis_df.columns else 0

        if pd.isna(lat) or pd.isna(lon):
            continue

        tooltip = f"{cell_name} - {band}"

        # Try to use actual geometry if available
        has_geometry = False
        if geom_col and cell.get(geom_col):
            geom_value = cell.get(geom_col)
            try:
                # Handle WKT string geometry
                if isinstance(geom_value, str) and geom_value.startswith('POLYGON'):
                    geom = wkt.loads(geom_value)
                    # GeoJSON expects [lon, lat] but folium expects [lat, lon] for some features
                    # For GeoJson, we use the geometry directly
                    folium.GeoJson(
                        mapping(geom),
                        style_function=lambda x: {
                            'fillColor': '#6c757d',
                            'color': '#495057',
                            'weight': 1,
                            'fillOpacity': 0.4,
                        },
                        tooltip=tooltip,
                    ).add_to(layer)
                    has_geometry = True
                # Handle shapely geometry object
                elif hasattr(geom_value, '__geo_interface__'):
                    folium.GeoJson(
                        geom_value.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': '#6c757d',
                            'color': '#495057',
                            'weight': 1,
                            'fillOpacity': 0.4,
                        },
                        tooltip=tooltip,
                    ).add_to(layer)
                    has_geometry = True
            except Exception:
                pass  # Fall back to calculated sector

        # Fall back to calculated sector if no geometry
        if not has_geometry and pd.notna(bearing) and bearing != 0:
            sector_points = calculate_sector_points(lat, lon, bearing, 400, 40.0)
            folium.Polygon(
                locations=sector_points,
                color='#495057',
                weight=1,
                fill=True,
                fillColor='#6c757d',
                fillOpacity=0.4,
                tooltip=tooltip,
            ).add_to(layer)

        # Draw small center marker for cell location
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='#343a40',
            fill=True,
            fillColor='#495057',
            fillOpacity=0.8,
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
    # Use cell_name as fallback for joining if cilac/CILAC not present
    cilac_col = next((c for c in ['cilac', 'CILAC', 'cell_name'] if c in hulls_gdf.columns), 'cell_name')
    area_col = 'area_km2' if 'area_km2' in hulls_gdf.columns else None

    # Merge band info from GIS data
    hulls_with_band = hulls_gdf.copy()
    if gis_df is not None:
        gis_cell_name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'
        gis_band_col = 'band' if 'band' in gis_df.columns else 'Band'

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
        gis_cell_name_col = next((c for c in ['cell_name', 'CILAC'] if c in gis_df.columns), 'cell_name')
        gis_name_col = next((c for c in ['name', 'Name'] if c in gis_df.columns), None)
        gis_band_col = next((c for c in ['band', 'Band'] if c in gis_df.columns), None)
        gis_azimuth_col = next((c for c in ['Bearing', 'bearing', 'azimuth_deg', 'azimuth'] if c in gis_df.columns), None)

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
        az_col = next((c for c in ['bearing', 'Bearing', 'azimuth_deg', 'azimuth'] if c in gis_df.columns), None)
        name_col = next((c for c in ['name', 'Name'] if c in gis_df.columns), None)
        band_col = next((c for c in ['band', 'Band'] if c in gis_df.columns), None)

        gis_cols = [cell_name_col]
        if lat_col in gis_df.columns:
            gis_cols.extend([lat_col, lon_col])
        if az_col and az_col in gis_df.columns:
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
        if az_col and az_col != 'azimuth_deg' and az_col in gis_subset.columns:
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


def _add_no_coverage_per_band_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
) -> folium.FeatureGroup:
    """Add no coverage per band layer."""
    layer = folium.FeatureGroup(name='No Coverage (Per Band)', show=False)

    # Color mapping for different bands
    band_colors = {
        'L700': '#17a2b8',    # Teal
        'L800': '#20c997',    # Cyan
        'L1800': '#6f42c1',   # Purple
        'L2100': '#e83e8c',   # Pink
        'L2600': '#fd7e14',   # Orange
    }
    default_color = '#17a2b8'

    for idx, row in gdf.iterrows():
        geom = row['geometry']
        cluster_id = row.get('cluster_id', idx)
        area = row.get('area_km2', 0)
        n_points = row.get('n_points', 0)
        band = row.get('band', 'Unknown')

        fill_color = band_colors.get(band, default_color)
        border_color = '#0d6efd'

        popup_html = f"""
        <div style="font-family: Arial; width: 220px;" data-band="{band}">
            <h4 style="margin: 0 0 8px 0; color: {fill_color};">No Coverage ({band})</h4>
            <div style="background: #e3f2fd; padding: 8px; border-radius: 4px;">
                <strong>Band:</strong> {band}<br>
                <strong>Cluster ID:</strong> {cluster_id}<br>
                <strong>Area:</strong> {area:.2f} km²<br>
                <strong>Grid points:</strong> {n_points}<br>
                <em style="color: #666; font-size: 0.9em;">
                    Area where {band} has no coverage (other bands may cover)
                </em>
            </div>
        </div>
        """

        folium.GeoJson(
            geom,
            style_function=lambda x, fc=fill_color, bc=border_color: {
                'fillColor': fc,
                'color': bc,
                'weight': 2,
                'fillOpacity': 0.4,
                'dashArray': '5,5',
            },
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"No Coverage ({band}) - {area:.2f} km²",
        ).add_to(layer)

    layer.add_to(m)
    return layer


def _add_low_coverage_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    band: str,
    cell_coords: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add low coverage layer with serving cell line toggle.

    Returns:
        Tuple of (FeatureGroup, line_data dict for this layer)
    """
    # Use simple name for combined layer
    layer_name = 'Low Coverage' if band == 'all' else f'Low Coverage ({band} MHz)'
    layer = folium.FeatureGroup(name=layer_name, show=False)
    line_data = {}

    for idx, row in gdf.iterrows():
        geom = row['geometry']
        cluster_id = row.get('cluster_id', idx)
        area = row.get('area_km2', 0)
        n_points = row.get('n_points', 0)
        serving_cells = row.get('serving_cells', '')
        serving_cell_names = row.get('serving_cell_names', '')
        n_serving_cells = row.get('n_serving_cells', 0)
        centroid_lat = row.get('centroid_lat', geom.centroid.y if geom else 0)
        centroid_lon = row.get('centroid_lon', geom.centroid.x if geom else 0)
        # Get band from row for combined layers
        row_band = row.get('band', band) if band == 'all' else band

        # Build serving cells list and line data
        serving_cells_html = ""
        has_line_data = False
        feature_key = f"low_coverage_{row_band}_{cluster_id}"

        if serving_cells and cell_coords:
            cells_list = [c.strip() for c in serving_cells.split(',') if c.strip()]

            # Build targets for line data
            targets = []
            for cell_name in cells_list:
                if cell_name in cell_coords:
                    targets.append({
                        'coords': cell_coords[cell_name],
                        'name': cell_name,
                        'color': LINE_COLORS['low_coverage']
                    })

            if targets:
                has_line_data = True
                line_data[feature_key] = {
                    'source': [centroid_lat, centroid_lon],
                    'targets': targets,
                    'color': LINE_COLORS['low_coverage'],
                    'style': 'dashed'
                }

            # Build display HTML
            cells_display_parts = cells_list[:5]
            cells_display = ', '.join(cells_display_parts)
            if n_serving_cells > 5:
                cells_display += f' (+{n_serving_cells - 5} more)'
            serving_cells_html = f"""
                <strong>Serving Cells:</strong> {cells_display}<br>
                <strong>Total Cells:</strong> {n_serving_cells}
            """

        # Build popup with optional line toggle button
        line_button_html = ""
        if has_line_data:
            line_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="lineBtn_{feature_key}"
                        data-show-text="Show Serving Cells"
                        data-hide-text="Hide Serving Cells"
                        onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                        style="background: {LINE_COLORS['low_coverage']}; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Serving Cells
                </button>
                <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;" data-band="{row_band}">
            <h4 style="margin: 0 0 8px 0; color: #fd7e14;">Low Coverage Area</h4>
            <div style="background: #ffe5d0; padding: 8px; border-radius: 4px;">
                <strong>Band:</strong> {row_band}<br>
                <strong>Cluster ID:</strong> {cluster_id}<br>
                <strong>Area:</strong> {area:.2f} km²<br>
                <strong>Grid points:</strong> {n_points}<br>
                {serving_cells_html}
            </div>
            {line_button_html}
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
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Low Coverage {row_band} - {area:.2f} km²",
        ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_interference_layer(
    m: folium.Map,
    gdf: gpd.GeoDataFrame,
    cell_coords: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add interference cluster layer with offending cell line toggle.

    Args:
        m: Folium map object
        gdf: GeoDataFrame with interference cluster polygons
        cell_coords: Dict mapping cell_name to [lat, lon]

    Returns:
        Tuple of (FeatureGroup, line_data dict for this layer)
    """
    layer = folium.FeatureGroup(name='Interference Clusters', show=False)
    line_data = {}

    if gdf is None or len(gdf) == 0:
        layer.add_to(m)
        return layer, line_data

    for idx, row in gdf.iterrows():
        geom = row.get('geometry')
        if geom is None or geom.is_empty:
            continue

        cluster_id = row.get('cluster_id', idx)
        band = row.get('band', 'Unknown')
        n_grids = row.get('n_grids', 0)
        n_cells = row.get('n_cells', 0)
        cells = row.get('cells', [])
        centroid_lat = row.get('centroid_lat', geom.centroid.y if geom else 0)
        centroid_lon = row.get('centroid_lon', geom.centroid.x if geom else 0)
        area_km2 = row.get('area_km2', 0)
        avg_rsrp = row.get('avg_rsrp', 0)

        # Parse cells if it's a string
        if isinstance(cells, str):
            cells = [c.strip() for c in cells.split(',') if c.strip()]

        # Build line data for offending cells
        feature_key = f"interference_{cluster_id}"
        has_line_data = False

        if cells and cell_coords:
            targets = []
            for cell_name in cells:
                if cell_name in cell_coords:
                    targets.append({
                        'coords': cell_coords[cell_name],
                        'name': cell_name,
                        'color': LINE_COLORS['interference']
                    })

            if targets:
                has_line_data = True
                line_data[feature_key] = {
                    'source': [centroid_lat, centroid_lon],
                    'targets': targets,
                    'color': LINE_COLORS['interference'],
                    'style': 'dashed'
                }

        # Build cells display
        cells_display = ', '.join(cells[:5])
        if len(cells) > 5:
            cells_display += f' (+{len(cells) - 5} more)'

        # Build popup with optional line toggle button
        line_button_html = ""
        if has_line_data:
            line_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="lineBtn_{feature_key}"
                        data-show-text="Show Offending Cells"
                        data-hide-text="Hide Offending Cells"
                        onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                        style="background: {LINE_COLORS['interference']}; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Offending Cells
                </button>
                <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            """

        # Get severity info
        severity_score = row.get('severity_score', 0)
        severity_category = row.get('severity_category', 'MEDIUM')
        severity_color = SEVERITY_COLORS.get(severity_category, '#6c757d')

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;" data-band="{band}" data-severity="{severity_category}">
            <h4 style="margin: 0 0 8px 0; color: #dc3545;">
                Interference Cluster
                <span style="float: right; font-size: 12px; background: {severity_color};
                             color: white; padding: 2px 8px; border-radius: 4px;">{severity_category}</span>
            </h4>
            <div style="background: #f8d7da; padding: 8px; border-radius: 4px;">
                <strong>Cluster ID:</strong> {cluster_id}<br>
                <strong>Band:</strong> {band}<br>
                <strong>Severity Score:</strong> {severity_score:.3f}<br>
                <strong>Area:</strong> {area_km2:.2f} km²<br>
                <strong>Grid points:</strong> {n_grids}<br>
                <strong>Cells involved:</strong> {n_cells}<br>
                <strong>Cells:</strong> {cells_display}<br>
                <strong>Avg RSRP:</strong> {avg_rsrp:.1f} dBm
            </div>
            {line_button_html}
        </div>
        """

        folium.GeoJson(
            geom,
            style_function=lambda x: {
                'fillColor': '#dc3545',
                'color': '#721c24',
                'weight': 2,
                'fillOpacity': 0.35,
            },
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Interference {band} - {area_km2:.2f} km²",
        ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_pci_confusions_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    cell_geometries: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add PCI confusions layer with lines to offending cells.

    Args:
        m: Folium map object
        df: DataFrame with PCI confusion data (serving, neighbors, confusion_pci, band)
        cell_coords: Dict mapping cell_name to [lat, lon]
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]

    Returns:
        Tuple of (FeatureGroup, line_data dict)
    """
    layer = folium.FeatureGroup(name='PCI Confusions', show=False)
    line_data = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    cell_geometries = cell_geometries or {}

    for idx, row in df.iterrows():
        serving = row.get('serving', '')
        band = row.get('band', 'Unknown')
        confusion_pci = row.get('confusion_pci', 0)
        neighbors_str = row.get('neighbors', '')
        group_size = row.get('group_size', 0)
        severity_sum = row.get('severity_act_sum_excl_max', 0)
        severity_score = row.get('severity_score', 0)
        severity_cat = row.get('severity_category', 'MEDIUM')

        # Get serving cell coordinates
        if serving not in cell_coords:
            continue
        source_coords = cell_coords[serving]

        # Parse neighbors
        neighbors = [n.strip() for n in neighbors_str.split(',') if n.strip()]

        # Build line data
        feature_key = f"pci_confusion_{idx}"
        targets = []
        for neighbor in neighbors:
            if neighbor in cell_coords:
                targets.append({
                    'coords': cell_coords[neighbor],
                    'name': neighbor,
                    'color': LINE_COLORS['pci']
                })

        has_line_data = len(targets) > 0
        if has_line_data:
            line_data[feature_key] = {
                'source': source_coords,
                'targets': targets,
                'color': LINE_COLORS['pci'],
                'style': 'dotted'
            }

        # Build popup
        neighbors_display = ', '.join(neighbors[:5])
        if len(neighbors) > 5:
            neighbors_display += f' (+{len(neighbors) - 5} more)'

        line_button_html = ""
        if has_line_data:
            line_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="lineBtn_{feature_key}"
                        data-show-text="Show Offending Cells"
                        data-hide-text="Hide Offending Cells"
                        onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                        style="background: {LINE_COLORS['pci']}; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Offending Cells
                </button>
                <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;" data-band="{band}" data-severity="{severity_cat}">
            <h4 style="margin: 0 0 8px 0; color: {LINE_COLORS['pci']};">PCI Confusion
                <span style="float: right; font-size: 11px; background: {'#dc3545' if severity_cat == 'CRITICAL' else '#fd7e14' if severity_cat == 'HIGH' else '#ffc107' if severity_cat == 'MEDIUM' else '#28a745'};
                             color: white; padding: 2px 6px; border-radius: 3px;">{severity_cat}</span>
            </h4>
            <div style="background: #f3e5f5; padding: 8px; border-radius: 4px;">
                <strong>Source Cell:</strong> {serving}<br>
                <strong>Band:</strong> {band}<br>
                <strong>Confusion PCI:</strong> {confusion_pci}<br>
                <strong>Group Size:</strong> {group_size}<br>
                <strong>Neighbors:</strong> {neighbors_display}<br>
                <strong>Severity Score:</strong> {severity_score:.3f}
            </div>
            <div style="background: #e1bee7; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Issue:</strong> Multiple cells share the same PCI, causing handover confusion.
            </div>
            {line_button_html}
        </div>
        """

        # Add cell sector polygon if geometry available, otherwise use marker
        if serving in cell_geometries:
            folium.Polygon(
                locations=cell_geometries[serving],
                color=LINE_COLORS['pci'],
                weight=2,
                fill=True,
                fillColor=LINE_COLORS['pci'],
                fillOpacity=0.5,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"PCI Confusion: {serving} (PCI {confusion_pci})",
            ).add_to(layer)
        else:
            folium.CircleMarker(
                location=source_coords,
                radius=8,
                color=LINE_COLORS['pci'],
                fill=True,
                fillColor=LINE_COLORS['pci'],
                fillOpacity=0.7,
                weight=2,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"PCI Confusion: {serving} (PCI {confusion_pci})",
            ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_pci_collisions_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    cell_geometries: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add PCI collisions layer with lines between cell pairs (EXACT collisions only).

    Args:
        m: Folium map object
        df: DataFrame with PCI collision data (cell_a, cell_b, pci, band, pair_weight)
        cell_coords: Dict mapping cell_name to [lat, lon]
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]

    Returns:
        Tuple of (FeatureGroup, line_data dict)
    """
    layer = folium.FeatureGroup(name='PCI Collision', show=False)
    line_data = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    # Filter for exact collisions only
    if 'conflict_type' in df.columns:
        df = df[df['conflict_type'] == 'exact'].copy()

    if len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    cell_geometries = cell_geometries or {}

    for idx, row in df.iterrows():
        cell_a = row.get('cell_a', '')
        cell_b = row.get('cell_b', '')
        pci_a = row.get('pci_a', 0)
        pci_b = row.get('pci_b', 0)
        band = row.get('band', 'Unknown')
        conflict_type = row.get('conflict_type', 'Unknown')
        hop_type = row.get('hop_type', '1-hop')
        pair_weight = row.get('pair_weight', 0)
        severity_score = row.get('severity_score', 0)
        severity_cat = row.get('severity_category', 'MEDIUM')
        severity_color = SEVERITY_COLORS.get(severity_cat, '#6c757d')

        # Get cell coordinates
        if cell_a not in cell_coords or cell_b not in cell_coords:
            continue

        coords_a = cell_coords[cell_a]
        coords_b = cell_coords[cell_b]

        # Calculate midpoint for marker
        mid_lat = (coords_a[0] + coords_b[0]) / 2
        mid_lon = (coords_a[1] + coords_b[1]) / 2

        # Build line data
        feature_key = f"pci_collision_{idx}"
        line_data[feature_key] = {
            'source': [mid_lat, mid_lon],
            'targets': [
                {'coords': coords_a, 'name': cell_a, 'color': LINE_COLORS['pci']},
                {'coords': coords_b, 'name': cell_b, 'color': LINE_COLORS['pci']}
            ],
            'color': LINE_COLORS['pci'],
            'style': 'dotted'
        }

        line_button_html = f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="lineBtn_{feature_key}"
                    data-show-text="Show Cell Pair"
                    data-hide-text="Hide Cell Pair"
                    onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                    style="background: {LINE_COLORS['pci']}; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Show Cell Pair
            </button>
            <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
        </div>
        """

        # Determine issue description based on conflict type and hop
        if hop_type == '2-hop':
            hop_note = " (via shared neighbor - not direct neighbors)"
        else:
            hop_note = ""

        if conflict_type == 'exact':
            issue_desc = f"Both cells use PCI {pci_a}{hop_note} - one must change."
        else:
            issue_desc = f"PCIs {pci_a} and {pci_b} cause {conflict_type} interference{hop_note}."

        # Badge for hop type
        hop_badge = f'<span style="background: #6c757d; color: white; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-left: 4px;">{hop_type}</span>' if hop_type == '2-hop' else ''

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;" data-band="{band}" data-severity="{severity_cat}">
            <h4 style="margin: 0 0 8px 0; color: {LINE_COLORS['pci']};">
                PCI Collision ({conflict_type}){hop_badge}
                <span style="float: right; font-size: 12px; background: {severity_color};
                             color: white; padding: 2px 8px; border-radius: 4px;">{severity_cat}</span>
            </h4>
            <div style="background: #f3e5f5; padding: 8px; border-radius: 4px;">
                <strong>{cell_a}:</strong> PCI {pci_a}<br>
                <strong>{cell_b}:</strong> PCI {pci_b}<br>
                <strong>Band:</strong> {band}<br>
                <strong>Hop Type:</strong> {hop_type}<br>
                <strong>Severity Score:</strong> {severity_score:.3f}
            </div>
            <div style="background: #e1bee7; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Issue:</strong> {issue_desc}
            </div>
            {line_button_html}
        </div>
        """

        # Add cell sector polygons for both cells if geometry available
        has_geom_a = cell_a in cell_geometries
        has_geom_b = cell_b in cell_geometries

        if has_geom_a or has_geom_b:
            # Draw polygons for cells that have geometry
            if has_geom_a:
                folium.Polygon(
                    locations=cell_geometries[cell_a],
                    color=LINE_COLORS['pci'],
                    weight=2,
                    fill=True,
                    fillColor=LINE_COLORS['pci'],
                    fillOpacity=0.5,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"PCI Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
                ).add_to(layer)
            if has_geom_b:
                folium.Polygon(
                    locations=cell_geometries[cell_b],
                    color='#ce93d8',
                    weight=2,
                    fill=True,
                    fillColor='#ce93d8',
                    fillOpacity=0.5,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"PCI Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
                ).add_to(layer)
        else:
            # Fallback to marker at midpoint
            folium.CircleMarker(
                location=[mid_lat, mid_lon],
                radius=6,
                color=LINE_COLORS['pci'],
                fill=True,
                fillColor='#ce93d8',
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"PCI Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
            ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_pci_mod3_collisions_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    cell_geometries: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add PCI MOD3 collisions layer with lines between cell pairs (MOD3 conflicts only).

    MOD3 conflicts cause PSS (Primary Sync Signal) interference per 3GPP TS 36.211.
    Intra-site MOD3 conflicts are more severe because cells are co-located.

    Args:
        m: Folium map object
        df: DataFrame with PCI collision data (cell_a, cell_b, pci, band, pair_weight)
        cell_coords: Dict mapping cell_name to [lat, lon]
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]

    Returns:
        Tuple of (FeatureGroup, line_data dict)
    """
    layer = folium.FeatureGroup(name='PCI Collision (MOD3)', show=False)
    line_data = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    # Filter for mod3 collisions only
    if 'conflict_type' in df.columns:
        df = df[df['conflict_type'] == 'mod3'].copy()

    if len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    cell_geometries = cell_geometries or {}

    # Use a distinct color for MOD3 conflicts (orange/amber to differentiate from exact collisions)
    mod3_color = '#ff9800'  # Orange
    mod3_fill_color = '#ffcc80'  # Light orange

    for idx, row in df.iterrows():
        cell_a = row.get('cell_a', '')
        cell_b = row.get('cell_b', '')
        pci_a = row.get('pci_a', 0)
        pci_b = row.get('pci_b', 0)
        band = row.get('band', 'Unknown')
        conflict_type = row.get('conflict_type', 'mod3')
        pair_weight = row.get('pair_weight', 0)
        severity_score = row.get('severity_score', 0)
        severity_cat = row.get('severity_category', 'MEDIUM')
        severity_color = SEVERITY_COLORS.get(severity_cat, '#6c757d')
        is_intra_site = row.get('intra_site', False)
        hop_type = row.get('hop_type', '1-hop')

        # Get cell coordinates
        if cell_a not in cell_coords or cell_b not in cell_coords:
            continue

        coords_a = cell_coords[cell_a]
        coords_b = cell_coords[cell_b]

        # Calculate midpoint for marker
        mid_lat = (coords_a[0] + coords_b[0]) / 2
        mid_lon = (coords_a[1] + coords_b[1]) / 2

        # Build line data
        feature_key = f"pci_mod3_{idx}"
        line_data[feature_key] = {
            'source': [mid_lat, mid_lon],
            'targets': [
                {'coords': coords_a, 'name': cell_a, 'color': mod3_color},
                {'coords': coords_b, 'name': cell_b, 'color': mod3_color}
            ],
            'color': mod3_color,
            'style': 'dotted'
        }

        line_button_html = f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="lineBtn_{feature_key}"
                    data-show-text="Show Cell Pair"
                    data-hide-text="Hide Cell Pair"
                    onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                    style="background: {mod3_color}; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Show Cell Pair
            </button>
            <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
        </div>
        """

        # Intra-site indicator
        site_indicator = "INTRA-SITE (co-located)" if is_intra_site else "INTER-SITE"
        site_note = "Co-located cells cause severe PSS interference" if is_intra_site else "Distant cells have reduced interference impact"

        # Hop type indicator
        if hop_type == '2-hop':
            hop_note = " (via shared neighbor)"
            hop_badge = f'<span style="background: #6c757d; color: white; padding: 1px 4px; border-radius: 3px; font-size: 10px; margin-left: 4px;">{hop_type}</span>'
        else:
            hop_note = ""
            hop_badge = ""

        popup_html = f"""
        <div style="font-family: Arial; width: 300px;" data-band="{band}" data-severity="{severity_cat}">
            <h4 style="margin: 0 0 8px 0; color: {mod3_color};">
                PCI MOD3 Collision{hop_badge}
                <span style="float: right; font-size: 12px; background: {severity_color};
                             color: white; padding: 2px 8px; border-radius: 4px;">{severity_cat}</span>
            </h4>
            <div style="background: #fff3e0; padding: 8px; border-radius: 4px;">
                <strong>{cell_a}:</strong> PCI {pci_a} (mod3 = {pci_a % 3})<br>
                <strong>{cell_b}:</strong> PCI {pci_b} (mod3 = {pci_b % 3})<br>
                <strong>Band:</strong> {band}<br>
                <strong>Hop Type:</strong> {hop_type}{hop_note}<br>
                <strong>Site Relationship:</strong> {site_indicator}<br>
                <strong>Severity Score:</strong> {severity_score:.3f}
            </div>
            <div style="background: #ffe0b2; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Issue:</strong> PCIs {pci_a} and {pci_b} share mod3 value ({pci_a % 3}),
                causing PSS interference per 3GPP TS 36.211{hop_note}.<br>
                <strong>Note:</strong> {site_note}
            </div>
            {line_button_html}
        </div>
        """

        # Add cell sector polygons for both cells if geometry available
        has_geom_a = cell_a in cell_geometries
        has_geom_b = cell_b in cell_geometries

        if has_geom_a or has_geom_b:
            # Draw polygons for cells that have geometry
            if has_geom_a:
                folium.Polygon(
                    locations=cell_geometries[cell_a],
                    color=mod3_color,
                    weight=2,
                    fill=True,
                    fillColor=mod3_color,
                    fillOpacity=0.5,
                    popup=folium.Popup(popup_html, max_width=340),
                    tooltip=f"MOD3 Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
                ).add_to(layer)
            if has_geom_b:
                folium.Polygon(
                    locations=cell_geometries[cell_b],
                    color=mod3_fill_color,
                    weight=2,
                    fill=True,
                    fillColor=mod3_fill_color,
                    fillOpacity=0.5,
                    popup=folium.Popup(popup_html, max_width=340),
                    tooltip=f"MOD3 Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
                ).add_to(layer)
        else:
            # Fallback to marker at midpoint
            folium.CircleMarker(
                location=[mid_lat, mid_lon],
                radius=6,
                color=mod3_color,
                fill=True,
                fillColor=mod3_fill_color,
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"MOD3 Collision: {cell_a} (PCI {pci_a}) <-> {cell_b} (PCI {pci_b})",
            ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_pci_blacklist_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    cell_geometries: Optional[dict] = None,
    cell_band_map: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add PCI blacklist suggestions layer with lines from serving to neighbor.

    Args:
        m: Folium map object
        df: DataFrame with blacklist data (serving, neighbor, reason, confusion_pci)
        cell_coords: Dict mapping cell_name to [lat, lon]
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]
        cell_band_map: Dict mapping cell_name to band

    Returns:
        Tuple of (FeatureGroup, line_data dict)
    """
    layer = folium.FeatureGroup(name='PCI Blacklist Suggestions', show=False)
    line_data = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    cell_geometries = cell_geometries or {}
    cell_band_map = cell_band_map or {}

    for idx, row in df.iterrows():
        serving = row.get('serving', '')
        neighbor = row.get('neighbor', '')
        reason = row.get('reason', 'Unknown')
        confusion_pci = row.get('confusion_pci', 0)
        out_ho = row.get('out_ho', 0)
        in_ho = row.get('in_ho', 0)
        act_ho = row.get('act_ho', 0)

        # Get band from lookup or dataframe
        band = row.get('band', cell_band_map.get(serving, 'Unknown'))

        # Get cell coordinates
        if serving not in cell_coords:
            continue

        source_coords = cell_coords[serving]

        # Build line data
        feature_key = f"pci_blacklist_{idx}"
        has_line_data = False

        if neighbor in cell_coords:
            has_line_data = True
            line_data[feature_key] = {
                'source': source_coords,
                'targets': [
                    {'coords': cell_coords[neighbor], 'name': neighbor, 'color': LINE_COLORS['blacklist']}
                ],
                'color': LINE_COLORS['blacklist'],
                'style': 'dotted'
            }

        line_button_html = ""
        if has_line_data:
            line_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="lineBtn_{feature_key}"
                        data-show-text="Show Neighbor"
                        data-hide-text="Hide Neighbor"
                        onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                        style="background: {LINE_COLORS['blacklist']}; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Neighbor
                </button>
                <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;" data-band="{band}">
            <h4 style="margin: 0 0 8px 0; color: {LINE_COLORS['blacklist']};">Blacklist Suggestion</h4>
            <div style="background: #fff3e0; padding: 8px; border-radius: 4px;">
                <strong>Serving Cell:</strong> {serving}<br>
                <strong>Band:</strong> {band}<br>
                <strong>Neighbor to Blacklist:</strong> {neighbor}<br>
                <strong>Reason:</strong> {reason}<br>
                <strong>Confusion PCI:</strong> {confusion_pci}<br>
                <strong>Handovers (Out/In/Active):</strong> {out_ho:.2f}/{in_ho:.2f}/{act_ho:.2f}
            </div>
            <div style="background: #ffe0b2; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Recommendation:</strong> Add {neighbor} to blacklist for {serving}.
            </div>
            {line_button_html}
        </div>
        """

        # Add cell sector polygon if geometry available, otherwise use marker
        if serving in cell_geometries:
            folium.Polygon(
                locations=cell_geometries[serving],
                color=LINE_COLORS['blacklist'],
                weight=2,
                fill=True,
                fillColor=LINE_COLORS['blacklist'],
                fillOpacity=0.5,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Blacklist: {serving} -> {neighbor}",
            ).add_to(layer)
        else:
            folium.CircleMarker(
                location=source_coords,
                radius=5,
                color=LINE_COLORS['blacklist'],
                fill=True,
                fillColor=LINE_COLORS['blacklist'],
                fillOpacity=0.6,
                weight=2,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Blacklist: {serving} -> {neighbor}",
            ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


def _add_ca_imbalance_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    hulls_gdf: Optional[gpd.GeoDataFrame] = None,
    cell_geometries: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add CA imbalance layer with hull toggle to show coverage/capacity areas.

    Args:
        m: Folium map object
        df: DataFrame with CA imbalance data
        cell_coords: Dict mapping cell_name to [lat, lon]
        hulls_gdf: GeoDataFrame with cell hull geometries
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]

    Returns:
        Tuple of (FeatureGroup, hull_geometries dict for JavaScript)
    """
    layer = folium.FeatureGroup(name='CA Imbalance', show=False)
    hull_geometries = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, hull_geometries

    cell_geometries = cell_geometries or {}

    # Build hull lookup
    hull_lookup = {}
    if hulls_gdf is not None and len(hulls_gdf) > 0:
        name_col = 'cell_name' if 'cell_name' in hulls_gdf.columns else 'Name'
        for _, row in hulls_gdf.iterrows():
            cell_name = str(row.get(name_col, ''))
            if cell_name and row.geometry is not None:
                hull_lookup[cell_name] = row.geometry.__geo_interface__

    for idx, row in df.iterrows():
        coverage_cell = row.get('coverage_cell_name', '')
        capacity_cell = row.get('capacity_cell_name', '')
        ca_pair = row.get('ca_pair', 'Unknown')
        coverage_band = row.get('coverage_band', '')
        capacity_band = row.get('capacity_band', '')
        severity = row.get('severity', 'medium')
        coverage_area = row.get('coverage_area_km2', 0)
        capacity_area = row.get('capacity_area_km2', 0)
        coverage_ratio = row.get('coverage_ratio', 0)
        coverage_pct = row.get('coverage_percentage', 0)
        recommendation = row.get('recommendation', '')

        # Get coverage cell coordinates
        if coverage_cell not in cell_coords:
            continue
        source_coords = cell_coords[coverage_cell]

        # Build hull geometries for JavaScript
        feature_key = f"ca_imbalance_{idx}"
        has_hull_data = False

        coverage_hull = hull_lookup.get(coverage_cell)
        capacity_hull = hull_lookup.get(capacity_cell)

        if coverage_hull or capacity_hull:
            has_hull_data = True
            hull_geometries[feature_key] = {
                'coverage': coverage_hull,
                'capacity': capacity_hull
            }

        # Severity color mapping
        severity_color = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#17a2b8'
        }.get(severity.lower(), '#17a2b8')

        hull_button_html = ""
        if has_hull_data:
            hull_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="hullBtn_ca_imbalance_{idx}"
                        onclick="toggleHulls('{feature_key}', 'hullBtn_ca_imbalance_{idx}')"
                        style="background: #17a2b8; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Coverage Hulls
                </button>
                <div id="hullStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            <div style="margin-top: 8px; font-size: 10px; color: #666;">
                <span style="color: {LINE_COLORS['ca_coverage']};">&#9632;</span> Coverage Band &nbsp;
                <span style="color: {LINE_COLORS['ca_capacity']};">&#9632;</span> Capacity Band
            </div>
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 300px;" data-band="{coverage_band}">
            <h4 style="margin: 0 0 8px 0; color: #17a2b8;">
                CA Imbalance
                <span style="float: right; font-size: 12px; background: {severity_color};
                             color: white; padding: 2px 8px; border-radius: 4px;">{severity.upper()}</span>
            </h4>
            <div style="background: #d1ecf1; padding: 8px; border-radius: 4px;">
                <strong>Band:</strong> {coverage_band}<br>
                <strong>CA Pair:</strong> {ca_pair}<br>
                <strong>Coverage Cell:</strong> {coverage_cell} ({coverage_band})<br>
                <strong>Capacity Cell:</strong> {capacity_cell} ({capacity_band})<br>
                <strong>Coverage Area:</strong> {coverage_area:.2f} km²<br>
                <strong>Capacity Area:</strong> {capacity_area:.2f} km²<br>
                <strong>Coverage Ratio:</strong> {coverage_ratio:.1%}<br>
                <strong>Coverage %:</strong> {coverage_pct:.1f}%
            </div>
            <div style="background: #bee5eb; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Issue:</strong> Capacity band covers less than expected area relative to coverage band.
            </div>
            {hull_button_html}
        </div>
        """

        # Add cell sector polygon if geometry available, otherwise use marker
        if coverage_cell in cell_geometries:
            folium.Polygon(
                locations=cell_geometries[coverage_cell],
                color=severity_color,
                weight=2,
                fill=True,
                fillColor='#17a2b8',
                fillOpacity=0.5,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"CA Imbalance: {coverage_cell} ({severity})",
            ).add_to(layer)
        else:
            folium.CircleMarker(
                location=source_coords,
                radius=7,
                color=severity_color,
                fill=True,
                fillColor='#17a2b8',
                fillOpacity=0.7,
                weight=2,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"CA Imbalance: {coverage_cell} ({severity})",
            ).add_to(layer)

    layer.add_to(m)
    return layer, hull_geometries


def _add_crossed_feeder_layer(
    m: folium.Map,
    df: pd.DataFrame,
    cell_coords: Optional[dict] = None,
    cell_geometries: Optional[dict] = None,
) -> Tuple[folium.FeatureGroup, dict]:
    """
    Add crossed feeder layer with lines to suspicious relations.

    Lines are colored based on whether the relation is in-beam or out-of-beam:
    - Green (in-beam): angle is within min(HBW * 1.5, 60) of the cell's bearing
    - Red (out-of-beam): angle is outside min(HBW * 1.5, 60) of the cell's bearing

    Args:
        m: Folium map object
        df: DataFrame with crossed feeder data
        cell_coords: Dict mapping cell_name to [lat, lon]
        cell_geometries: Dict mapping cell_name to polygon coordinates [[lat, lon], ...]

    Returns:
        Tuple of (FeatureGroup, line_data dict)
    """
    import re

    layer = folium.FeatureGroup(name='Crossed Feeders', show=False)
    line_data = {}

    if df is None or len(df) == 0:
        layer.add_to(m)
        return layer, line_data

    cell_geometries = cell_geometries or {}

    def is_in_beam(bearing: float, hbw: float, angle: float) -> bool:
        """Check if angle is within the beam width of the cell."""
        # Calculate effective beam tolerance: min(HBW * 1.5, 60)
        beam_tolerance = min(hbw * 1.5, 60.0)

        # Normalize angles to 0-360
        bearing = bearing % 360
        angle = angle % 360

        # Calculate angular difference (accounting for wrap-around)
        diff = abs(bearing - angle)
        if diff > 180:
            diff = 360 - diff

        return diff <= beam_tolerance

    # Only show flagged cells
    flagged_df = df[df.get('flagged', False) == True] if 'flagged' in df.columns else df

    for idx, row in flagged_df.iterrows():
        cell_name = row.get('cell_name', '')
        site = row.get('site', '')
        band = row.get('band', '')
        bearing = float(row.get('bearing', 0) or 0)
        hbw = float(row.get('hbw', 60) or 60)
        cell_score = row.get('cell_score', 0)
        threshold = row.get('threshold', 0)
        suspicious_relations = row.get('top_suspicious_relations', '')
        severity_score = row.get('severity_score', 0)
        severity_category = row.get('severity_category', 'MEDIUM')
        severity_color = SEVERITY_COLORS.get(severity_category, '#6c757d')

        # Get cell coordinates
        if cell_name not in cell_coords:
            continue
        source_coords = cell_coords[cell_name]

        # Parse suspicious relations: "CK640L1 (d=9413m, w=3.4, angle=228°) | ..."
        feature_key = f"crossed_feeder_{idx}"
        targets = []
        in_beam_count = 0
        out_of_beam_count = 0

        if suspicious_relations:
            # Try new format with angle first
            pattern_with_angle = r'(\w+)\s*\(d=(\d+)m,\s*w=([\d.]+),\s*angle=(\d+)°?\)'
            matches = re.findall(pattern_with_angle, suspicious_relations)

            if matches:
                for neighbor, distance, weight, angle_str in matches:
                    if neighbor in cell_coords:
                        weight_val = float(weight)
                        angle_val = float(angle_str)

                        # Determine if in-beam or out-of-beam
                        in_beam = is_in_beam(bearing, hbw, angle_val)
                        if in_beam:
                            color = LINE_COLORS['crossed_normal']  # Green for in-beam
                            in_beam_count += 1
                        else:
                            color = LINE_COLORS['crossed_offender']  # Red for out-of-beam
                            out_of_beam_count += 1

                        # Get neighbor geometry if available
                        neighbor_geom = cell_geometries.get(neighbor)

                        # Calculate centroid if geometry available, otherwise use coords
                        if neighbor_geom and len(neighbor_geom) > 0:
                            centroid_lat = sum(p[0] for p in neighbor_geom) / len(neighbor_geom)
                            centroid_lon = sum(p[1] for p in neighbor_geom) / len(neighbor_geom)
                            line_endpoint = [centroid_lat, centroid_lon]
                        else:
                            line_endpoint = cell_coords[neighbor]

                        beam_status = "in-beam" if in_beam else "out-of-beam"
                        targets.append({
                            'coords': line_endpoint,
                            'name': f"{neighbor} ({beam_status}, angle={angle_val:.0f}°)",
                            'color': color,
                            'geometry': neighbor_geom
                        })
            else:
                # Fallback to old format without angle
                pattern_legacy = r'(\w+)\s*\(d=(\d+)m,\s*score=([\d.]+)\)'
                matches = re.findall(pattern_legacy, suspicious_relations)

                for neighbor, distance, score in matches:
                    if neighbor in cell_coords:
                        score_val = float(score)
                        color = LINE_COLORS['crossed_offender']
                        out_of_beam_count += 1

                        neighbor_geom = cell_geometries.get(neighbor)

                        if neighbor_geom and len(neighbor_geom) > 0:
                            centroid_lat = sum(p[0] for p in neighbor_geom) / len(neighbor_geom)
                            centroid_lon = sum(p[1] for p in neighbor_geom) / len(neighbor_geom)
                            line_endpoint = [centroid_lat, centroid_lon]
                        else:
                            line_endpoint = cell_coords[neighbor]

                        targets.append({
                            'coords': line_endpoint,
                            'name': f"{neighbor} (score={score_val:.1f})",
                            'color': color,
                            'geometry': neighbor_geom
                        })

        has_line_data = len(targets) > 0
        if has_line_data:
            line_data[feature_key] = {
                'source': source_coords,
                'targets': targets,
                'color': LINE_COLORS['crossed_offender'],
                'style': 'dotted'
            }

        # Build relations display
        relations_display = suspicious_relations[:100] + '...' if len(suspicious_relations) > 100 else suspicious_relations

        line_button_html = ""
        if has_line_data:
            line_button_html = f"""
            <div style="text-align: center; margin-top: 10px;">
                <button id="lineBtn_{feature_key}"
                        data-show-text="Show Relations"
                        data-hide-text="Hide Relations"
                        onclick="toggleLines('{feature_key}', 'lineBtn_{feature_key}')"
                        style="background: {LINE_COLORS['crossed_offender']}; color: white; border: none;
                               padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    Show Relations
                </button>
                <div id="lineStatus_{feature_key}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
            </div>
            <div style="margin-top: 8px; font-size: 10px; color: #666;">
                <span style="color: {LINE_COLORS['crossed_normal']};">----</span> In-beam ({in_beam_count}) &nbsp;
                <span style="color: {LINE_COLORS['crossed_offender']};">----</span> Out-of-beam ({out_of_beam_count})
            </div>
            """

        popup_html = f"""
        <div style="font-family: Arial; width: 300px;" data-band="{band}" data-severity="{severity_category}">
            <h4 style="margin: 0 0 8px 0; color: {LINE_COLORS['crossed_offender']};">
                Crossed Feeder Suspect
                <span style="float: right; font-size: 12px; background: {severity_color};
                             color: white; padding: 2px 8px; border-radius: 4px;">{severity_category}</span>
            </h4>
            <div style="background: #f8d7da; padding: 8px; border-radius: 4px;">
                <strong>Cell:</strong> {cell_name}<br>
                <strong>Site:</strong> {site}<br>
                <strong>Band:</strong> {band}<br>
                <strong>Severity Score:</strong> {severity_score:.3f}<br>
                <strong>Bearing:</strong> {bearing:.0f}°<br>
                <strong>HBW:</strong> {hbw:.0f}°<br>
                <strong>Anomaly Score:</strong> {cell_score:.2f}<br>
                <strong>Threshold:</strong> {threshold:.2f}<br>
                <strong>Suspicious Relations:</strong><br>
                <span style="font-size: 10px;">{relations_display}</span>
            </div>
            <div style="background: #f5c6cb; padding: 6px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <strong>Issue:</strong> Cell has unusual relation pattern suggesting possible feeder swap.
            </div>
            {line_button_html}
        </div>
        """

        # Add cell sector polygon if geometry available, otherwise use marker
        if cell_name in cell_geometries:
            # Draw the cell sector polygon
            folium.Polygon(
                locations=cell_geometries[cell_name],
                color=LINE_COLORS['crossed_offender'],
                weight=2,
                fill=True,
                fillColor='#f5c6cb',
                fillOpacity=0.5,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"Crossed Feeder: {cell_name} (score={cell_score:.1f})",
            ).add_to(layer)
        else:
            # Fallback to circle marker if no geometry
            folium.CircleMarker(
                location=source_coords,
                radius=8,
                color=LINE_COLORS['crossed_offender'],
                fill=True,
                fillColor='#f5c6cb',
                fillOpacity=0.8,
                weight=2,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"Crossed Feeder: {cell_name} (score={cell_score:.1f})",
            ).add_to(layer)

    layer.add_to(m)
    return layer, line_data


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
        safe_cell_name = _sanitize_js_name(cell_name)
        html += f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="gridBtn_overshooting_{safe_cell_name}" onclick="toggleOvershootingGrids_{safe_cell_name}()"
                    style="background: #dc3545; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Load & Show Grids
            </button>
            <div id="gridStatus_overshooting_{safe_cell_name}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
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
        safe_cell_name = _sanitize_js_name(cell_name)
        html += f"""
        <div style="text-align: center; margin-top: 10px;">
            <button id="gridBtn_undershooting_{safe_cell_name}" onclick="toggleUndershootingGrids_{safe_cell_name}()"
                    style="background: #0d6efd; color: white; border: none;
                           padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                Load & Show Grids
            </button>
            <div id="gridStatus_undershooting_{safe_cell_name}" style="margin-top: 5px; font-size: 11px; color: #666;"></div>
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
    no_cov_pb_total = stats['no_coverage_per_band']['total']
    low_cov_total = stats['low_coverage']['total']
    pci_total = stats['pci_confusions']['total'] + stats['pci_collisions']['total'] + stats['pci_blacklist']['total']
    ca_total = stats['ca_imbalance']['total']
    xf_total = stats['crossed_feeders']['total']
    interf_total = stats['interference']['total']
    total_issues = over_total + under_total + no_cov_total + no_cov_pb_total + low_cov_total + pci_total + ca_total + xf_total + interf_total

    html = f"""
    <div id="summaryPanel" style="
        position: fixed; top: 80px; left: 10px; width: 240px;
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9999; padding: 12px; font-family: Arial;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-height: calc(100vh - 150px); overflow-y: auto;
    ">
        <h4 style="margin: 0 0 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 8px;">
            Summary Statistics
        </h4>

        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span id="summaryTotalIssues" style="font-size: 28px; font-weight: bold; color: #333;">{total_issues}</span>
            <span style="font-size: 12px; color: #6c757d; align-self: flex-end;">Total Issues</span>
        </div>

        <div style="margin-top: 12px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['overshooting']}; border-radius: 2px; margin-right: 5px;"></span>
                    Overshooting
                </span>
                <strong id="summaryOvershooting">{over_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['undershooting']}; border-radius: 2px; margin-right: 5px;"></span>
                    Undershooting
                </span>
                <strong id="summaryUndershooting">{under_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['no_coverage']}; border-radius: 2px; margin-right: 5px;"></span>
                    No Coverage
                </span>
                <strong>{no_cov_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: #17a2b8; border-radius: 2px; margin-right: 5px;"></span>
                    No Cov (Per Band)
                </span>
                <strong>{no_cov_pb_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['low_coverage']}; border-radius: 2px; margin-right: 5px;"></span>
                    Low Coverage
                </span>
                <strong>{low_cov_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['interference']}; border-radius: 2px; margin-right: 5px;"></span>
                    Interference
                </span>
                <strong>{interf_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['pci_confusion']}; border-radius: 2px; margin-right: 5px;"></span>
                    PCI Issues
                </span>
                <strong>{pci_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['ca_imbalance']}; border-radius: 2px; margin-right: 5px;"></span>
                    CA Imbalance
                </span>
                <strong>{ca_total}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="display: flex; align-items: center;">
                    <span style="width: 10px; height: 10px; background: {ISSUE_COLORS['crossed_feeder']}; border-radius: 2px; margin-right: 5px;"></span>
                    Crossed Feeders
                </span>
                <strong>{xf_total}</strong>
            </div>
        </div>

        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">

        <div style="font-size: 10px; color: #6c757d;">
            <div>Avg downtilt: +{stats['overshooting']['avg_tilt_change']:.1f}°</div>
            <div>Avg uptilt: -{stats['undershooting']['avg_uptilt']:.1f}°</div>
            <div>No cov area: {stats['no_coverage']['total_area_km2']:.1f} km²</div>
            <div>No cov/band: {stats['no_coverage_per_band']['total_area_km2']:.1f} km²</div>
            <div>Low cov area: {stats['low_coverage']['total_area_km2']:.1f} km²</div>
        </div>

        <div style="margin-top: 8px; font-size: 9px; color: #aaa;">
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
        z-index: 9999; padding: 12px; font-family: Arial; font-size: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-height: 350px; overflow-y: auto;
    ">
        <h4 style="margin: 0 0 8px 0; font-size: 12px;">Legend</h4>

        <p style="margin: 4px 0; font-weight: bold;">Severity (Sectors)</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['CRITICAL']};">&#9650;</span> CRITICAL</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['HIGH']};">&#9650;</span> HIGH</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['MEDIUM']};">&#9650;</span> MEDIUM</p>
        <p style="margin: 2px 0;"><span style="color: {SEVERITY_COLORS['LOW']};">&#9650;</span> LOW</p>

        <p style="margin: 8px 0 4px 0; font-weight: bold;">Environment</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['URBAN']};">&#9679;</span> URBAN</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['SUBURBAN']};">&#9679;</span> SUBURBAN</p>
        <p style="margin: 2px 0;"><span style="color: {ENVIRONMENT_COLORS['RURAL']};">&#9679;</span> RURAL</p>

        <p style="margin: 8px 0 4px 0; font-weight: bold;">Coverage Areas</p>
        <p style="margin: 2px 0;"><span style="color: {ISSUE_COLORS['no_coverage']};">&#9632;</span> No Coverage</p>
        <p style="margin: 2px 0;"><span style="color: {ISSUE_COLORS['low_coverage']};">&#9632;</span> Low Coverage</p>
        <p style="margin: 2px 0;"><span style="color: {ISSUE_COLORS['interference']};">&#9632;</span> Interference</p>

        <p style="margin: 8px 0 4px 0; font-weight: bold;">Relationship Lines</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['pci']};">- - -</span> PCI Relations</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['blacklist']};">- - -</span> Blacklist</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['crossed_offender']};">&#8212;</span> Crossed (suspicious)</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['crossed_normal']};">&#8212;</span> Crossed (normal)</p>

        <p style="margin: 8px 0 4px 0; font-weight: bold;">CA Imbalance Hulls</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['ca_coverage']};">&#9632;</span> Coverage Band</p>
        <p style="margin: 2px 0;"><span style="color: {LINE_COLORS['ca_capacity']};">&#9632;</span> Capacity Band</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_filter_panel(m: folium.Map, stats: dict, all_bands: Optional[list] = None):
    """Add filter panel."""
    # Get all severities from all detector types
    severity_keys = [
        'overshooting', 'undershooting', 'pci_confusions', 'pci_collisions',
        'crossed_feeders', 'interference', 'ca_imbalance'
    ]
    severities = set()
    for key in severity_keys:
        severities.update(stats.get(key, {}).get('by_severity', {}).keys())
    severities = sorted(severities, key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'].index(x) if x in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'] else 99)

    # Get all environments from all detector types
    environments = set()
    for key in ['overshooting', 'undershooting']:
        environments.update(stats.get(key, {}).get('by_environment', {}).keys())

    # Get all bands - use provided list or extract from stats
    if all_bands:
        bands = sorted(set(all_bands))
    else:
        bands = set()
        for key in severity_keys:
            bands.update(stats.get(key, {}).get('by_band', {}).keys())
        bands.update(stats.get('low_coverage', {}).get('by_band', {}).keys())
        bands.update(stats.get('no_coverage_per_band', {}).get('by_band', {}).keys())
        bands = sorted(bands)

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
            <label style="font-weight: bold; display: block; margin-bottom: 4px;">Severity</label>
            <div id="severityFilter" style="font-size: 11px;">
    """

    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
        if sev in severities:
            color = {'CRITICAL': '#dc3545', 'HIGH': '#fd7e14', 'MEDIUM': '#ffc107', 'LOW': '#28a745', 'MINIMAL': '#6c757d'}.get(sev, '#666')
            html += f'''
                <label style="display: block; margin: 2px 0; cursor: pointer;">
                    <input type="checkbox" class="severity-checkbox" value="{sev}" checked onchange="applyFilters()" style="margin-right: 4px;">
                    <span style="color: {color}; font-weight: 500;">{sev}</span>
                </label>'''

    html += """
            </div>
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
        # Format band display - handle both "L800" and numeric formats
        if isinstance(band, str) and band.startswith('L'):
            display_band = band
        else:
            display_band = f"{band} MHz" if band else str(band)
        html += f'<option value="{band}">{display_band}</option>'

    html += """
            </select>
        </div>

        <button onclick="resetFilters()" style="
            width: 100%; padding: 8px; background: #6c757d; color: white;
            border: none; border-radius: 4px; cursor: pointer;
        ">Reset Filters</button>

        <div id="filterStatus" style="margin-top: 8px; font-size: 10px; color: #666; text-align: center;"></div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_filter_javascript(m: folium.Map):
    """Add JavaScript for filtering markers and recommendations table based on filter values."""
    map_var = m.get_name()

    js = f"""
    <script>
    function applyFilters() {{
        // Get selected severities from checkboxes
        var selectedSeverities = [];
        document.querySelectorAll('.severity-checkbox:checked').forEach(function(cb) {{
            selectedSeverities.push(cb.value.toUpperCase());
        }});
        var env = document.getElementById('envFilter').value.toUpperCase();
        var band = document.getElementById('bandFilter').value;
        var statusDiv = document.getElementById('filterStatus');

        var hidden = 0;
        var total = 0;

        // Helper function to filter a single layer
        function filterLayer(layer) {{
            if (!layer.getPopup || !layer.getPopup()) return;
            var popup = layer.getPopup();
            var content = popup.getContent();

            // Content can be string, DOM element, or function
            var contentStr = '';
            if (typeof content === 'string') {{
                contentStr = content;
            }} else if (content && content.innerHTML) {{
                contentStr = content.innerHTML;
            }} else if (content && content.outerHTML) {{
                contentStr = content.outerHTML;
            }}
            if (!contentStr) return;

            total++;
            var show = true;

            // Severity filter (multi-select checkboxes)
            if (selectedSeverities.length > 0 && selectedSeverities.length < 5) {{
                var sevMatch = contentStr.match(/>(CRITICAL|HIGH|MEDIUM|LOW|MINIMAL)</i);
                if (sevMatch && selectedSeverities.indexOf(sevMatch[1].toUpperCase()) === -1) {{
                    show = false;
                }}
            }}

            // Environment filter - handles "Environment:</strong>" and "Environment</div>"
            if (env !== 'ALL' && show) {{
                var envMatch = contentStr.match(/Environment[:<].*?(URBAN|SUBURBAN|RURAL)/i);
                if (envMatch && envMatch[1].toUpperCase() !== env) {{
                    show = false;
                }}
            }}

            // Band filter - first try data-band attribute, then fall back to Band text
            // This ensures all layers with data-band can be filtered by source cell band
            if (band !== 'all' && show) {{
                // Try data-band attribute first (most reliable)
                var dataBandMatch = contentStr.match(/data-band="([^"]+)"/i);
                if (dataBandMatch) {{
                    if (dataBandMatch[1] !== band) {{
                        show = false;
                    }}
                }} else {{
                    // Fall back to Band text match
                    var bandMatch = contentStr.match(/Band[:<].*?(L\\d+)/i);
                    if (bandMatch) {{
                        if (bandMatch[1] !== band) {{
                            show = false;
                        }}
                    }}
                }}
            }}

            // Hide/show the layer element
            if (layer._path) {{
                layer._path.style.display = show ? '' : 'none';
            }}
            if (layer._icon) {{
                layer._icon.style.display = show ? '' : 'none';
            }}
            // For CircleMarkers and other SVG elements
            if (layer.getElement) {{
                var el = layer.getElement();
                if (el) el.style.display = show ? '' : 'none';
            }}
            if (!show) hidden++;
        }}

        // Filter all layers recursively
        // For GeoJson layers, the popup is bound to the parent layer, not child features
        // So we need to filter the parent AND propagate visibility to children
        var layerCount = 0;
        var featureGroupCount = 0;

        function processLayers(parentLayer, parentVisible) {{
            if (parentLayer.eachLayer) {{
                featureGroupCount++;
                // First, check if this layer itself has a popup (GeoJson case)
                var layerPopupContent = null;
                var layerShouldShow = parentVisible !== false;

                if (parentLayer.getPopup && parentLayer.getPopup()) {{
                    var popup = parentLayer.getPopup();
                    var content = popup.getContent();
                    if (typeof content === 'string') {{
                        layerPopupContent = content;
                    }} else if (content && content.innerHTML) {{
                        layerPopupContent = content.innerHTML;
                    }} else if (content && content.outerHTML) {{
                        layerPopupContent = content.outerHTML;
                    }}

                    if (layerPopupContent) {{
                        total++;
                        layerShouldShow = true;

                        // Severity filter
                        if (selectedSeverities.length > 0 && selectedSeverities.length < 5) {{
                            var sevMatch = layerPopupContent.match(/>(CRITICAL|HIGH|MEDIUM|LOW|MINIMAL)</i);
                            if (sevMatch && selectedSeverities.indexOf(sevMatch[1].toUpperCase()) === -1) {{
                                layerShouldShow = false;
                            }}
                        }}

                        // Environment filter
                        if (env !== 'ALL' && layerShouldShow) {{
                            var envMatch = layerPopupContent.match(/Environment[:<].*?(URBAN|SUBURBAN|RURAL)/i);
                            if (envMatch && envMatch[1].toUpperCase() !== env) {{
                                layerShouldShow = false;
                            }}
                        }}

                        // Band filter
                        if (band !== 'all' && layerShouldShow) {{
                            var dataBandMatch = layerPopupContent.match(/data-band="([^"]+)"/i);
                            if (dataBandMatch) {{
                                if (dataBandMatch[1] !== band) {{
                                    layerShouldShow = false;
                                }}
                            }} else {{
                                var bandMatch = layerPopupContent.match(/Band[:<].*?(L\\d+)/i);
                                if (bandMatch && bandMatch[1] !== band) {{
                                    layerShouldShow = false;
                                }}
                            }}
                        }}

                        if (!layerShouldShow) hidden++;
                    }}
                }}

                // Process child layers
                parentLayer.eachLayer(function(layer) {{
                    layerCount++;

                    // If parent GeoJson was filtered, hide all children
                    if (layerPopupContent && !layerShouldShow) {{
                        if (layer._path) layer._path.style.display = 'none';
                        if (layer._icon) layer._icon.style.display = 'none';
                        if (layer.getElement) {{
                            var el = layer.getElement();
                            if (el) el.style.display = 'none';
                        }}
                    }} else if (layerPopupContent && layerShouldShow) {{
                        // Parent visible, show children
                        if (layer._path) layer._path.style.display = '';
                        if (layer._icon) layer._icon.style.display = '';
                        if (layer.getElement) {{
                            var el = layer.getElement();
                            if (el) el.style.display = '';
                        }}
                    }} else {{
                        // No parent popup, filter this layer directly
                        filterLayer(layer);
                    }}

                    if (layer.eachLayer) processLayers(layer, layerShouldShow);
                }});
            }}
        }}

        processLayers({map_var}, true);
        console.log('Map filter: found ' + featureGroupCount + ' groups, ' + layerCount + ' layers, ' + total + ' with popups, ' + hidden + ' hidden');

        // Filter recommendations table rows
        var tableHidden = 0;
        var tableTotal = 0;
        console.log('Filter values: band=' + band + ', severities=' + selectedSeverities.join(',') + ', env=' + env);
        document.querySelectorAll('.rec-row').forEach(function(row) {{
            tableTotal++;
            var show = true;
            var rowSeverity = row.getAttribute('data-severity') || '';
            var rowBand = row.getAttribute('data-band') || '';
            var rowEnv = row.getAttribute('data-environment') || '';

            // Severity filter (multi-select checkboxes)
            if (selectedSeverities.length > 0 && selectedSeverities.length < 5 && rowSeverity) {{
                if (selectedSeverities.indexOf(rowSeverity.toUpperCase()) === -1) {{
                    show = false;
                }}
            }}

            // Environment filter
            if (env !== 'ALL' && rowEnv && rowEnv.toUpperCase() !== env) {{
                show = false;
            }}

            // Band filter - compare values directly (both should be like "L800")
            if (band !== 'all' && rowBand) {{
                if (rowBand !== band) {{
                    show = false;
                }}
            }}

            row.style.display = show ? '' : 'none';
            if (!show) tableHidden++;
        }});
        console.log('Table filter: ' + tableHidden + ' hidden of ' + tableTotal + ' total');

        // Update filter status
        if (statusDiv) {{
            var statusParts = [];
            if (hidden > 0) statusParts.push(hidden + ' map markers hidden');
            if (tableHidden > 0) statusParts.push(tableHidden + ' table rows hidden');
            statusDiv.textContent = statusParts.join(', ') || '';
        }}

        // Update tab counts to show filtered counts
        updateTabCounts();
    }}

    function updateTabCounts() {{
        var tabs = ['overshooting', 'undershooting', 'pci', 'ca_imbalance', 'crossed_feeders'];
        var totalVisible = 0;
        var counts = {{}};

        tabs.forEach(function(tabId) {{
            var tabContent = document.getElementById('recTab_' + tabId);
            if (tabContent) {{
                var visibleRows = tabContent.querySelectorAll('.rec-row:not([style*="display: none"])').length;
                var totalRows = tabContent.querySelectorAll('.rec-row').length;
                var tabBtn = document.querySelector('[data-tab="' + tabId + '"] span');
                if (tabBtn && totalRows > 0) {{
                    tabBtn.textContent = visibleRows;
                }}
                counts[tabId] = visibleRows;
                totalVisible += visibleRows;
            }}
        }});

        // Update summary panel counts
        var totalEl = document.getElementById('summaryTotalIssues');
        if (totalEl) totalEl.textContent = totalVisible;

        var overEl = document.getElementById('summaryOvershooting');
        if (overEl) overEl.textContent = counts['overshooting'] || 0;

        var underEl = document.getElementById('summaryUndershooting');
        if (underEl) underEl.textContent = counts['undershooting'] || 0;
    }}

    function resetFilters() {{
        // Check all severity checkboxes
        document.querySelectorAll('.severity-checkbox').forEach(function(cb) {{
            cb.checked = true;
        }});
        document.getElementById('envFilter').value = 'all';
        document.getElementById('bandFilter').value = 'all';
        applyFilters();
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


def _add_recommendations_table_panel(
    m: folium.Map,
    cell_coords: dict,
    overshooting_df: Optional[pd.DataFrame] = None,
    undershooting_df: Optional[pd.DataFrame] = None,
    pci_confusions_df: Optional[pd.DataFrame] = None,
    pci_collisions_df: Optional[pd.DataFrame] = None,
    pci_blacklist_df: Optional[pd.DataFrame] = None,
    ca_imbalance_df: Optional[pd.DataFrame] = None,
    crossed_feeder_df: Optional[pd.DataFrame] = None,
):
    """Add collapsible recommendations table panel with tabs for each detector type."""

    # Build recommendations data for each detector
    recommendations = {
        'overshooting': [],
        'undershooting': [],
        'pci': [],
        'ca_imbalance': [],
        'crossed_feeders': []
    }

    # Helper to extract band from cell name based on VF-IE naming convention
    # e.g., "CK052L2" -> "L800", "CK329H3" -> "L1800", "CK144T1" -> "L2100", "CK052K1" -> "L700"
    def extract_band_from_cell(cell_name):
        if not cell_name:
            return ''
        import re
        # Pattern: letter + digit at end (L1, L2, L3, H1, H2, H3, T1, T2, T3, K1, K2, K3)
        match = re.search(r'([LHTK])(\d)$', str(cell_name))
        if match:
            band_letter = match.group(1)
            band_map = {'L': 'L800', 'H': 'L1800', 'T': 'L2100', 'K': 'L700'}
            return band_map.get(band_letter, '')
        return ''

    # Overshooting recommendations
    if overshooting_df is not None and len(overshooting_df) > 0:
        for _, row in overshooting_df.iterrows():
            cell_name = row.get('cell_name', '')
            severity = row.get('severity_category', 'MEDIUM')
            tilt_change = row.get('recommended_tilt_change', 0)
            environment = row.get('environment', '')
            band = row.get('band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords.get(cell_name, [0, 0])
            recommendations['overshooting'].append({
                'cell': cell_name,
                'severity': severity,
                'recommendation': f"Increase tilt by {tilt_change}°",
                'coords': coords,
                'band': band,
                'environment': environment
            })

    # Undershooting recommendations
    if undershooting_df is not None and len(undershooting_df) > 0:
        for _, row in undershooting_df.iterrows():
            cell_name = row.get('cell_name', '')
            severity = row.get('severity_category', 'MEDIUM')
            uptilt = row.get('recommended_uptilt', 0)
            environment = row.get('environment', '')
            band = row.get('band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords.get(cell_name, [0, 0])
            recommendations['undershooting'].append({
                'cell': cell_name,
                'severity': severity,
                'recommendation': f"Decrease tilt by {uptilt}°",
                'coords': coords,
                'band': band,
                'environment': environment
            })

    # PCI recommendations (combine confusions, collisions, blacklist) - only include items with coordinates
    if pci_confusions_df is not None and len(pci_confusions_df) > 0:
        for _, row in pci_confusions_df.iterrows():
            cell_name = row.get('serving', '')
            if cell_name not in cell_coords:
                continue  # Skip items without coordinates
            band = row.get('band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords[cell_name]
            recommendations['pci'].append({
                'cell': cell_name,
                'severity': 'HIGH',
                'recommendation': f"PCI confusion with {row.get('neighbors', '')}",
                'coords': coords,
                'band': band,
                'environment': ''
            })

    if pci_collisions_df is not None and len(pci_collisions_df) > 0:
        for _, row in pci_collisions_df.iterrows():
            cell_a = row.get('cell_a', '')
            cell_b = row.get('cell_b', '')
            if cell_a not in cell_coords or cell_b not in cell_coords:
                continue  # Skip items without coordinates for both cells
            band = row.get('band', '') or extract_band_from_cell(cell_a)
            coords = cell_coords[cell_a]
            recommendations['pci'].append({
                'cell': f"{cell_a} / {cell_b}",
                'severity': 'MEDIUM',
                'recommendation': f"PCI collision (PCI={row.get('pci', '')})",
                'coords': coords,
                'band': band,
                'environment': ''
            })

    if pci_blacklist_df is not None and len(pci_blacklist_df) > 0:
        for _, row in pci_blacklist_df.iterrows():
            cell_name = row.get('serving', '')
            if cell_name not in cell_coords:
                continue  # Skip items without coordinates
            band = row.get('band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords[cell_name]
            recommendations['pci'].append({
                'cell': cell_name,
                'severity': 'LOW',
                'recommendation': f"Blacklist {row.get('neighbor', '')}",
                'coords': coords,
                'band': band,
                'environment': ''
            })

    # CA Imbalance recommendations - only include items with coordinates
    if ca_imbalance_df is not None and len(ca_imbalance_df) > 0:
        for _, row in ca_imbalance_df.iterrows():
            cell_name = row.get('coverage_cell_name', '')
            if cell_name not in cell_coords:
                continue  # Skip items without coordinates
            severity = row.get('severity', 'medium').upper()
            band = row.get('coverage_band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords[cell_name]
            recommendations['ca_imbalance'].append({
                'cell': cell_name,
                'severity': severity,
                'recommendation': row.get('recommendation', 'Review CA configuration'),
                'coords': coords,
                'band': band,
                'environment': ''
            })

    # Crossed Feeder recommendations - only include items with coordinates
    if crossed_feeder_df is not None and len(crossed_feeder_df) > 0:
        flagged_df = crossed_feeder_df[crossed_feeder_df.get('flagged', False) == True] if 'flagged' in crossed_feeder_df.columns else crossed_feeder_df
        for _, row in flagged_df.iterrows():
            cell_name = row.get('cell_name', '')
            if cell_name not in cell_coords:
                continue  # Skip items without coordinates
            band = row.get('band', '') or extract_band_from_cell(cell_name)
            coords = cell_coords[cell_name]
            recommendations['crossed_feeders'].append({
                'cell': cell_name,
                'severity': 'HIGH',
                'recommendation': 'Verify antenna connections',
                'coords': coords,
                'band': band,
                'environment': ''
            })

    # Build tab buttons and content
    tab_buttons = ""
    tab_contents = ""
    tabs = [
        ('overshooting', 'Overshooting', ISSUE_COLORS['overshooting']),
        ('undershooting', 'Undershooting', ISSUE_COLORS['undershooting']),
        ('pci', 'PCI Issues', ISSUE_COLORS['pci_confusion']),
        ('ca_imbalance', 'CA Imbalance', ISSUE_COLORS['ca_imbalance']),
        ('crossed_feeders', 'Crossed Feeders', ISSUE_COLORS['crossed_feeder'])
    ]

    for idx, (tab_id, tab_name, color) in enumerate(tabs):
        count = len(recommendations[tab_id])
        is_active = idx == 0
        tab_buttons += f"""
        <button class="rec-tab-btn {'rec-tab-active' if is_active else ''}"
                onclick="showRecTab('{tab_id}')"
                data-tab="{tab_id}"
                style="padding: 6px 10px; margin-right: 4px; border: none; border-radius: 4px 4px 0 0;
                       background: {'white' if is_active else '#e9ecef'}; cursor: pointer;
                       border-bottom: 2px solid {color if is_active else 'transparent'};
                       font-size: 11px;">
            {tab_name} <span style="background: {color}; color: white; padding: 1px 5px;
                               border-radius: 10px; font-size: 10px; margin-left: 3px;">{count}</span>
        </button>
        """

        rows_html = ""
        for rec in recommendations[tab_id]:  # Show all recommendations
            severity_color = {
                'CRITICAL': '#dc3545', 'HIGH': '#fd7e14', 'MEDIUM': '#ffc107', 'LOW': '#17a2b8', 'MINIMAL': '#28a745'
            }.get(rec['severity'], '#6c757d')
            rec_band = rec.get('band', '')
            rec_env = rec.get('environment', '')
            rows_html += f"""
            <tr onclick="zoomToCell({rec['coords'][0]}, {rec['coords'][1]})"
                style="cursor: pointer;" class="rec-row"
                data-severity="{rec['severity']}" data-band="{rec_band}" data-environment="{rec_env}">
                <td style="padding: 4px 6px; border-bottom: 1px solid #eee;">{rec['cell'][:20]}</td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #eee;">
                    <span style="background: {severity_color}; color: white; padding: 1px 5px;
                                 border-radius: 3px; font-size: 10px;">{rec['severity']}</span>
                </td>
                <td style="padding: 4px 6px; border-bottom: 1px solid #eee; font-size: 10px;">{rec['recommendation'][:40]}</td>
            </tr>
            """

        if not rows_html:
            rows_html = '<tr><td colspan="3" style="padding: 20px; text-align: center; color: #999;">No issues found</td></tr>'

        tab_contents += f"""
        <div id="recTab_{tab_id}" class="rec-tab-content" style="display: {'block' if is_active else 'none'};">
            <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 6px; text-align: left; border-bottom: 2px solid #dee2e6;">Cell</th>
                        <th style="padding: 6px; text-align: left; border-bottom: 2px solid #dee2e6;">Severity</th>
                        <th style="padding: 6px; text-align: left; border-bottom: 2px solid #dee2e6;">Recommendation</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    # Total recommendations count
    total_recs = sum(len(r) for r in recommendations.values())

    html = f"""
    <div id="recTablePanel" style="
        position: fixed; bottom: 30px; left: 220px; width: 450px;
        background-color: white; border: 2px solid #333; border-radius: 8px;
        z-index: 9998; font-family: Arial;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-height: 60px; overflow: hidden;
        transition: max-height 0.3s ease-out;
    ">
        <div id="recTableHeader" onclick="toggleRecTable()" style="
            padding: 10px 12px; cursor: pointer; background: #343a40; color: white;
            border-radius: 6px 6px 0 0; display: flex; justify-content: space-between;
            align-items: center;
        ">
            <span style="font-weight: bold;">
                <span id="recTableIcon">▶</span> Recommendations Table
            </span>
            <span style="background: #0d6efd; padding: 2px 8px; border-radius: 10px; font-size: 12px;">
                {total_recs} issues
            </span>
        </div>

        <div id="recTableBody" style="display: none; padding: 0;">
            <div style="padding: 8px 12px 4px 12px; border-bottom: 1px solid #dee2e6;">
                {tab_buttons}
            </div>
            <div style="max-height: 250px; overflow-y: auto; padding: 0 12px 12px 12px;">
                {tab_contents}
            </div>
        </div>
    </div>

    <style>
    .rec-row:hover {{ background: #f0f7ff !important; }}
    .rec-tab-btn:hover {{ background: #dee2e6 !important; }}
    </style>

    <script>
    var recTableExpanded = false;

    function toggleRecTable() {{
        var panel = document.getElementById('recTablePanel');
        var body = document.getElementById('recTableBody');
        var icon = document.getElementById('recTableIcon');

        recTableExpanded = !recTableExpanded;

        if (recTableExpanded) {{
            panel.style.maxHeight = '400px';
            body.style.display = 'block';
            icon.textContent = '▼';
        }} else {{
            panel.style.maxHeight = '60px';
            body.style.display = 'none';
            icon.textContent = '▶';
        }}
    }}

    function showRecTab(tabId) {{
        // Hide all tab contents
        document.querySelectorAll('.rec-tab-content').forEach(function(el) {{
            el.style.display = 'none';
        }});

        // Show selected tab
        document.getElementById('recTab_' + tabId).style.display = 'block';

        // Update button styles
        document.querySelectorAll('.rec-tab-btn').forEach(function(btn) {{
            btn.classList.remove('rec-tab-active');
            btn.style.background = '#e9ecef';
            btn.style.borderBottom = '2px solid transparent';
        }});

        var activeBtn = document.querySelector('[data-tab="' + tabId + '"]');
        if (activeBtn) {{
            activeBtn.classList.add('rec-tab-active');
            activeBtn.style.background = 'white';
            // Get color from the span inside
            var colors = {{
                'overshooting': '{ISSUE_COLORS["overshooting"]}',
                'undershooting': '{ISSUE_COLORS["undershooting"]}',
                'pci': '{ISSUE_COLORS["pci_confusion"]}',
                'ca_imbalance': '{ISSUE_COLORS["ca_imbalance"]}',
                'crossed_feeders': '{ISSUE_COLORS["crossed_feeder"]}'
            }};
            activeBtn.style.borderBottom = '2px solid ' + colors[tabId];
        }}
    }}

    function zoomToCell(lat, lon) {{
        if (lat && lon && lat !== 0 && lon !== 0) {{
            {m.get_name()}.setView([lat, lon], 15);
        }}
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(html))


def _add_grid_loading_javascript(
    m: folium.Map,
    overshooting_data: Optional[dict] = None,
    undershooting_data: Optional[dict] = None
):
    """
    Add JavaScript for displaying grid data embedded inline in the HTML.

    This approach embeds grid data directly as JavaScript objects to avoid CORS
    issues when opening the HTML file via file:// protocol.

    Args:
        m: Folium map object
        overshooting_data: Dict of cell_name -> grid data for overshooting cells
        undershooting_data: Dict of cell_name -> grid data for undershooting cells
    """
    map_var_name = m.get_name()

    overshooting_data = overshooting_data or {}
    undershooting_data = undershooting_data or {}

    js = f"""
    <script>
    // Store loaded grids by cell_name and type
    var loadedGrids = {{}};
    var gridLayers = {{}};

    // Grid data embedded inline (no CORS issues with file:// protocol)
    var overshootingGridData = {json.dumps(overshooting_data)};
    var undershootingGridData = {json.dumps(undershooting_data)};

    // Sanitize cell name for use in DOM element IDs (match Python _sanitize_js_name)
    function sanitizeJsName(name) {{
        return name.replace(/[^a-zA-Z0-9_]/g, '_');
    }}

    // Function to load and display grids for a cell
    function loadGridsForCell(cellId, gridType) {{
        var data = gridType === 'overshooting' ? overshootingGridData[cellId] : undershootingGridData[cellId];
        if (!data) {{
            console.log('No grid data for cell ' + cellId + ' type ' + gridType);
            return;
        }}

        var safeCellId = sanitizeJsName(cellId);
        var key = gridType + '_' + safeCellId;
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

        // Render from embedded data
        if (statusDiv) statusDiv.textContent = 'Loading...';
        if (btn) btn.disabled = true;

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
    }}

    // Create toggle functions for each cell
    """

    # Add toggle functions for overshooting cells
    for cell_name in overshooting_data.keys():
        safe_name = _sanitize_js_name(cell_name)
        js += f"""
    function toggleOvershootingGrids_{safe_name}() {{
        loadGridsForCell('{cell_name}', 'overshooting');
    }}
    """

    # Add toggle functions for undershooting cells
    for cell_name in undershooting_data.keys():
        safe_name = _sanitize_js_name(cell_name)
        js += f"""
    function toggleUndershootingGrids_{safe_name}() {{
        loadGridsForCell('{cell_name}', 'undershooting');
    }}
    """

    js += "</script>"
    m.get_root().html.add_child(folium.Element(js))


def _build_cell_coords_lookup(gis_df: pd.DataFrame) -> dict:
    """
    Build a lookup dictionary of cell coordinates from GIS data.

    Args:
        gis_df: DataFrame with cell GIS data

    Returns:
        Dict mapping cell_name to [latitude, longitude]
    """
    if gis_df is None or len(gis_df) == 0:
        return {}

    cell_coords = {}
    lat_col = 'latitude' if 'latitude' in gis_df.columns else 'Latitude'
    lon_col = 'longitude' if 'longitude' in gis_df.columns else 'Longitude'
    name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'

    for _, row in gis_df.iterrows():
        cell_name = str(row.get(name_col, ''))
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if cell_name and pd.notna(lat) and pd.notna(lon):
            cell_coords[cell_name] = [float(lat), float(lon)]

    return cell_coords


def _build_cell_geometries_lookup(gis_df: pd.DataFrame) -> dict:
    """
    Build a lookup dictionary of cell sector geometries from GIS data.

    Args:
        gis_df: DataFrame with cell GIS data including geometry column

    Returns:
        Dict mapping cell_name to list of [lat, lon] coordinate pairs forming the polygon
    """
    from shapely import wkt

    if gis_df is None or len(gis_df) == 0:
        return {}

    cell_geometries = {}
    name_col = 'cell_name' if 'cell_name' in gis_df.columns else 'CILAC'
    geom_col = 'geometry' if 'geometry' in gis_df.columns else None

    if geom_col is None:
        return {}

    for _, row in gis_df.iterrows():
        cell_name = str(row.get(name_col, ''))
        geom_str = row.get(geom_col)

        if cell_name and geom_str and pd.notna(geom_str):
            try:
                geom = wkt.loads(str(geom_str))
                if hasattr(geom, 'exterior'):
                    # Extract coordinates from polygon exterior ring
                    # Note: Folium/Leaflet expects [lat, lon] but WKT is [lon, lat]
                    coords = [[y, x] for x, y in geom.exterior.coords]
                    cell_geometries[cell_name] = coords
            except Exception:
                continue

    return cell_geometries


def _add_line_toggle_javascript(
    m: folium.Map,
    line_data: dict,
    cell_coords: dict,
    hull_geometries: Optional[dict] = None,
) -> None:
    """
    Add JavaScript for dynamically drawing/hiding lines between features.

    Args:
        m: Folium map object
        line_data: Dict mapping feature_key to line configuration:
            {
                'pci_confusion_0': {
                    'source': [lat, lon],
                    'targets': [[lat, lon], ...],
                    'color': '#9b59b6',
                    'style': 'dotted',  # or 'dashed'
                },
                ...
            }
        cell_coords: Dict mapping cell_name to [lat, lon]
        hull_geometries: Dict mapping feature_key to hull geometry data for CA imbalance
    """
    if not line_data and not hull_geometries:
        return

    map_var_name = m.get_name()

    js = f"""
    <script>
    // Line toggle state and layers
    var lineLayers = {{}};
    var lineLayersVisible = {{}};

    // Cell coordinates lookup
    var cellCoords = {json.dumps(cell_coords)};

    // Line data for each feature
    var lineData = {json.dumps(line_data)};

    // Generic function to toggle lines for any feature
    function toggleLines(featureKey, btnId) {{
        var btn = document.getElementById(btnId);
        var statusDiv = document.getElementById('lineStatus_' + featureKey);

        if (lineLayersVisible[featureKey]) {{
            // Hide lines
            if (lineLayers[featureKey]) {{
                {map_var_name}.removeLayer(lineLayers[featureKey]);
            }}
            lineLayersVisible[featureKey] = false;
            if (btn) btn.textContent = btn.dataset.showText || 'Show Lines';
            if (statusDiv) statusDiv.textContent = '';
        }} else {{
            // Show lines
            var data = lineData[featureKey];
            if (data && data.source && data.targets && data.targets.length > 0) {{
                var lineGroup = L.layerGroup();
                var source = data.source;

                data.targets.forEach(function(targetInfo) {{
                    var target = targetInfo.coords || targetInfo;
                    var lineColor = targetInfo.color || data.color || '#999';

                    var style = {{
                        color: lineColor,
                        weight: 2,
                        opacity: 0.8
                    }};

                    if (data.style === 'dotted') {{
                        style.dashArray = '5, 10';
                    }} else if (data.style === 'dashed') {{
                        style.dashArray = '10, 5';
                    }}

                    var line = L.polyline([source, target], style);

                    // Add tooltip if target has name
                    if (targetInfo.name) {{
                        line.bindTooltip(targetInfo.name, {{permanent: false, direction: 'center'}});
                    }}

                    line.addTo(lineGroup);

                    // Add polygon if geometry available, otherwise small circle at target
                    if (targetInfo.geometry && targetInfo.geometry.length > 0) {{
                        var polygon = L.polygon(targetInfo.geometry, {{
                            color: lineColor,
                            weight: 2,
                            fill: true,
                            fillColor: lineColor,
                            fillOpacity: 0.3
                        }});
                        if (targetInfo.name) {{
                            polygon.bindTooltip(targetInfo.name, {{permanent: false, direction: 'center'}});
                        }}
                        polygon.addTo(lineGroup);
                    }} else {{
                        L.circleMarker(target, {{
                            radius: 4,
                            color: lineColor,
                            fillColor: lineColor,
                            fillOpacity: 0.8,
                            weight: 1
                        }}).addTo(lineGroup);
                    }}
                }});

                lineGroup.addTo({map_var_name});
                lineLayers[featureKey] = lineGroup;
                lineLayersVisible[featureKey] = true;

                if (btn) btn.textContent = btn.dataset.hideText || 'Hide Lines';
                if (statusDiv) statusDiv.textContent = data.targets.length + ' connections shown';
            }} else {{
                if (statusDiv) statusDiv.textContent = 'No connections available';
            }}
        }}
    }}

    // Hull toggle state and layers
    var hullLayers = {{}};
    var hullLayersVisible = {{}};
    var hullGeometries = {json.dumps(hull_geometries if hull_geometries else {})};


    // Function to toggle hull display for CA imbalance
    function toggleHulls(featureKey, btnId) {{
        var btn = document.getElementById(btnId);
        var statusDiv = document.getElementById('hullStatus_' + featureKey);

        if (hullLayersVisible[featureKey]) {{
            // Hide hulls
            if (hullLayers[featureKey]) {{
                {map_var_name}.removeLayer(hullLayers[featureKey]);
            }}
            hullLayersVisible[featureKey] = false;
            if (btn) btn.textContent = 'Show Hulls';
            if (statusDiv) statusDiv.textContent = '';
        }} else {{
            // Show hulls
            var data = hullGeometries[featureKey];
            if (data) {{
                var hullGroup = L.layerGroup();

                // Coverage cell hull (blue)
                if (data.coverage) {{
                    L.geoJSON(data.coverage, {{
                        style: {{
                            fillColor: '{LINE_COLORS["ca_coverage"]}',
                            color: '{LINE_COLORS["ca_coverage"]}',
                            weight: 2,
                            fillOpacity: 0.2
                        }}
                    }}).bindTooltip('Coverage Band', {{permanent: false}}).addTo(hullGroup);
                }}

                // Capacity cell hull (green)
                if (data.capacity) {{
                    L.geoJSON(data.capacity, {{
                        style: {{
                            fillColor: '{LINE_COLORS["ca_capacity"]}',
                            color: '{LINE_COLORS["ca_capacity"]}',
                            weight: 2,
                            fillOpacity: 0.2
                        }}
                    }}).bindTooltip('Capacity Band', {{permanent: false}}).addTo(hullGroup);
                }}

                hullGroup.addTo({map_var_name});
                hullLayers[featureKey] = hullGroup;
                hullLayersVisible[featureKey] = true;

                if (btn) btn.textContent = 'Hide Hulls';
                if (statusDiv) statusDiv.textContent = 'Coverage comparison shown';
            }} else {{
                if (statusDiv) statusDiv.textContent = 'Hull data not available';
            }}
        }}
    }}
    </script>
    """
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
