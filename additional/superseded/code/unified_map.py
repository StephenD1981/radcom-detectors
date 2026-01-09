"""
Unified map visualization showing all RAN optimization layers.

This module creates a comprehensive interactive map with 5 layers:
1. Overshooter cells (red)
2. Undershooter cells (blue)
3. No coverage gaps (yellow polygons)
4. Low coverage areas (orange polygons)
5. All cell convex-hulls (light blue polygons)
"""
import folium
from folium import plugins
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional
import json

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_unified_map(
    overshooting_csv: Path,
    undershooting_csv: Path,
    gis_data: pd.DataFrame,
    cell_hulls: gpd.GeoDataFrame,
    no_coverage_polygons: Optional[gpd.GeoDataFrame] = None,
    low_coverage_polygons: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    output_file: Optional[Path] = None,
) -> folium.Map:
    """
    Create unified map with all RAN optimization layers.

    Args:
        overshooting_csv: Path to overshooting cells CSV
        undershooting_csv: Path to undershooting cells CSV
        gis_data: DataFrame with cell location data (needs cell_name, latitude, longitude)
        cell_hulls: GeoDataFrame with cell convex hulls
        no_coverage_polygons: GeoDataFrame with no coverage gap polygons (optional)
        low_coverage_polygons: Dict of band -> GeoDataFrame with low coverage polygons (optional)
        output_file: Path to save HTML map (optional)

    Returns:
        folium.Map object
    """
    # Load overshooting and undershooting data
    overshooting_df = pd.read_csv(overshooting_csv) if overshooting_csv.exists() else pd.DataFrame()
    undershooting_df = pd.read_csv(undershooting_csv) if undershooting_csv.exists() else pd.DataFrame()

    # Get center of map from GIS data
    center_lat = gis_data['latitude'].mean()
    center_lon = gis_data['longitude'].mean()

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Add layer control
    folium.plugins.Fullscreen().add_to(m)

    # Layer 1: Cell Convex Hulls (show by default, light blue, semi-transparent)
    hull_layer = folium.FeatureGroup(name='Cell Hulls', show=True)
    if len(cell_hulls) > 0:
        for idx, row in cell_hulls.iterrows():
            geom = row['geometry']
            cell_name = row.get('cell_name', str(idx))

            # Create popup with cell info
            popup_text = f"""
            <b>Cell:</b> {cell_name}<br>
            <b>Type:</b> Convex Hull
            """

            folium.GeoJson(
                geom,
                style_function=lambda x: {
                    'fillColor': '#ADD8E6',
                    'color': '#4682B4',
                    'weight': 1,
                    'fillOpacity': 0.15
                },
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(hull_layer)
    hull_layer.add_to(m)

    # Layer 2: No Coverage Gaps (yellow polygons)
    if no_coverage_polygons is not None and len(no_coverage_polygons) > 0:
        no_cov_layer = folium.FeatureGroup(name='No Coverage Gaps', show=True)
        for idx, row in no_coverage_polygons.iterrows():
            geom = row['geometry']
            area_km2 = row.get('area_km2', 0)
            cluster_id = row.get('cluster_id', idx)

            popup_text = f"""
            <b>Gap ID:</b> {cluster_id}<br>
            <b>Type:</b> No Coverage<br>
            <b>Area:</b> {area_km2:.2f} km²
            """

            folium.GeoJson(
                geom,
                style_function=lambda x: {
                    'fillColor': '#FFFF00',
                    'color': '#FFD700',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(no_cov_layer)
        no_cov_layer.add_to(m)

    # Layer 3: Low Coverage Areas (orange polygons, by band)
    if low_coverage_polygons is not None:
        for band, polygons_df in low_coverage_polygons.items():
            if len(polygons_df) > 0:
                low_cov_layer = folium.FeatureGroup(name=f'Low Coverage ({band})', show=True)
                for idx, row in polygons_df.iterrows():
                    geom = row['geometry']
                    area_km2 = row.get('area_km2', 0)
                    cluster_id = row.get('cluster_id', idx)

                    popup_text = f"""
                    <b>Gap ID:</b> {cluster_id}<br>
                    <b>Type:</b> Low Coverage<br>
                    <b>Band:</b> {band}<br>
                    <b>Area:</b> {area_km2:.2f} km²
                    """

                    folium.GeoJson(
                        geom,
                        style_function=lambda x: {
                            'fillColor': '#FFA500',
                            'color': '#FF8C00',
                            'weight': 2,
                            'fillOpacity': 0.4
                        },
                        popup=folium.Popup(popup_text, max_width=250)
                    ).add_to(low_cov_layer)
                low_cov_layer.add_to(m)

    # Layer 4: Overshooting Cells (red markers)
    if len(overshooting_df) > 0:
        overshooting_layer = folium.FeatureGroup(name='Overshooters', show=True)
        overshooting_cells = set(overshooting_df['cell_name'].astype(str))

        for _, cell in gis_data[gis_data['cell_name'].astype(str).isin(overshooting_cells)].iterrows():
            popup_text = f"""
            <b>Cell:</b> {cell['cell_name']}<br>
            <b>Type:</b> Overshooter<br>
            <b>Location:</b> ({cell['latitude']:.6f}, {cell['longitude']:.6f})
            """

            folium.CircleMarker(
                location=[cell['latitude'], cell['longitude']],
                radius=8,
                popup=folium.Popup(popup_text, max_width=250),
                color='#FF0000',
                fillColor='#FF0000',
                fillOpacity=0.7,
                weight=2
            ).add_to(overshooting_layer)
        overshooting_layer.add_to(m)

    # Layer 5: Undershooting Cells (blue markers)
    if len(undershooting_df) > 0:
        undershooting_layer = folium.FeatureGroup(name='Undershooters', show=True)
        undershooting_cells = set(undershooting_df['cell_name'].astype(str))

        for _, cell in gis_data[gis_data['cell_name'].astype(str).isin(undershooting_cells)].iterrows():
            popup_text = f"""
            <b>Cell:</b> {cell['cell_name']}<br>
            <b>Type:</b> Undershooter<br>
            <b>Location:</b> ({cell['latitude']:.6f}, {cell['longitude']:.6f})
            """

            folium.CircleMarker(
                location=[cell['latitude'], cell['longitude']],
                radius=8,
                popup=folium.Popup(popup_text, max_width=250),
                color='#0000FF',
                fillColor='#0000FF',
                fillOpacity=0.7,
                weight=2
            ).add_to(undershooting_layer)
        undershooting_layer.add_to(m)

    # Add layer control to toggle layers
    folium.LayerControl(collapsed=False).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <p style="margin:0; padding:0 0 5px 0; font-weight:bold;">Layer Legend</p>
    <p style="margin:0; padding:2px 0;"><span style="color:#FF0000;">●</span> Overshooters ({len(overshooting_df)})</p>
    <p style="margin:0; padding:2px 0;"><span style="color:#0000FF;">●</span> Undershooters ({len(undershooting_df)})</p>
    <p style="margin:0; padding:2px 0;"><span style="color:#FFFF00;">▬</span> No Coverage Gaps</p>
    <p style="margin:0; padding:2px 0;"><span style="color:#FFA500;">▬</span> Low Coverage</p>
    <p style="margin:0; padding:2px 0;"><span style="color:#ADD8E6;">▬</span> Cell Hulls ({len(cell_hulls)})</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save if output file specified
    if output_file is not None:
        m.save(str(output_file))
        logger.info("unified_map_saved", path=str(output_file))

    return m
