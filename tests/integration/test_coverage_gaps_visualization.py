"""
Integration test for coverage gap detection and visualization with real data.

Tests the correct notebook-based approach:
1. Load cell hulls (not grid data!)
2. Cluster cell hulls to group nearby coverage
3. For each cluster, find gap polygons (uncovered areas)
4. Get geohashes in gap polygons
5. Apply k-ring density filtering
6. Cluster gap geohashes with HDBSCAN
7. Create alpha shape polygons
8. Visualize on map
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ran_optimizer.data.loaders import load_grid_data, load_gis_data, load_cell_hulls
from ran_optimizer.recommendations.coverage_gaps import (
    CoverageGapParams,
    CoverageGapDetector,
    CoverageGapAnalyzer,
    LowCoverageParams,
    LowCoverageDetector
)
from ran_optimizer.utils.logging_config import configure_logging, get_logger
import folium
import pandas as pd

logger = get_logger(__name__)


def test_coverage_gaps_detection_and_visualization():
    """
    Test coverage gap detection with real Vodafone Ireland data.

    This test:
    1. Loads real grid data and GIS data
    2. Generates cell convex hulls
    3. Detects coverage gap clusters using the correct notebook approach
    4. Creates alpha shape polygons for clusters
    5. Generates map visualization with gap cluster polygons
    """
    # Configure logging
    configure_logging(log_level="INFO", json_output=False)

    logger.info("=== Starting Coverage Gap Detection Test ===")

    # Define file paths
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "vf-ie"
    input_path = data_path / "input-data"
    output_path = data_path / "output-data" / "maps"

    grid_file = input_path / "cell_coverage.csv"
    gis_file = input_path / "cork-gis.csv"
    hulls_file = input_path / "cell_hulls.csv"

    # Check files exist
    assert grid_file.exists(), f"Grid file not found: {grid_file}"
    assert gis_file.exists(), f"GIS file not found: {gis_file}"
    assert hulls_file.exists(), f"Hulls file not found: {hulls_file}"

    logger.info("loading_data_files")

    # Load grid data (for gap analysis only)
    logger.info("loading_grid_data", file=str(grid_file))
    grid_data = load_grid_data(
        grid_file,
        operator="Vodafone_Ireland",
        validate=False,  # Skip validation for integration test
        decode_geohash=True
    )
    logger.info("grid_loaded", rows=len(grid_data))

    # Load GIS data for cell sites visualization
    logger.info("loading_gis_data", file=str(gis_file))
    gis_data = load_gis_data(gis_file, operator="Vodafone_Ireland", validate=False)
    logger.info("gis_loaded", cells=len(gis_data))

    # Load cell hulls
    logger.info("loading_cell_hulls", file=str(hulls_file))
    cell_hulls = load_cell_hulls(hulls_file)
    logger.info("cell_hulls_loaded", hulls=len(cell_hulls))

    # Add band information to grid data by joining with GIS data
    logger.info("adding_band_info_to_grid")
    # Find band column in GIS data (case-insensitive)
    band_col = None
    for col in gis_data.columns:
        if col.lower() == 'band':
            band_col = col
            break

    # Find cell identifier columns (case-insensitive)
    grid_cell_col = None
    gis_cell_col = None
    for col in grid_data.columns:
        if col.lower() in ['cell_name', 'cellname', 'name']:
            grid_cell_col = col
            break
    for col in gis_data.columns:
        if col.lower() in ['cell_name', 'cellname', 'name']:
            gis_cell_col = col
            break

    if band_col and grid_cell_col and gis_cell_col:
        # Create a mapping of cell_name -> band
        cell_band_map = gis_data[[gis_cell_col, band_col]].drop_duplicates(gis_cell_col).copy()
        cell_band_map.columns = ['cell_name', 'band']  # Normalize column names

        # Convert cell_name to string in both dataframes to ensure type match
        cell_band_map['cell_name'] = cell_band_map['cell_name'].astype(str)
        grid_data_cell_col_str = grid_data[grid_cell_col].astype(str)

        # Join band info to grid data
        grid_data = grid_data.merge(
            cell_band_map,
            left_on=grid_cell_col,
            right_on='cell_name',
            how='left',
            suffixes=('', '_map')
        )

        # Count how many grid points have band info
        band_coverage = grid_data['band'].notna().sum()
        logger.info("band_info_added", rows_with_band=band_coverage, total_rows=len(grid_data))
    else:
        logger.warning("band_join_skipped", reason="Missing required columns")

    # Configure coverage gap detection parameters
    config_path = base_path / "config" / "coverage_gaps.json"
    params = CoverageGapParams.from_config(config_path, environment="suburban")

    logger.info(
        "coverage_gap_params",
        cell_cluster_eps_km=params.cell_cluster_eps_km,
        k_ring_steps=params.k_ring_steps,
        min_missing_neighbors=params.min_missing_neighbors
    )

    # Detect coverage gaps
    logger.info("detecting_coverage_gaps")
    detector = CoverageGapDetector(params)
    gap_clusters = detector.detect(cell_hulls)

    logger.info(
        "coverage_gaps_detected",
        num_clusters=len(gap_clusters)
    )

    if len(gap_clusters) > 0:
        logger.info("gap_cluster_summary")
        for _, cluster in gap_clusters.iterrows():
            logger.info(
                "cluster_details",
                cluster_id=int(cluster['cluster_id']),
                grid_count=int(cluster['n_points']),
                centroid_lat=float(cluster['centroid_lat']),
                centroid_lon=float(cluster['centroid_lon'])
            )

        # Analyze gaps (find nearby cells)
        logger.info("analyzing_gap_serving_cells")
        analyzer = CoverageGapAnalyzer(params)
        gap_analysis = analyzer.find_cells_for_gaps(gap_clusters, grid_data)

        logger.info("gap_analysis_complete", analyzed_clusters=len(gap_analysis))
        for _, analysis in gap_analysis.iterrows():
            logger.info(
                "gap_analysis_result",
                cluster_id=int(analysis['cluster_id']),
                nearby_cell_count=int(analysis['nearby_cell_count']),
                avg_distance_m=float(analysis['avg_distance_to_coverage_m'])
            )

    # Detect LOW COVERAGE (band-specific)
    logger.info("detecting_low_coverage")
    low_cov_params = LowCoverageParams.from_config(config_path, environment="suburban")

    # Use boundary shapefile to filter out offshore gaps
    boundary_shapefile = input_path / "county_bounds" / "bounds.shp"
    low_cov_detector = LowCoverageDetector(
        low_cov_params,
        boundary_shapefile=str(boundary_shapefile) if boundary_shapefile.exists() else None
    )

    logger.info(
        "low_coverage_params",
        rsrp_threshold_dbm=low_cov_params.rsrp_threshold_dbm,
        k_ring_steps=low_cov_params.k_ring_steps,
        min_missing_neighbors=low_cov_params.min_missing_neighbors
    )

    low_coverage_by_band = low_cov_detector.detect(cell_hulls, grid_data, gis_data)

    logger.info(
        "low_coverage_detected",
        num_bands=len(low_coverage_by_band)
    )

    if len(low_coverage_by_band) > 0:
        logger.info("low_coverage_summary")
        for band, clusters in low_coverage_by_band.items():
            logger.info(
                "band_low_coverage",
                band=band,
                num_clusters=len(clusters)
            )
            for _, cluster in clusters.iterrows():
                logger.info(
                    "low_coverage_cluster_details",
                    band=band,
                    cluster_id=int(cluster['cluster_id']),
                    grid_count=int(cluster['n_points']),
                    centroid_lat=float(cluster['centroid_lat']),
                    centroid_lon=float(cluster['centroid_lon'])
                )

    # Create visualization
    if len(gap_clusters) > 0 or len(low_coverage_by_band) > 0:
        logger.info("creating_map_visualization")
        output_path.mkdir(parents=True, exist_ok=True)
        map_file = output_path / "coverage_gaps_map.html"

        # Calculate map center from all clusters (no coverage + low coverage)
        all_lats = []
        all_lons = []

        if len(gap_clusters) > 0:
            all_lats.extend(gap_clusters['centroid_lat'].tolist())
            all_lons.extend(gap_clusters['centroid_lon'].tolist())

        for band, clusters in low_coverage_by_band.items():
            all_lats.extend(clusters['centroid_lat'].tolist())
            all_lons.extend(clusters['centroid_lon'].tolist())

        center_lat = sum(all_lats) / len(all_lats) if all_lats else gis_data['latitude'].mean()
        center_lon = sum(all_lons) / len(all_lons) if all_lons else gis_data['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Add coverage gap cluster polygons layer
        gap_layer = folium.FeatureGroup(name="Coverage Gap Clusters", show=True)

        # Add each cluster polygon
        for _, cluster in gap_clusters.iterrows():
            cluster_id = int(cluster['cluster_id'])
            n_points = int(cluster['n_points'])
            centroid_lat = float(cluster['centroid_lat'])
            centroid_lon = float(cluster['centroid_lon'])
            geometry = cluster['geometry']

            # Get analysis data for this cluster
            cluster_analysis = gap_analysis[gap_analysis['cluster_id'] == cluster_id]
            nearby_cells = []
            avg_dist = 0
            if len(cluster_analysis) > 0:
                nearby_cells = cluster_analysis.iloc[0]['nearby_cells']
                avg_dist = cluster_analysis.iloc[0]['avg_distance_to_coverage_m']

            # Add cluster polygon as GeoJson
            folium.GeoJson(
                geometry,
                style_function=lambda x: {
                    'fillColor': 'red',
                    'color': 'darkred',
                    'weight': 2,
                    'fillOpacity': 0.3
                },
                popup=folium.Popup(
                    f"<b>Coverage Gap Cluster {cluster_id}</b><br>"
                    f"Grid Count: {n_points}<br>"
                    f"Centroid: ({centroid_lat:.4f}, {centroid_lon:.4f})<br>"
                    f"Nearby Cells: {len(nearby_cells)}<br>"
                    f"Avg Distance: {avg_dist:.0f}m",
                    max_width=300
                ),
                tooltip=f"Gap Cluster {cluster_id} ({n_points} grids)"
            ).add_to(gap_layer)

            # Add cluster centroid marker
            folium.Marker(
                location=[centroid_lat, centroid_lon],
                popup=folium.Popup(
                    f"<b>Gap Cluster {cluster_id} Centroid</b><br>"
                    f"Grid Count: {n_points}<br>"
                    f"Nearby Cells: {', '.join(str(c) for c in nearby_cells[:5])}",
                    max_width=300
                ),
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                tooltip=f"Cluster {cluster_id} Centroid"
            ).add_to(gap_layer)

        gap_layer.add_to(m)

        # Add low coverage layers (one per band)
        band_colors = {
            'Band 20': 'orange',
            'Band 3': 'yellow',
            'Band 7': 'purple',
            'Band 1': 'pink',
            'Band 8': 'lightblue',
            'Band 28': 'lightgreen'
        }

        for band, clusters in low_coverage_by_band.items():
            band_layer = folium.FeatureGroup(name=f"Low Coverage - {band}", show=True)
            color = band_colors.get(band, 'orange')

            for _, cluster in clusters.iterrows():
                cluster_id = int(cluster['cluster_id'])
                n_points = int(cluster['n_points'])
                centroid_lat = float(cluster['centroid_lat'])
                centroid_lon = float(cluster['centroid_lon'])
                geometry = cluster['geometry']

                # Add cluster polygon as GeoJson
                folium.GeoJson(
                    geometry,
                    style_function=lambda x, c=color: {
                        'fillColor': c,
                        'color': c,
                        'weight': 2,
                        'fillOpacity': 0.3
                    },
                    popup=folium.Popup(
                        f"<b>Low Coverage - {band}</b><br>"
                        f"Cluster {cluster_id}<br>"
                        f"Grid Count: {n_points}<br>"
                        f"Centroid: ({centroid_lat:.4f}, {centroid_lon:.4f})<br>"
                        f"RSRP ≤ {low_cov_params.rsrp_threshold_dbm} dBm",
                        max_width=300
                    ),
                    tooltip=f"{band} Low Coverage {cluster_id} ({n_points} grids)"
                ).add_to(band_layer)

                # Add cluster centroid marker
                folium.Marker(
                    location=[centroid_lat, centroid_lon],
                    popup=folium.Popup(
                        f"<b>{band} Low Coverage Cluster {cluster_id}</b><br>"
                        f"Grid Count: {n_points}<br>"
                        f"RSRP ≤ {low_cov_params.rsrp_threshold_dbm} dBm",
                        max_width=300
                    ),
                    icon=folium.Icon(color='orange', icon='signal', prefix='fa'),
                    tooltip=f"{band} Cluster {cluster_id}"
                ).add_to(band_layer)

            band_layer.add_to(m)

        # Add cell sites layer
        sites_layer = folium.FeatureGroup(name="Cell Sites", show=True)
        for _, site in gis_data.iterrows():
            # Build popup text with available fields
            popup_text = f"<b>{site.get('site_name', 'Unknown')}</b><br>"
            popup_text += f"Cell: {site.get('cell_id', 'Unknown')}<br>"
            if 'azimuth' in site and pd.notna(site['azimuth']):
                popup_text += f"Azimuth: {site['azimuth']}°"

            folium.CircleMarker(
                location=[site['latitude'], site['longitude']],
                radius=4,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.6,
                popup=popup_text,
                weight=2
            ).add_to(sites_layer)

        sites_layer.add_to(m)

        # Add cell hulls layer - grey with high opacity (transparent) to show coverage
        hulls_layer = folium.FeatureGroup(name="Cell Coverage Hulls", show=True)
        for _, hull in cell_hulls.iterrows():
            folium.GeoJson(
                hull['geometry'],
                style_function=lambda x: {
                    'fillColor': 'grey',
                    'color': 'darkgrey',
                    'weight': 1,
                    'fillOpacity': 0.15,  # High opacity = very transparent
                    'opacity': 0.3
                },
                tooltip=f"Cell {hull.get('cell_name', hull.get('cell_id', 'Unknown'))}"
            ).add_to(hulls_layer)
        hulls_layer.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        m.save(str(map_file))
        logger.info("map_saved", file=str(map_file))

        print(f"\n{'='*70}")
        print(f"Coverage Detection Results:")
        print(f"{'='*70}")
        print(f"Grid Points Analyzed:       {len(grid_data):,}")
        print(f"Cell Hulls Generated:       {len(cell_hulls)}")
        print(f"No Coverage Clusters:       {len(gap_clusters)}")

        if len(gap_clusters) > 0:
            print(f"\nNo Coverage Gap Details:")
            print(f"{'-'*70}")
            for _, cluster in gap_clusters.iterrows():
                print(f"  Cluster {int(cluster['cluster_id'])}: "
                      f"{int(cluster['n_points'])} grids at "
                      f"({cluster['centroid_lat']:.4f}, {cluster['centroid_lon']:.4f})")

        print(f"\nLow Coverage Bands Found:   {len(low_coverage_by_band)}")
        if len(low_coverage_by_band) > 0:
            print(f"\nLow Coverage Details:")
            print(f"{'-'*70}")
            for band, clusters in low_coverage_by_band.items():
                print(f"  {band}: {len(clusters)} clusters")
                for _, cluster in clusters.iterrows():
                    print(f"    Cluster {int(cluster['cluster_id'])}: "
                          f"{int(cluster['n_points'])} grids at "
                          f"({cluster['centroid_lat']:.4f}, {cluster['centroid_lon']:.4f})")

        print(f"\nVisualization saved to: {map_file}")
        print(f"Open in browser: file://{map_file.absolute()}")
        print(f"{'='*70}\n")

    else:
        logger.info("no_coverage_issues_found", message="No coverage gaps or low coverage areas found")
        print("\nNo significant coverage issues found in the analyzed data.")

    logger.info("=== Coverage Gap Detection Test Complete ===")


if __name__ == "__main__":
    test_coverage_gaps_detection_and_visualization()
