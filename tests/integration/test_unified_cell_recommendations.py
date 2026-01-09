"""
Generate unified cell-level recommendations combining all analyses.

This script:
1. Runs overshooting detection
2. Runs undershooting detection
3. Runs no coverage gap detection
4. Runs low coverage detection
5. Consolidates all findings into a single cell-level report with 4 columns:
   - overshooter: Yes/No (or count of overshooting issues)
   - undershooter: Yes/No (or count of undershooting issues)
   - no_coverage: Yes/No (if cell is recommended for coverage gaps)
   - low_coverage: Yes/No (if cell is recommended for low coverage gaps)
"""
from pathlib import Path
import sys
import pandas as pd

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
from ran_optimizer.recommendations.overshooters import OvershooterDetector, OvershooterParams
from ran_optimizer.recommendations.undershooters import UndershooterDetector, UndershooterParams
from ran_optimizer.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def run_unified_cell_recommendations():
    """Generate unified cell-level recommendations."""
    # Configure logging
    configure_logging(log_level="INFO", json_output=False)

    logger.info("=== Starting Unified Cell Recommendations Generation ===")

    # Define file paths
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "vf-ie"
    input_path = data_path / "input-data"
    output_path = data_path / "output-data"
    output_path.mkdir(parents=True, exist_ok=True)

    grid_file = input_path / "cell_coverage.csv"
    gis_file = input_path / "cork-gis.csv"
    hulls_file = input_path / "cell_hulls.csv"
    boundary_file = input_path / "county_bounds" / "bounds.shp"

    # Load data - use raw CSV loading like the working undershooting test
    logger.info("loading_data")
    from ran_optimizer.utils.dtypes import load_cell_coverage_csv
    import pandas as pd

    df = load_cell_coverage_csv(grid_file)
    logger.info("loaded_raw_csv", rows=len(df))

    # Prepare grid data - match working standalone test format
    grid_data = df[[
        'grid',
        'cilac',
        'Band',  # CRITICAL: Include frequency band for RSRP competition
        'avg_rsrp',
        'event_count',
        'distance_to_cell',
    ]].copy()

    grid_data = grid_data.rename(columns={
        'grid': 'geohash7',
        'cilac': 'cell_id',
        'Band': 'Band',  # Keep as 'Band'
        'avg_rsrp': 'rsrp',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
    })

    # Prepare GIS data - match working undershooting test format
    gis_data = df[['cilac', 'Latitude', 'Longitude', 'Bearing', 'TiltM', 'TiltE', 'Height']].drop_duplicates('cilac').copy()
    gis_data = gis_data.rename(columns={
        'cilac': 'cell_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Bearing': 'azimuth_deg',
        'TiltM': 'mechanical_tilt',
        'TiltE': 'electrical_tilt',
        'Height': 'height',
    })

    # Load cell hulls and GIS for coverage gaps
    cell_hulls = load_cell_hulls(hulls_file)
    gis_full = load_gis_data(gis_file, operator="Vodafone_Ireland", validate=False)
    logger.info("data_loaded", grid_rows=len(grid_data), gis_rows=len(gis_data), hulls=len(cell_hulls))

    # Get list of all cells
    all_cells = set(gis_data['cell_id'].astype(str).unique())
    logger.info("total_cells_in_network", count=len(all_cells))

    # Initialize results dataframe with all cells
    results = pd.DataFrame({
        'cell_name': sorted(all_cells),
        'overshooter': 'No',
        'undershooter': 'No',
        'no_coverage': 'No',
        'low_coverage': 'No'
    })

    # 1. OVERSHOOTING DETECTION
    logger.info("=== Running Overshooting Detection ===")
    try:
        # Use DEFAULT parameters (no environment override) to match standalone test
        overshot_params = OvershooterParams()
        overshot_detector = OvershooterDetector(overshot_params)
        overshot_results = overshot_detector.detect(grid_data, gis_data)

        if len(overshot_results) > 0:
            overshot_cells = set(overshot_results['cell_id'].astype(str).unique())
            results.loc[results['cell_name'].isin(overshot_cells), 'overshooter'] = 'Yes'
            logger.info("overshooting_complete", overshooters_found=len(overshot_cells))
        else:
            logger.info("no_overshooters_found")
    except Exception as e:
        logger.error("overshooting_detection_failed", error=str(e))

    # 2. UNDERSHOOTING DETECTION
    logger.info("=== Running Undershooting Detection ===")
    try:
        # Use DEFAULT parameters (no environment override) to match standalone test
        undershot_params = UndershooterParams()
        undershot_detector = UndershooterDetector(undershot_params)
        undershot_results = undershot_detector.detect(grid_data, gis_data)

        if len(undershot_results) > 0:
            undershot_cells = set(undershot_results['cell_id'].astype(str).unique())
            results.loc[results['cell_name'].isin(undershot_cells), 'undershooter'] = 'Yes'
            logger.info("undershooting_complete", undershooters_found=len(undershot_cells))
        else:
            logger.info("no_undershooters_found")
    except Exception as e:
        logger.error("undershooting_detection_failed", error=str(e))

    # 3. NO COVERAGE GAP DETECTION
    logger.info("=== Running No Coverage Gap Detection ===")
    try:
        # Use DEFAULT parameters (no environment override)
        gap_params = CoverageGapParams.from_config(
            base_path / "config" / "coverage_gaps.json"
        )
        gap_detector = CoverageGapDetector(gap_params)
        gap_clusters = gap_detector.detect(cell_hulls)

        if len(gap_clusters) > 0:
            gap_analyzer = CoverageGapAnalyzer(gap_params)
            # Need to prepare grid data with decoded geohash for gap analysis
            grid_for_gaps = df[['grid', 'cilac', 'Latitude', 'Longitude']].copy()
            grid_for_gaps = grid_for_gaps.rename(columns={
                'grid': 'geohash7',
                'cilac': 'cell_name',
                'Latitude': 'latitude',
                'Longitude': 'longitude',
            })
            gap_analysis = gap_analyzer.find_cells_for_gaps(gap_clusters, grid_for_gaps)

            # Extract nearby cells for each gap
            gap_cells = set()
            for _, row in gap_analysis.iterrows():
                nearby = row.get('nearby_cells', [])
                if isinstance(nearby, list):
                    gap_cells.update(str(c) for c in nearby)

            results.loc[results['cell_name'].isin(gap_cells), 'no_coverage'] = 'Yes'
            logger.info("no_coverage_complete", gap_clusters=len(gap_clusters), recommended_cells=len(gap_cells))
        else:
            logger.info("no_coverage_gaps_found")
    except Exception as e:
        logger.error("no_coverage_detection_failed", error=str(e))

    # 4. LOW COVERAGE DETECTION
    logger.info("=== Running Low Coverage Detection ===")
    try:
        # Use DEFAULT parameters (no environment override)
        low_cov_params = LowCoverageParams.from_config(
            base_path / "config" / "coverage_gaps.json"
        )
        low_cov_detector = LowCoverageDetector(
            low_cov_params,
            boundary_shapefile=str(boundary_file) if boundary_file.exists() else None
        )
        # Prepare grid data for low coverage (needs geohash7, rsrp, and band)
        grid_with_band = df[['grid', 'avg_rsrp', 'cilac']].copy()
        # Add band from GIS
        cell_band = gis_full[['Name', 'Band']].drop_duplicates('Name').copy()
        cell_band.columns = ['cell_name', 'band']
        cell_band['cell_name'] = cell_band['cell_name'].astype(str)
        grid_with_band['cell_name'] = grid_with_band['cilac'].astype(str)
        grid_with_band = grid_with_band.merge(cell_band, on='cell_name', how='left')
        grid_with_band = grid_with_band.rename(columns={'grid': 'geohash7', 'avg_rsrp': 'rsrp'})

        low_coverage_by_band = low_cov_detector.detect(cell_hulls, grid_with_band, gis_full)

        if len(low_coverage_by_band) > 0:
            # For low coverage, we need to find which cells serve these areas
            # Similar to gap analysis but for low coverage clusters
            low_cov_cells = set()

            for band, clusters in low_coverage_by_band.items():
                if len(clusters) > 0:
                    # For each cluster, find nearby cells
                    for _, cluster in clusters.iterrows():
                        # Get cluster centroid
                        centroid_lat = cluster['centroid_lat']
                        centroid_lon = cluster['centroid_lon']

                        # Find nearest cells to this cluster
                        # Calculate distance from each cell to cluster centroid
                        gis_with_dist = gis_full.copy()
                        gis_with_dist['dist_to_cluster'] = (
                            (gis_with_dist['Latitude'] - centroid_lat) ** 2 +
                            (gis_with_dist['Longitude'] - centroid_lon) ** 2
                        ) ** 0.5

                        # Get nearest cells (top 5) - use cell_id not Name
                        nearest = gis_with_dist.nsmallest(5, 'dist_to_cluster')
                        low_cov_cells.update(nearest['cell_id'].astype(str).unique())

            results.loc[results['cell_name'].isin(low_cov_cells), 'low_coverage'] = 'Yes'
            logger.info("low_coverage_complete",
                       bands=len(low_coverage_by_band),
                       recommended_cells=len(low_cov_cells))
        else:
            logger.info("no_low_coverage_found")
    except Exception as e:
        logger.error("low_coverage_detection_failed", error=str(e))

    # Save consolidated results
    output_file = output_path / "unified_cell_recommendations.csv"
    results.to_csv(output_file, index=False)
    logger.info("results_saved", file=str(output_file))

    # Print summary statistics
    print("\n" + "=" * 80)
    print("UNIFIED CELL RECOMMENDATIONS SUMMARY")
    print("=" * 80)
    print(f"Total Cells Analyzed:        {len(results):,}")
    print(f"\nIssue Breakdown:")
    print(f"  Overshooters:              {(results['overshooter'] == 'Yes').sum():,}")
    print(f"  Undershooters:             {(results['undershooter'] == 'Yes').sum():,}")
    print(f"  No Coverage (nearby):      {(results['no_coverage'] == 'Yes').sum():,}")
    print(f"  Low Coverage (nearby):     {(results['low_coverage'] == 'Yes').sum():,}")

    # Count cells with multiple issues
    results['issue_count'] = (
        (results['overshooter'] == 'Yes').astype(int) +
        (results['undershooter'] == 'Yes').astype(int) +
        (results['no_coverage'] == 'Yes').astype(int) +
        (results['low_coverage'] == 'Yes').astype(int)
    )

    print(f"\nCells with Multiple Issues:")
    for i in range(2, 5):
        count = (results['issue_count'] == i).sum()
        if count > 0:
            print(f"  {i} issues:                  {count:,}")

    print(f"\nCells with NO Issues:        {(results['issue_count'] == 0).sum():,}")
    print(f"Cells with ANY Issue:        {(results['issue_count'] > 0).sum():,}")

    print(f"\nOutput saved to: {output_file}")
    print("=" * 80 + "\n")

    # Show sample of cells with issues
    cells_with_issues = results[results['issue_count'] > 0].head(20)
    if len(cells_with_issues) > 0:
        print("\nSample Cells with Issues (first 20):")
        print(cells_with_issues.to_string(index=False))

    logger.info("=== Unified Cell Recommendations Complete ===")


if __name__ == "__main__":
    run_unified_cell_recommendations()
