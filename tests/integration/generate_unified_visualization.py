"""
Generate unified RAN optimization visualization.

This script creates a single comprehensive map showing:
1. Overshooter cells (red markers)
2. Undershooter cells (blue markers)
3. No coverage gaps (yellow polygons)
4. Low coverage areas (orange polygons)
5. All cell convex-hulls (light blue polygons)
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
from ran_optimizer.visualization.unified_map import create_unified_map
from ran_optimizer.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Generate unified RAN optimization visualization."""
    # Configure logging
    configure_logging(log_level="INFO", json_output=False)

    logger.info("=== Generating Unified RAN Optimization Visualization ===")

    # Define file paths
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "vf-ie"
    input_path = data_path / "input-data"
    output_path = data_path / "output-data"
    maps_path = output_path / "maps"
    maps_path.mkdir(parents=True, exist_ok=True)

    grid_file = input_path / "cell_coverage.csv"
    gis_file = input_path / "cork-gis.csv"
    hulls_file = input_path / "cell_hulls.csv"
    boundary_file = input_path / "county_bounds" / "bounds.shp"

    overshooting_csv = output_path / "overshooting_cells.csv"
    undershooting_csv = output_path / "undershooting_cells.csv"

    output_file = maps_path / "unified_ran_optimization.html"

    # Load data
    logger.info("loading_data")
    from ran_optimizer.utils.dtypes import load_cell_coverage_csv

    df = load_cell_coverage_csv(grid_file)
    logger.info("loaded_raw_csv", rows=len(df))

    # Load GIS data
    gis_data = load_gis_data(gis_file, operator="Vodafone_Ireland", validate=False)
    logger.info("loaded_gis", rows=len(gis_data))

    # Load cell hulls
    cell_hulls = load_cell_hulls(hulls_file)
    logger.info("loaded_cell_hulls", count=len(cell_hulls))

    # Prepare GIS data for visualization (needs cell_id, latitude, longitude)
    # Note: load_gis_data already applies adapter which renames columns to lowercase
    gis_viz = gis_data[['cell_id', 'latitude', 'longitude']].copy()
    gis_viz['cell_id'] = gis_viz['cell_id'].astype(str)

    # Detect NO COVERAGE GAPS
    logger.info("=== Detecting No Coverage Gaps ===")
    no_coverage_polygons = None
    try:
        gap_params = CoverageGapParams.from_config(
            base_path / "config" / "coverage_gaps.json"
        )
        gap_detector = CoverageGapDetector(gap_params)
        gap_clusters = gap_detector.detect(cell_hulls)

        if len(gap_clusters) > 0:
            logger.info("no_coverage_gaps_detected", count=len(gap_clusters))
            no_coverage_polygons = gap_clusters
        else:
            logger.info("no_coverage_gaps_found")
    except Exception as e:
        logger.error("no_coverage_detection_failed", error=str(e))

    # Detect LOW COVERAGE AREAS
    logger.info("=== Detecting Low Coverage Areas ===")
    low_coverage_polygons = None
    try:
        low_cov_params = LowCoverageParams.from_config(
            base_path / "config" / "coverage_gaps.json"
        )
        low_cov_detector = LowCoverageDetector(
            low_cov_params,
            boundary_shapefile=str(boundary_file) if boundary_file.exists() else None
        )

        # Prepare grid data for low coverage (needs geohash7, rsrp, and band)
        # Note: df has original column names (Band), gis_data has adapter-renamed columns (band)
        grid_with_band = df[['grid', 'avg_rsrp', 'cilac', 'Band']].copy()
        grid_with_band = grid_with_band.rename(columns={
            'grid': 'geohash7',
            'avg_rsrp': 'rsrp',
            'cilac': 'cell_id',
            'Band': 'band'
        })

        low_coverage_by_band = low_cov_detector.detect(cell_hulls, grid_with_band, gis_data)

        if len(low_coverage_by_band) > 0:
            total_clusters = sum(len(clusters) for clusters in low_coverage_by_band.values())
            logger.info("low_coverage_detected", bands=len(low_coverage_by_band), total_clusters=total_clusters)
            low_coverage_polygons = low_coverage_by_band
        else:
            logger.info("no_low_coverage_found")
    except Exception as e:
        logger.error("low_coverage_detection_failed", error=str(e))

    # Create unified map
    logger.info("=== Creating Unified Map ===")
    unified_map = create_unified_map(
        overshooting_csv=overshooting_csv,
        undershooting_csv=undershooting_csv,
        gis_data=gis_viz,
        cell_hulls=cell_hulls,
        no_coverage_polygons=no_coverage_polygons,
        low_coverage_polygons=low_coverage_polygons,
        output_file=output_file
    )

    print("\n" + "=" * 80)
    print("UNIFIED RAN OPTIMIZATION VISUALIZATION")
    print("=" * 80)
    print(f"Map saved to: {output_file}")
    print(f"\nLayers included:")
    print(f"  1. Overshooters:    {len(pd.read_csv(overshooting_csv)) if overshooting_csv.exists() else 0} cells")
    print(f"  2. Undershooters:   {len(pd.read_csv(undershooting_csv)) if undershooting_csv.exists() else 0} cells")
    print(f"  3. No Coverage:     {len(no_coverage_polygons) if no_coverage_polygons is not None else 0} gaps")
    if low_coverage_polygons:
        for band, polygons in low_coverage_polygons.items():
            print(f"  4. Low Coverage ({band}): {len(polygons)} areas")
    else:
        print(f"  4. Low Coverage:    0 areas")
    print(f"  5. Cell Hulls:      {len(cell_hulls)} cells")
    print("=" * 80 + "\n")

    logger.info("=== Unified Visualization Complete ===")


if __name__ == "__main__":
    main()
