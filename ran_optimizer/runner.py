"""
Unified runner for all RAN optimization algorithms.

This script runs all four detection algorithms:
1. Overshooting detection (with environment-aware parameters)
2. Undershooting detection (with environment-aware parameters, band-aware)
3. No coverage gap detection
4. Low coverage detection (per-band)

Usage:
    python -m ran_optimizer.runner --input-dir data/vf-ie/input-data --output-dir data/vf-ie/output-data

    # Run specific algorithms only
    python -m ran_optimizer.runner --algorithms overshooting undershooting

    # Use environment-aware detection (default)
    python -m ran_optimizer.runner --environment-aware

    # Use standard detection (single parameter set)
    python -m ran_optimizer.runner --no-environment-aware
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd
import geopandas as gpd
from shapely import wkt

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.core import load_or_create_cell_environments

# Algorithm imports
from ran_optimizer.recommendations.overshooters import (
    OvershooterDetector,
    OvershooterParams,
)
from ran_optimizer.recommendations.undershooters import (
    UndershooterDetector,
    UndershooterParams,
    detect_undershooting_with_environment_awareness,
    compare_undershooting_detection_approaches,
)
from ran_optimizer.recommendations.coverage_gaps import (
    CoverageGapDetector,
    CoverageGapParams,
    LowCoverageDetector,
    LowCoverageParams,
)
from ran_optimizer.visualization.enhanced_map import create_enhanced_map
from ran_optimizer.recommendations.daily_resolution import (
    generate_daily_resolution_recommendations,
    DailyResolutionConfig,
)

logger = get_logger(__name__)

AVAILABLE_ALGORITHMS = ['overshooting', 'undershooting', 'no_coverage', 'low_coverage']


def load_input_data(input_dir: Path) -> dict:
    """
    Load all required input data files.

    Expected files:
    - cell_coverage.csv: Grid measurements with RSRP, distance, traffic
    - cork-gis.csv: Cell GIS data (location, tilt, height, etc.)
    - cell_hulls.csv: Cell coverage hull polygons

    Returns:
        Dictionary with 'grid_df', 'gis_df', 'hulls_gdf' keys
    """
    logger.info("Loading input data", input_dir=str(input_dir))

    # Load grid data
    grid_path = input_dir / 'cell_coverage.csv'
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid data not found: {grid_path}")

    logger.info("Loading grid data", path=str(grid_path))
    grid_df = pd.read_csv(grid_path, low_memory=False)
    logger.info("Grid data loaded", rows=len(grid_df), columns=len(grid_df.columns))

    # Load GIS data (prefer cell_gis.csv with canonical schema)
    gis_path = input_dir / 'cell_gis.csv'
    if not gis_path.exists():
        gis_path = input_dir / 'gis.csv'
    if not gis_path.exists():
        raise FileNotFoundError(f"GIS data not found in {input_dir}")

    logger.info("Loading GIS data", path=str(gis_path))
    gis_df = pd.read_csv(gis_path)
    logger.info("GIS data loaded", rows=len(gis_df), columns=len(gis_df.columns))

    # Load hulls data (optional - only needed for coverage gaps)
    hulls_gdf = None
    hulls_path = input_dir / 'cell_hulls.csv'
    if hulls_path.exists():
        logger.info("Loading cell hulls", path=str(hulls_path))
        hulls_df = pd.read_csv(hulls_path)
        hulls_df['geometry'] = hulls_df['geometry'].apply(wkt.loads)
        hulls_gdf = gpd.GeoDataFrame(hulls_df, geometry='geometry', crs='EPSG:4326')
        logger.info("Cell hulls loaded", rows=len(hulls_gdf))
    else:
        logger.warning("Cell hulls not found - coverage gap detection will be skipped")

    return {
        'grid_df': grid_df,
        'gis_df': gis_df,
        'hulls_gdf': hulls_gdf,
    }


def run_overshooting(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    env_df: pd.DataFrame,
    output_dir: Path,
    environment_aware: bool = True,
    config_path: Optional[str] = None,
    return_grids: bool = False,
) -> pd.DataFrame:
    """
    Run overshooting detection.

    Args:
        grid_df: Grid measurements DataFrame
        gis_df: Cell GIS data DataFrame
        env_df: Environment classification DataFrame
        output_dir: Output directory
        environment_aware: Use environment-aware parameters
        config_path: Path to config file
        return_grids: If True, returns tuple of (results, overshooting_grids)

    Returns:
        DataFrame with overshooting results, or tuple if return_grids=True
    """
    logger.info("=" * 80)
    logger.info("Running OVERSHOOTING detection", environment_aware=environment_aware)
    logger.info("=" * 80)

    overshooting_grids = None

    if environment_aware:
        # Load environment-specific parameters
        env_params = {}
        for env in ['urban', 'suburban', 'rural']:
            env_params[env] = OvershooterParams.from_config(
                config_path=config_path, environment=env
            )

        # Use detector with environment-aware parameters
        detector = OvershooterDetector(env_params['suburban'])
        if return_grids:
            results, overshooting_grids = detector.detect_with_environments(
                grid_df, gis_df, env_df, env_params, return_grids=True
            )
        else:
            results = detector.detect_with_environments(
                grid_df, gis_df, env_df, env_params, return_grids=False
            )
        output_file = output_dir / 'overshooting_cells.csv'
    else:
        params = OvershooterParams.from_config(config_path=config_path)
        detector = OvershooterDetector(params)

        if return_grids:
            results, overshooting_grids = detector.detect_with_grids(grid_df, gis_df)
        else:
            results = detector.detect(grid_df, gis_df)

        # Add environment info
        if len(results) > 0:
            results = results.merge(
                env_df[['cell_name', 'environment', 'intersite_distance_km']],
                on='cell_name',
                how='left'
            )
        output_file = output_dir / 'overshooting_cells_standard.csv'

    # Save results
    if len(results) > 0:
        results.to_csv(output_file, index=False)
        logger.info("Overshooting results saved", path=str(output_file), count=len(results))
    else:
        logger.warning("No overshooting cells detected")

    if return_grids:
        return results, overshooting_grids
    return results


def run_undershooting(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    env_df: pd.DataFrame,
    output_dir: Path,
    environment_aware: bool = True,
    config_path: Optional[str] = None,
    return_grids: bool = False,
) -> pd.DataFrame:
    """
    Run undershooting detection.

    Args:
        grid_df: Grid measurements DataFrame
        gis_df: Cell GIS data DataFrame
        env_df: Environment classification DataFrame
        output_dir: Output directory
        environment_aware: Use environment-aware parameters
        config_path: Path to config file
        return_grids: If True, returns tuple of (results, interference_grids)

    Returns:
        DataFrame with undershooting results, or tuple if return_grids=True
    """
    logger.info("=" * 80)
    logger.info("Running UNDERSHOOTING detection", environment_aware=environment_aware)
    logger.info("=" * 80)

    interference_grids = None

    if environment_aware:
        results = detect_undershooting_with_environment_awareness(
            grid_df, gis_df, env_df, config_path=config_path
        )
        output_file = output_dir / 'undershooting_cells.csv'

        # For environment-aware, run additional pass to get grids if needed
        if return_grids and len(results) > 0:
            params = UndershooterParams.from_config(config_path=config_path)
            detector = UndershooterDetector(params)
            _, interference_grids = detector.detect_with_grids(grid_df, gis_df)
    else:
        params = UndershooterParams.from_config(config_path=config_path)
        detector = UndershooterDetector(params)

        if return_grids:
            results, interference_grids = detector.detect_with_grids(grid_df, gis_df)
        else:
            results = detector.detect(grid_df, gis_df)

        # Add environment info
        if len(results) > 0:
            results = results.merge(
                env_df[['cell_name', 'environment', 'intersite_distance_km']],
                on='cell_name',
                how='left'
            )
        output_file = output_dir / 'undershooting_cells_standard.csv'

    # Save results
    if len(results) > 0:
        results.to_csv(output_file, index=False)
        logger.info("Undershooting results saved", path=str(output_file), count=len(results))
    else:
        logger.warning("No undershooting cells detected")

    if return_grids:
        return results, interference_grids
    return results


def run_no_coverage(
    hulls_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
    boundary_shapefile: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Run no-coverage gap detection."""
    logger.info("=" * 80)
    logger.info("Running NO COVERAGE detection")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping no coverage detection")
        return gpd.GeoDataFrame()

    params = CoverageGapParams.from_config(config_path=Path(config_path) if config_path else None)
    detector = CoverageGapDetector(params, boundary_shapefile=boundary_shapefile)
    results = detector.detect(hulls_gdf)

    # Save results
    if len(results) > 0:
        output_file = output_dir / 'no_coverage_clusters.geojson'
        results.to_file(output_file, driver='GeoJSON')
        logger.info("No coverage results saved", path=str(output_file), clusters=len(results))
    else:
        logger.warning("No coverage gap clusters detected")

    return results


def run_low_coverage(
    hulls_gdf: gpd.GeoDataFrame,
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    boundary_shapefile: Optional[str] = None,
) -> dict:
    """Run low-coverage detection per band."""
    logger.info("=" * 80)
    logger.info("Running LOW COVERAGE detection")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping low coverage detection")
        return {}

    params = LowCoverageParams.from_config(config_path=Path(config_path) if config_path else None)
    detector = LowCoverageDetector(params, boundary_shapefile=boundary_shapefile)
    results = detector.detect(hulls_gdf, grid_df, gis_df, bands=bands)

    # Combine all bands into single GeoDataFrame
    all_clusters = []
    for band, gdf in results.items():
        if len(gdf) > 0:
            gdf = gdf.copy()
            gdf['band'] = band  # Ensure band column is set
            all_clusters.append(gdf)
            logger.info("Low coverage found", band=band, clusters=len(gdf))

    if all_clusters:
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_clusters, ignore_index=True))
        output_file = output_dir / 'low_coverage.geojson'
        combined_gdf.to_file(output_file, driver='GeoJSON')
        logger.info("Low coverage results saved", path=str(output_file), total_clusters=len(combined_gdf))
    else:
        logger.warning("No low coverage clusters detected")

    return results


def run_all(
    input_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    environment_aware: bool = True,
    config_dir: Optional[Path] = None,
) -> dict:
    """
    Run all specified algorithms.

    Args:
        input_dir: Directory containing input data files
        output_dir: Directory for output files
        algorithms: List of algorithms to run
        environment_aware: Use environment-aware parameters
        config_dir: Directory containing config files

    Returns:
        Dictionary with results from each algorithm
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("RAN OPTIMIZER - UNIFIED RUNNER")
    logger.info("=" * 80)
    logger.info("Configuration",
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                algorithms=algorithms,
                environment_aware=environment_aware)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    data = load_input_data(input_dir)
    grid_df = data['grid_df']
    gis_df = data['gis_df']
    hulls_gdf = data['hulls_gdf']

    # Load or create environment classification
    env_output_path = output_dir / 'cell_environment.csv'
    env_df = load_or_create_cell_environments(gis_df, output_path=str(env_output_path))

    # Config paths
    overshooting_config = str(config_dir / 'overshooting_params.json') if config_dir else None
    undershooting_config = str(config_dir / 'undershooting_params.json') if config_dir else None
    coverage_config = str(config_dir / 'coverage_gaps.json') if config_dir else None

    results = {}
    overshooting_grids = None
    interference_grids = None

    # Run overshooting
    if 'overshooting' in algorithms:
        overshooting_result = run_overshooting(
            grid_df, gis_df, env_df, output_dir,
            environment_aware=environment_aware,
            config_path=overshooting_config,
            return_grids=True,
        )
        results['overshooting'] = overshooting_result[0]
        overshooting_grids = overshooting_result[1]

    # Run undershooting
    if 'undershooting' in algorithms:
        undershooting_result = run_undershooting(
            grid_df, gis_df, env_df, output_dir,
            environment_aware=environment_aware,
            config_path=undershooting_config,
            return_grids=True,
        )
        results['undershooting'] = undershooting_result[0]
        interference_grids = undershooting_result[1]

    # Check for boundary shapefile
    boundary_shapefile = input_dir / 'county_bounds' / 'bounds.shp'
    boundary_path = str(boundary_shapefile) if boundary_shapefile.exists() else None
    if boundary_path:
        logger.info("Using boundary shapefile for clipping", path=boundary_path)

    # Run no coverage
    if 'no_coverage' in algorithms:
        results['no_coverage'] = run_no_coverage(
            hulls_gdf, output_dir,
            config_path=coverage_config,
            boundary_shapefile=boundary_path,
        )

    # Run low coverage
    if 'low_coverage' in algorithms:
        results['low_coverage'] = run_low_coverage(
            hulls_gdf, grid_df, gis_df, output_dir,
            config_path=coverage_config,
            boundary_shapefile=boundary_path,
        )

    # Generate daily resolution recommendations
    if ('overshooting' in algorithms or 'undershooting' in algorithms):
        logger.info("=" * 80)
        logger.info("Generating daily resolution recommendations")
        logger.info("=" * 80)

        try:
            daily_recommendations = generate_daily_resolution_recommendations(
                overshooting_df=results.get('overshooting'),
                undershooting_df=results.get('undershooting'),
                grid_df=grid_df,
                gis_df=gis_df,
                overshooting_grids_df=overshooting_grids,
                interference_grids_df=interference_grids,
                output_dir=output_dir,
            )
            results['daily_recommendations'] = daily_recommendations
            logger.info(
                "Daily resolution recommendations generated",
                total=len(daily_recommendations) if daily_recommendations is not None else 0
            )
        except Exception as e:
            logger.warning("Failed to generate daily recommendations", error=str(e))

    # Generate visualization
    logger.info("=" * 80)
    logger.info("Generating enhanced visualization")
    logger.info("=" * 80)

    try:
        map_output = output_dir / 'maps' / 'enhanced_dashboard.html'

        # Prepare grid data for lazy loading - filter to just overshooting/undershooting cells
        overshooting_grid_df = None
        undershooting_grid_df = None

        # Determine the column name mappings for grid_df
        col_map = {}
        if 'cilac' in grid_df.columns:
            col_map['cilac'] = 'cell_name'
        if 'grid' in grid_df.columns and 'geohash7' not in grid_df.columns:
            col_map['grid'] = 'geohash7'
        if 'Latitude' in grid_df.columns and 'latitude' not in grid_df.columns:
            col_map['Latitude'] = 'latitude'
        if 'Longitude' in grid_df.columns and 'longitude' not in grid_df.columns:
            col_map['Longitude'] = 'longitude'

        grid_cell_name_col = 'cell_name' if 'cell_name' in grid_df.columns else 'cilac'

        # Filter grid data for overshooting cells
        if results.get('overshooting') is not None and len(results.get('overshooting')) > 0:
            # Convert cell_names to same type as grid_df for matching
            over_cell_names_raw = results['overshooting']['cell_name'].unique()
            # Convert to same dtype as grid_df cell_name column
            grid_dtype = grid_df[grid_cell_name_col].dtype
            if pd.api.types.is_integer_dtype(grid_dtype):
                # Convert to numeric, handling the numpy array
                converted = pd.to_numeric(pd.Series(over_cell_names_raw), errors='coerce').dropna().astype(int)
                over_cell_names = set(converted.tolist())
            else:
                over_cell_names = set(str(x) for x in over_cell_names_raw)
            overshooting_grid_df = grid_df[grid_df[grid_cell_name_col].isin(over_cell_names)].copy()
            # Rename columns to standard names for downstream processing
            if col_map:
                overshooting_grid_df = overshooting_grid_df.rename(columns=col_map)
            # Mark which grids are actually overshooting using the overshooting_grids data
            overshooting_grid_df['is_overshooting'] = False
            if overshooting_grids is not None and len(overshooting_grids) > 0:
                # Create a set of (cell_name, geohash7) tuples for fast lookup
                grid_geohash_col = 'geohash7' if 'geohash7' in overshooting_grid_df.columns else 'grid'
                over_grids_geohash_col = 'geohash7' if 'geohash7' in overshooting_grids.columns else 'grid'
                flagged_grids = set(zip(
                    overshooting_grids['cell_name'].astype(str),
                    overshooting_grids[over_grids_geohash_col].astype(str)
                ))
                # Mark grids that are in the flagged set
                overshooting_grid_df['is_overshooting'] = list(zip(
                    overshooting_grid_df['cell_name'].astype(str),
                    overshooting_grid_df[grid_geohash_col].astype(str)
                ))
                overshooting_grid_df['is_overshooting'] = overshooting_grid_df['is_overshooting'].isin(flagged_grids)
                flagged_count = overshooting_grid_df['is_overshooting'].sum()
                logger.info("Marked overshooting grids", flagged=flagged_count, total=len(overshooting_grid_df))
            logger.info("Prepared overshooting grid data", cells=len(over_cell_names), grids=len(overshooting_grid_df))

        # Filter grid data for undershooting cells
        if results.get('undershooting') is not None and len(results.get('undershooting')) > 0:
            # Convert cell_names to same type as grid_df for matching
            under_cell_names_raw = results['undershooting']['cell_name'].unique()
            # Convert to same dtype as grid_df cell_name column
            if pd.api.types.is_integer_dtype(grid_dtype):
                # Convert to numeric, handling the numpy array
                converted = pd.to_numeric(pd.Series(under_cell_names_raw), errors='coerce').dropna().astype(int)
                under_cell_names = set(converted.tolist())
            else:
                under_cell_names = set(str(x) for x in under_cell_names_raw)
            undershooting_grid_df = grid_df[grid_df[grid_cell_name_col].isin(under_cell_names)].copy()
            # Rename columns to standard names for downstream processing
            if col_map:
                undershooting_grid_df = undershooting_grid_df.rename(columns=col_map)
            # Mark which grids are high interference using the interference_grids data
            undershooting_grid_df['is_interfering'] = False
            if interference_grids is not None and len(interference_grids) > 0:
                # Create a set of (cell_name, geohash7) tuples for fast lookup
                grid_geohash_col = 'geohash7' if 'geohash7' in undershooting_grid_df.columns else 'grid'
                inter_grids_geohash_col = 'geohash7' if 'geohash7' in interference_grids.columns else 'grid'
                flagged_grids = set(zip(
                    interference_grids['cell_name'].astype(str),
                    interference_grids[inter_grids_geohash_col].astype(str)
                ))
                # Mark grids that are in the flagged set
                undershooting_grid_df['is_interfering'] = list(zip(
                    undershooting_grid_df['cell_name'].astype(str),
                    undershooting_grid_df[grid_geohash_col].astype(str)
                ))
                undershooting_grid_df['is_interfering'] = undershooting_grid_df['is_interfering'].isin(flagged_grids)
                flagged_count = undershooting_grid_df['is_interfering'].sum()
                logger.info("Marked interfering grids", flagged=flagged_count, total=len(undershooting_grid_df))
            logger.info("Prepared undershooting grid data", cells=len(under_cell_names), grids=len(undershooting_grid_df))

        create_enhanced_map(
            overshooting_df=results.get('overshooting'),
            undershooting_df=results.get('undershooting'),
            gis_df=gis_df,
            no_coverage_gdf=results.get('no_coverage'),
            low_coverage_gdfs=results.get('low_coverage'),
            overshooting_grid_df=overshooting_grid_df,
            undershooting_grid_df=undershooting_grid_df,
            cell_hulls_gdf=hulls_gdf,
            output_file=map_output,
            title="RAN Optimizer - Network Issues Dashboard",
        )
        logger.info("Visualization generated", path=str(map_output))
    except Exception as e:
        logger.warning("Failed to generate visualization", error=str(e))

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info("Summary",
                elapsed_seconds=elapsed.total_seconds(),
                overshooting_count=len(results.get('overshooting', [])),
                undershooting_count=len(results.get('undershooting', [])),
                no_coverage_clusters=len(results.get('no_coverage', [])),
                low_coverage_bands=len(results.get('low_coverage', {})))

    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='RAN Optimizer - Unified detection runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all algorithms with environment-aware parameters
  python -m ran_optimizer.runner --input-dir data/vf-ie/input-data --output-dir data/vf-ie/output-data

  # Run specific algorithms only
  python -m ran_optimizer.runner --algorithms overshooting undershooting

  # Use standard detection (single parameter set)
  python -m ran_optimizer.runner --no-environment-aware
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/vf-ie/input-data'),
        help='Directory containing input data files (default: data/vf-ie/input-data)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/vf-ie/output-data'),
        help='Directory for output files (default: data/vf-ie/output-data)'
    )

    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('config'),
        help='Directory containing config JSON files (default: config)'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=AVAILABLE_ALGORITHMS,
        default=AVAILABLE_ALGORITHMS,
        help=f'Algorithms to run (default: all). Choices: {AVAILABLE_ALGORITHMS}'
    )

    parser.add_argument(
        '--environment-aware',
        action='store_true',
        default=True,
        help='Use environment-aware parameters (default: True)'
    )

    parser.add_argument(
        '--no-environment-aware',
        action='store_true',
        help='Use standard parameters (single parameter set for all environments)'
    )

    args = parser.parse_args()

    # Handle environment-aware flag
    environment_aware = not args.no_environment_aware

    try:
        run_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            algorithms=args.algorithms,
            environment_aware=environment_aware,
            config_dir=args.config_dir,
        )
        return 0
    except Exception as e:
        logger.error("Execution failed", error=str(e), exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
