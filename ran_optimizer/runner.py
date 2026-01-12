"""
Unified runner for all RAN optimization algorithms.

This script runs all detection algorithms:
1. Overshooting detection (with environment-aware parameters)
2. Undershooting detection (with environment-aware parameters, band-aware)
3. No coverage gap detection
4. Low coverage detection (per-band)
5. Interference detection (high interference clusters)
6. PCI planning (confusions, collisions, blacklist suggestions)
7. PCI conflict detection (hull overlap-based same-PCI detection)
8. CA imbalance detection (carrier aggregation coverage gaps)
9. Crossed feeder detection

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
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

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
from ran_optimizer.recommendations.interference import (
    InterferenceDetector,
    InterferenceParams,
)
from ran_optimizer.recommendations.pci_planner import (
    PCIPlanner,
    PCIPlannerParams,
)
from ran_optimizer.recommendations.pci_conflict import (
    PCIConflictDetector,
    PCIConflictParams,
)
from ran_optimizer.recommendations.ca_imbalance import (
    CAImbalanceDetector,
    CAImbalanceParams,
)
from ran_optimizer.recommendations.crossed_feeder import (
    CrossedFeederDetector,
    CrossedFeederParams,
)
from ran_optimizer.visualization.enhanced_map import create_enhanced_map
from ran_optimizer.recommendations.daily_resolution import (
    generate_daily_resolution_recommendations,
    DailyResolutionConfig,
)

logger = get_logger(__name__)

AVAILABLE_ALGORITHMS = [
    'overshooting', 'undershooting', 'no_coverage', 'no_coverage_per_band', 'low_coverage',
    'interference', 'pci', 'pci_conflict', 'ca_imbalance', 'crossed_feeder'
]


def load_input_data(input_dir: Path) -> dict:
    """
    Load all required input data files.

    Expected files:
    - cell_coverage.csv: Grid measurements with RSRP, distance, traffic
    - cell_gis.csv: Cell GIS data (location, tilt, height, etc.)
    - cell_hulls.csv: Cell coverage hull polygons
    - cell_impacts.csv / relations.csv: Cell relations for PCI/crossed feeder

    Returns:
        Dictionary with 'grid_df', 'gis_df', 'hulls_gdf', 'relations_df' keys
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

    # Load relations/impacts data (optional - needed for PCI, crossed feeder)
    relations_df = None
    relations_path = input_dir / 'cell_impacts.csv'
    if not relations_path.exists():
        relations_path = input_dir / 'relations.csv'
    if relations_path.exists():
        logger.info("Loading relations data", path=str(relations_path))
        relations_df = pd.read_csv(relations_path)
        logger.info("Relations data loaded", rows=len(relations_df), columns=len(relations_df.columns))
    else:
        logger.warning("Relations data not found - PCI/crossed feeder detection will be skipped")

    return {
        'grid_df': grid_df,
        'gis_df': gis_df,
        'hulls_gdf': hulls_gdf,
        'relations_df': relations_df,
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

    # Save results (always write file, even if empty, to avoid stale data)
    output_file = output_dir / 'no_coverage_clusters.geojson'
    if len(results) > 0:
        results.to_file(output_file, driver='GeoJSON')
        logger.info("No coverage results saved", path=str(output_file), clusters=len(results))
    else:
        # Write empty GeoDataFrame to clear any stale data
        empty_gdf = gpd.GeoDataFrame(columns=['cluster_id', 'geometry'], crs="EPSG:4326")
        empty_gdf.to_file(output_file, driver='GeoJSON')
        logger.warning("No coverage gap clusters detected - wrote empty file")

    return results


def run_no_coverage_per_band(
    hulls_gdf: gpd.GeoDataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
    boundary_shapefile: Optional[str] = None,
    bands: Optional[List[str]] = None,
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Run no-coverage gap detection per band.

    For each band, finds areas where cells of that band don't provide coverage,
    even if other bands cover the area. Complements low coverage detection.

    Args:
        hulls_gdf: Cell convex hulls
        gis_df: Cell GIS data with band information
        output_dir: Output directory for GeoJSON files
        config_path: Path to coverage_gaps.json
        boundary_shapefile: Optional boundary to clip results
        bands: Optional list of bands to process (None = all bands)

    Returns:
        Dict mapping band to GeoDataFrame of no coverage clusters
    """
    logger.info("=" * 80)
    logger.info("Running NO COVERAGE PER BAND detection")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping no coverage per band detection")
        return {}

    if gis_df is None:
        logger.warning("GIS data not available - skipping no coverage per band detection")
        return {}

    params = CoverageGapParams.from_config(config_path=Path(config_path) if config_path else None)
    detector = CoverageGapDetector(params, boundary_shapefile=boundary_shapefile)
    results = detector.detect_per_band(hulls_gdf, gis_df, bands=bands)

    # Save results per band
    for band, band_gdf in results.items():
        output_file = output_dir / f'no_coverage_{band}.geojson'
        if len(band_gdf) > 0:
            band_gdf.to_file(output_file, driver='GeoJSON')
            total_area = band_gdf['area_km2'].sum() if 'area_km2' in band_gdf.columns else 0
            logger.info(
                "No coverage per band results saved",
                band=band,
                path=str(output_file),
                clusters=len(band_gdf),
                total_area_km2=round(total_area, 2)
            )
        else:
            # Write empty GeoDataFrame
            empty_gdf = gpd.GeoDataFrame(columns=['cluster_id', 'band', 'geometry'], crs="EPSG:4326")
            empty_gdf.to_file(output_file, driver='GeoJSON')
            logger.info("No coverage per band - no gaps found", band=band)

    # Also save combined file
    if results:
        combined_gdf = gpd.GeoDataFrame(pd.concat(results.values(), ignore_index=True), crs="EPSG:4326")
        combined_file = output_dir / 'no_coverage_per_band_combined.geojson'
        combined_gdf.to_file(combined_file, driver='GeoJSON')
        logger.info(
            "Combined no coverage per band saved",
            path=str(combined_file),
            total_clusters=len(combined_gdf),
            bands=list(results.keys())
        )

    return results


def run_low_coverage(
    hulls_gdf: gpd.GeoDataFrame,
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
    bands: Optional[List[int]] = None,
    boundary_shapefile: Optional[str] = None,
    env_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run low-coverage detection per band with environment-aware thresholds.

    Args:
        hulls_gdf: Cell convex hulls
        grid_df: Grid measurements
        gis_df: Cell GIS data
        output_dir: Output directory
        config_path: Path to coverage_gaps.json
        bands: Optional list of bands to process
        boundary_shapefile: Optional boundary to clip results
        env_df: Optional environment classification per cell (for environment-aware thresholds)

    Returns:
        Dict mapping band to GeoDataFrame of low coverage clusters
    """
    logger.info("=" * 80)
    logger.info("Running LOW COVERAGE detection")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping low coverage detection")
        return {}

    config_path_obj = Path(config_path) if config_path else None
    params = LowCoverageParams.from_config(config_path=config_path_obj)
    detector = LowCoverageDetector(params, boundary_shapefile=boundary_shapefile)
    results = detector.detect(
        hulls_gdf,
        grid_df,
        gis_df,
        bands=bands,
        env_df=env_df,
        config_path=config_path_obj,
    )

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


def run_interference(
    grid_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Run interference detection."""
    logger.info("=" * 80)
    logger.info("Running INTERFERENCE detection")
    logger.info("=" * 80)

    # Normalize column names for detector (expects lowercase)
    col_map = {}
    if 'Band' in grid_df.columns and 'band' not in grid_df.columns:
        col_map['Band'] = 'band'
    if 'cilac' in grid_df.columns and 'cell_name' not in grid_df.columns:
        col_map['cilac'] = 'cell_name'

    if col_map:
        grid_df = grid_df.rename(columns=col_map)
        logger.info("Normalized column names", mapping=col_map)

    params = InterferenceParams.from_config(config_path)
    detector = InterferenceDetector(params)
    results = detector.detect(grid_df, data_type='measured')

    # Save results
    output_file = output_dir / 'interference_clusters.geojson'
    if len(results) > 0:
        results.to_file(output_file, driver='GeoJSON')
        logger.info("Interference results saved", path=str(output_file), clusters=len(results))
    else:
        # Write empty file to avoid stale data
        empty_gdf = gpd.GeoDataFrame(columns=['cluster_id', 'geometry'], crs="EPSG:4326")
        empty_gdf.to_file(output_file, driver='GeoJSON')
        logger.warning("No interference clusters detected - wrote empty file")

    return results


def run_pci_planner(
    relations_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run PCI planning (confusions, collisions, blacklist)."""
    logger.info("=" * 80)
    logger.info("Running PCI PLANNER detection")
    logger.info("=" * 80)

    if relations_df is None:
        logger.warning("Relations data not available - skipping PCI detection")
        return {}

    # Normalize column names for detector
    # Expected: cell_name, to_cell_name, pci, to_pci, band, to_band, weight
    col_map = {}
    if 'cell_impact_name' in relations_df.columns and 'to_cell_name' not in relations_df.columns:
        col_map['cell_impact_name'] = 'to_cell_name'
    if 'cell_pci' in relations_df.columns and 'pci' not in relations_df.columns:
        col_map['cell_pci'] = 'pci'
    if 'cell_impact_pci' in relations_df.columns and 'to_pci' not in relations_df.columns:
        col_map['cell_impact_pci'] = 'to_pci'
    if 'cell_band' in relations_df.columns and 'band' not in relations_df.columns:
        col_map['cell_band'] = 'band'
    if 'cell_impact_band' in relations_df.columns and 'to_band' not in relations_df.columns:
        col_map['cell_impact_band'] = 'to_band'
    if 'relation_impact_data_perc' in relations_df.columns and 'weight' not in relations_df.columns:
        col_map['relation_impact_data_perc'] = 'weight'

    if col_map:
        relations_df = relations_df.rename(columns=col_map)
        logger.info("Normalized column names for PCI planner", mapping=col_map)

    # Check if we have the required columns after mapping
    required = ["cell_name", "to_cell_name", "pci", "to_pci", "band", "to_band", "weight"]
    missing = [c for c in required if c not in relations_df.columns]
    if missing:
        logger.warning(f"Missing required columns for PCI planner: {missing} - skipping")
        return {}

    params = PCIPlannerParams.from_config(config_path)
    planner = PCIPlanner(relations_df, params)

    # Detect confusions
    confusions = planner.detect_confusions()
    if len(confusions) > 0:
        output_file = output_dir / 'pci_confusions.csv'
        confusions.to_csv(output_file, index=False)
        logger.info("PCI confusions saved", path=str(output_file), count=len(confusions))

    # Detect collisions
    collisions = planner.detect_collisions()
    if len(collisions) > 0:
        # Add intra_site flag based on cell names (first 5 chars = site)
        collisions['site_a'] = collisions['cell_a'].str[:5]
        collisions['site_b'] = collisions['cell_b'].str[:5]
        collisions['intra_site'] = collisions['site_a'] == collisions['site_b']

        # Save full results
        output_file = output_dir / 'pci_collisions_all.csv'
        collisions.to_csv(output_file, index=False)
        logger.info("PCI collisions (all) saved", path=str(output_file), count=len(collisions))

        # Filter based on config: exact collisions always included, mod3 depends on include_mod3_inter_site
        if params.include_mod3_inter_site:
            # Include all exact + all mod3 (inter and intra site)
            filtered_collisions = collisions.copy()
            filter_desc = "all (exact + mod3)"
        else:
            # Include exact (all sites) + only intra-site mod3
            filtered_collisions = collisions[
                (collisions['conflict_type'] == 'exact') |
                ((collisions['conflict_type'] == 'mod3') & (collisions['intra_site'] == True))
            ].copy()
            filter_desc = "exact + intra-site mod3"
        output_file = output_dir / 'pci_collisions.csv'
        filtered_collisions.to_csv(output_file, index=False)
        logger.info(f"PCI collisions ({filter_desc}) saved", path=str(output_file), count=len(filtered_collisions))

    # Suggest blacklists
    blacklist_df, auto_apply = planner.suggest_blacklists()
    if len(blacklist_df) > 0:
        output_file = output_dir / 'pci_blacklist_suggestions.csv'
        blacklist_df.to_csv(output_file, index=False)
        logger.info("PCI blacklist suggestions saved", path=str(output_file), count=len(blacklist_df))

    return {
        'confusions': confusions,
        'collisions': collisions,
        'blacklist_suggestions': blacklist_df,
    }


def run_pci_conflict(
    hulls_gdf: gpd.GeoDataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run PCI conflict detection (hull overlap-based)."""
    logger.info("=" * 80)
    logger.info("Running PCI CONFLICT detection (hull overlap)")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping PCI conflict detection")
        return pd.DataFrame()

    # Enrich hulls with band and pci info from GIS if missing
    hulls_enriched = hulls_gdf.copy()

    if gis_df is not None:
        gis_lookup = gis_df.set_index('cell_name')

        if 'band' not in hulls_enriched.columns and 'band' in gis_df.columns:
            hulls_enriched['band'] = hulls_enriched['cell_name'].map(gis_lookup['band'].to_dict())
            logger.info("Enriched hulls with band info from GIS data")

        if 'pci' not in hulls_enriched.columns and 'pci' in gis_df.columns:
            hulls_enriched['pci'] = hulls_enriched['cell_name'].map(gis_lookup['pci'].to_dict())
            logger.info("Enriched hulls with PCI info from GIS data")

    # Drop rows without required info
    before_count = len(hulls_enriched)
    hulls_enriched = hulls_enriched.dropna(subset=['band', 'pci'])
    if len(hulls_enriched) < before_count:
        logger.warning(f"Dropped {before_count - len(hulls_enriched)} hulls without band/pci info")

    if len(hulls_enriched) == 0:
        logger.warning("No hulls with band and PCI info - skipping PCI conflict detection")
        return pd.DataFrame()

    try:
        params = PCIConflictParams.from_config(config_path)
        detector = PCIConflictDetector(params)
        results_list = detector.detect(hulls_enriched)

        # Convert list of dicts to DataFrame
        if results_list and len(results_list) > 0:
            results = pd.DataFrame(results_list)

            # Save all results (including mod3)
            output_file = output_dir / 'pci_conflicts_all.csv'
            results.to_csv(output_file, index=False)
            logger.info("PCI conflicts (all) saved", path=str(output_file), count=len(results))

            # Filter to exact/collision only for main output (exclude mod3)
            if 'conflict_type' in results.columns:
                exact_conflicts = results[results['conflict_type'] == 'collision'].copy()
                output_file = output_dir / 'pci_conflicts.csv'
                exact_conflicts.to_csv(output_file, index=False)
                logger.info("PCI conflicts (exact only) saved", path=str(output_file), count=len(exact_conflicts))
            else:
                output_file = output_dir / 'pci_conflicts.csv'
                results.to_csv(output_file, index=False)
                logger.info("PCI conflict results saved", path=str(output_file), count=len(results))
        else:
            results = pd.DataFrame()
            logger.info("No PCI conflicts detected")

        return results
    except Exception as e:
        logger.warning(f"PCI conflict detection failed: {e}")
        return pd.DataFrame()


def run_ca_imbalance(
    hulls_gdf: gpd.GeoDataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run CA imbalance detection."""
    logger.info("=" * 80)
    logger.info("Running CA IMBALANCE detection")
    logger.info("=" * 80)

    if hulls_gdf is None:
        logger.warning("Cell hulls not available - skipping CA imbalance detection")
        return pd.DataFrame()

    if config_path is None:
        config_path = 'config/ca_imbalance_params.json'

    # Check if config exists
    if not Path(config_path).exists():
        logger.warning(f"CA imbalance config not found: {config_path} - skipping")
        return pd.DataFrame()

    # Enrich hulls with band info from GIS if missing
    if 'band' not in hulls_gdf.columns and gis_df is not None and 'band' in gis_df.columns:
        logger.info("Enriching hulls with band info from GIS data")
        band_map = gis_df.set_index('cell_name')['band'].to_dict()
        hulls_gdf = hulls_gdf.copy()
        hulls_gdf['band'] = hulls_gdf['cell_name'].map(band_map)
        # Drop rows without band info
        before_count = len(hulls_gdf)
        hulls_gdf = hulls_gdf.dropna(subset=['band'])
        if len(hulls_gdf) < before_count:
            logger.warning(f"Dropped {before_count - len(hulls_gdf)} hulls without band info")

    try:
        params = CAImbalanceParams.from_config(config_path)
        detector = CAImbalanceDetector(params)
        results_list = detector.detect(hulls_gdf)

        # Convert list of dicts to DataFrame
        if results_list and len(results_list) > 0:
            results = pd.DataFrame(results_list)
            output_file = output_dir / 'ca_imbalance.csv'
            results.to_csv(output_file, index=False)
            logger.info("CA imbalance results saved", path=str(output_file), count=len(results))
        else:
            results = pd.DataFrame()
            logger.info("No CA imbalance issues detected")

        return results
    except Exception as e:
        logger.warning(f"CA imbalance detection failed: {e}")
        return pd.DataFrame()


def run_crossed_feeder(
    relations_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    output_dir: Path,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run crossed feeder detection."""
    logger.info("=" * 80)
    logger.info("Running CROSSED FEEDER detection")
    logger.info("=" * 80)

    if relations_df is None:
        logger.warning("Relations data not available - skipping crossed feeder detection")
        return pd.DataFrame()

    # Normalize column names for detector
    # Expected REL: cell_name, to_cell_name, distance, band, to_band, intra_site, intra_cell, weight
    col_map = {}
    if 'cell_impact_name' in relations_df.columns and 'to_cell_name' not in relations_df.columns:
        col_map['cell_impact_name'] = 'to_cell_name'
    if 'cell_band' in relations_df.columns and 'band' not in relations_df.columns:
        col_map['cell_band'] = 'band'
    if 'cell_impact_band' in relations_df.columns and 'to_band' not in relations_df.columns:
        col_map['cell_impact_band'] = 'to_band'
    if 'relation_impact_data_perc' in relations_df.columns and 'weight' not in relations_df.columns:
        col_map['relation_impact_data_perc'] = 'weight'

    if col_map:
        relations_df = relations_df.rename(columns=col_map)
        logger.info("Normalized column names for crossed feeder", mapping=col_map)

    # Derive intra_site from co_site if missing
    if 'intra_site' not in relations_df.columns and 'co_site' in relations_df.columns:
        relations_df['intra_site'] = relations_df['co_site'].map(lambda x: 'y' if x == 'Y' or x == 1 or x == True else 'n')
        logger.info("Derived intra_site from co_site column")
    elif 'intra_site' not in relations_df.columns:
        relations_df['intra_site'] = 'n'
        logger.info("Added default intra_site=n (no data available)")

    # Derive intra_cell from co_sectored if missing
    if 'intra_cell' not in relations_df.columns and 'co_sectored' in relations_df.columns:
        relations_df['intra_cell'] = relations_df['co_sectored'].map(lambda x: 'y' if x == 'Y' or x == 1 or x == True else 'n')
        logger.info("Derived intra_cell from co_sectored column")
    elif 'intra_cell' not in relations_df.columns:
        relations_df['intra_cell'] = 'n'
        logger.info("Added default intra_cell=n (no data available)")

    # Check required columns
    required = ["cell_name", "to_cell_name", "distance", "band", "to_band", "intra_site", "intra_cell", "weight"]
    missing = [c for c in required if c not in relations_df.columns]
    if missing:
        logger.warning(f"Missing required columns for crossed feeder: {missing} - skipping")
        return pd.DataFrame()

    params = CrossedFeederParams.from_config(config_path)
    detector = CrossedFeederDetector(params)
    results = detector.detect(relations_df, gis_df)

    # Results is a dict: cells, sites, swap_pairs, relation_details
    cells_df = results.get('cells', pd.DataFrame())
    sites_df = results.get('sites', pd.DataFrame())
    swap_pairs_df = results.get('swap_pairs', pd.DataFrame())

    # Save cell results
    if len(cells_df) > 0:
        output_file = output_dir / 'crossed_feeder_cells.csv'
        cells_df.to_csv(output_file, index=False)
        logger.info("Crossed feeder cells saved", path=str(output_file), count=len(cells_df))

    # Save site summary
    if len(sites_df) > 0:
        output_file = output_dir / 'crossed_feeder_sites.csv'
        sites_df.to_csv(output_file, index=False)
        logger.info("Crossed feeder sites saved", path=str(output_file), count=len(sites_df))

    # Save swap pairs (HIGH confidence detections)
    if len(swap_pairs_df) > 0:
        output_file = output_dir / 'crossed_feeder_swap_pairs.csv'
        swap_pairs_df.to_csv(output_file, index=False)
        logger.info("Crossed feeder swap pairs saved", path=str(output_file), count=len(swap_pairs_df))

    # Log summary by confidence level
    if len(cells_df) > 0 and 'confidence_level' in cells_df.columns:
        high_swap = len(cells_df[cells_df['confidence_level'] == 'HIGH_POTENTIAL_SWAP'])
        possible_swap = len(cells_df[cells_df['confidence_level'] == 'POSSIBLE_SWAP'])
        single_anomaly = len(cells_df[cells_df['confidence_level'] == 'SINGLE_ANOMALY'])
        repan = len(cells_df[cells_df['confidence_level'] == 'REPAN'])
        logger.info(
            "Crossed feeder detection summary",
            high_potential_swap=high_swap,
            possible_swap=possible_swap,
            single_anomaly=single_anomaly,
            repan=repan,
            swap_pairs=len(swap_pairs_df),
        )

    # Return flagged cells for summary
    if len(cells_df) > 0 and 'flagged' in cells_df.columns:
        flagged = cells_df[cells_df['flagged'] == True]
        return flagged
    else:
        logger.info("No crossed feeders detected")
        return pd.DataFrame()


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
    relations_df = data['relations_df']

    # Load or create environment classification
    env_output_path = output_dir / 'cell_environment.csv'
    env_df = load_or_create_cell_environments(gis_df, output_path=str(env_output_path))

    # Config paths
    overshooting_config = str(config_dir / 'overshooting_params.json') if config_dir else None
    undershooting_config = str(config_dir / 'undershooting_params.json') if config_dir else None
    coverage_config = str(config_dir / 'coverage_gaps.json') if config_dir else None
    interference_config = str(config_dir / 'interference_params.json') if config_dir else None
    pci_config = str(config_dir / 'pci_planner_params.json') if config_dir else None
    pci_conflict_config = str(config_dir / 'pci_conflict_params.json') if config_dir else None
    ca_imbalance_config = str(config_dir / 'ca_imbalance_params.json') if config_dir else None
    crossed_feeder_config = str(config_dir / 'crossed_feeder_params.json') if config_dir else None

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

    # Run no coverage per band
    if 'no_coverage_per_band' in algorithms:
        results['no_coverage_per_band'] = run_no_coverage_per_band(
            hulls_gdf, gis_df, output_dir,
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

    # Run interference detection
    if 'interference' in algorithms:
        results['interference'] = run_interference(
            grid_df, output_dir,
            config_path=interference_config,
        )

    # Run PCI planner
    if 'pci' in algorithms:
        pci_results = run_pci_planner(
            relations_df, output_dir,
            config_path=pci_config,
        )
        results['confusions'] = pci_results.get('confusions', pd.DataFrame())
        results['collisions'] = pci_results.get('collisions', pd.DataFrame())
        results['blacklist_suggestions'] = pci_results.get('blacklist_suggestions', pd.DataFrame())

    # Run PCI conflict detection (hull overlap-based)
    if 'pci_conflict' in algorithms:
        results['pci_conflicts'] = run_pci_conflict(
            hulls_gdf, gis_df, output_dir,
            config_path=pci_conflict_config,
        )

    # Run CA imbalance detection
    if 'ca_imbalance' in algorithms:
        results['ca_imbalance'] = run_ca_imbalance(
            hulls_gdf, gis_df, output_dir,
            config_path=ca_imbalance_config,
        )

    # Run crossed feeder detection
    if 'crossed_feeder' in algorithms:
        results['crossed_feeder'] = run_crossed_feeder(
            relations_df, gis_df, output_dir,
            config_path=crossed_feeder_config,
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
        # Only rename if the destination column doesn't already exist to avoid duplicates
        col_map = {}
        if 'cilac' in grid_df.columns and 'cell_name' not in grid_df.columns:
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
            no_coverage_per_band_gdfs=results.get('no_coverage_per_band'),
            low_coverage_gdfs=results.get('low_coverage'),
            overshooting_grid_df=overshooting_grid_df,
            undershooting_grid_df=undershooting_grid_df,
            cell_hulls_gdf=hulls_gdf,
            # New detector DataFrames
            pci_confusions_df=results.get('confusions'),
            pci_collisions_df=results.get('collisions'),
            pci_blacklist_df=results.get('blacklist_suggestions'),
            ca_imbalance_df=results.get('ca_imbalance'),
            crossed_feeder_df=results.get('crossed_feeder'),
            interference_gdf=results.get('interference'),
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
    # Count per-band results properly
    no_coverage_per_band = results.get('no_coverage_per_band', {})
    no_coverage_total = sum(len(gdf) for gdf in no_coverage_per_band.values()) if no_coverage_per_band else 0
    low_coverage_dict = results.get('low_coverage', {})
    low_coverage_total = sum(len(gdf) for gdf in low_coverage_dict.values()) if low_coverage_dict else 0

    logger.info("Summary",
                elapsed_seconds=elapsed.total_seconds(),
                overshooting_count=len(results.get('overshooting', [])),
                undershooting_count=len(results.get('undershooting', [])),
                no_coverage_clusters=no_coverage_total,
                no_coverage_bands=len(no_coverage_per_band),
                low_coverage_clusters=low_coverage_total,
                low_coverage_bands=len(low_coverage_dict),
                interference_clusters=len(results.get('interference', [])),
                pci_confusions=len(results.get('confusions', [])),
                pci_collisions=len(results.get('collisions', [])),
                pci_conflicts=len(results.get('pci_conflicts', [])),
                blacklist_suggestions=len(results.get('blacklist_suggestions', [])),
                ca_imbalance_count=len(results.get('ca_imbalance', [])),
                crossed_feeders=len(results.get('crossed_feeder', [])))

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
