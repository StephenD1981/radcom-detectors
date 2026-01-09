"""
Production pipeline runner for RAN Optimizer.

This module provides a configuration-driven pipeline runner that:
- Loads configuration from JSON
- Uses the DataManager abstraction for data sources
- Runs all 8 detectors based on configuration
- Writes outputs in standardized formats

Usage:
    python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json

    # Run specific detectors only
    python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json --detectors overshooters undershooters

    # Dry run (validate config only)
    python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.pipeline_config import PipelineConfig, load_pipeline_config
from ran_optimizer.data.sources import DataManager, create_data_source
from ran_optimizer.utils.exceptions import DataLoadError

logger = get_logger(__name__)


class PipelineRunner:
    """
    Production pipeline runner for all RAN optimization detectors.

    Attributes:
        config: Pipeline configuration
        data_manager: Data source manager
        results: Dictionary of detector results
    """

    DETECTOR_ORDER = [
        'overshooters',
        'undershooters',
        'low_coverage',
        'no_coverage',
        'interference',
        'ca_imbalance',
        'crossed_feeders',
        'pci',
    ]

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline runner.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.data_manager = DataManager(config)
        self.results: Dict[str, Any] = {}

        logger.info(
            "pipeline_runner_initialized",
            operator=config.operator,
            region=config.region,
            enabled_detectors=config.detectors.get_enabled_detectors()
        )

    def run(self, detectors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the pipeline with specified or all enabled detectors.

        Args:
            detectors: List of detector names to run. If None, run all enabled.

        Returns:
            Dictionary with results from each detector
        """
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("RAN OPTIMIZER - PRODUCTION PIPELINE")
        logger.info("=" * 80)

        # Determine which detectors to run
        if detectors is None:
            detectors_to_run = self.config.detectors.get_enabled_detectors()
        else:
            # Validate requested detectors
            enabled = set(self.config.detectors.get_enabled_detectors())
            detectors_to_run = [d for d in detectors if d in enabled]
            skipped = [d for d in detectors if d not in enabled]
            if skipped:
                logger.warning("detectors_disabled_in_config", skipped=skipped)

        logger.info("detectors_to_run", detectors=detectors_to_run)

        # Run detectors in order
        for detector_name in self.DETECTOR_ORDER:
            if detector_name not in detectors_to_run:
                continue

            logger.info("=" * 60)
            logger.info(f"Running {detector_name.upper()} detector")
            logger.info("=" * 60)

            try:
                result = self._run_detector(detector_name)
                self.results[detector_name] = result
                self._log_detector_result(detector_name, result)
            except Exception as e:
                logger.error(
                    f"{detector_name}_failed",
                    error=str(e),
                    exc_info=True
                )
                self.results[detector_name] = None

        # Write outputs
        self._write_outputs()

        # Summary
        elapsed = datetime.now() - start_time
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        self._log_summary(elapsed)

        return self.results

    def _run_detector(self, detector_name: str) -> Any:
        """Run a single detector and return results."""

        if detector_name == 'overshooters':
            return self._run_overshooters()
        elif detector_name == 'undershooters':
            return self._run_undershooters()
        elif detector_name == 'low_coverage':
            return self._run_low_coverage()
        elif detector_name == 'no_coverage':
            return self._run_no_coverage()
        elif detector_name == 'interference':
            return self._run_interference()
        elif detector_name == 'ca_imbalance':
            return self._run_ca_imbalance()
        elif detector_name == 'crossed_feeders':
            return self._run_crossed_feeders()
        elif detector_name == 'pci':
            return self._run_pci()
        else:
            raise ValueError(f"Unknown detector: {detector_name}")

    def _run_overshooters(self) -> pd.DataFrame:
        """Run overshooting detector."""
        from ran_optimizer.recommendations.overshooters import (
            OvershooterDetector, OvershooterParams
        )

        coverage_df = self.data_manager.get_coverage_for_detector('overshooters')
        gis_df = self.data_manager.source.load_gis()

        config = self.config.detectors.overshooters
        params = OvershooterParams(
            edge_traffic_percent=config.edge_traffic_percent,
            min_cell_distance=config.min_cell_distance,
            interference_threshold_db=config.interference_threshold_db,
            min_cell_count_in_grid=config.min_cell_count_in_grid,
            max_percentage_grid_events=config.max_percentage_grid_events,
            min_relative_reach=config.min_relative_reach,
            rsrp_degradation_db=config.rsrp_degradation_db,
            min_overshooting_grids=config.min_overshooting_grids,
            percentage_overshooting_grids=config.percentage_overshooting_grids,
        )

        detector = OvershooterDetector(params)
        results = detector.detect(coverage_df, gis_df)

        return results

    def _run_undershooters(self) -> pd.DataFrame:
        """Run undershooting detector."""
        from ran_optimizer.recommendations.undershooters import (
            UndershooterDetector, UndershooterParams
        )

        coverage_df = self.data_manager.get_coverage_for_detector('undershooters')
        gis_df = self.data_manager.source.load_gis()

        config = self.config.detectors.undershooters
        params = UndershooterParams(
            max_cell_distance=config.max_cell_distance,
            min_cell_event_count=config.min_cell_event_count,
            max_interference_percentage=config.max_interference_percentage,
            interference_threshold_db=config.interference_threshold_db,
            max_cell_grid_count=config.max_cell_grid_count,
        )

        detector = UndershooterDetector(params)
        results = detector.detect(coverage_df, gis_df)

        return results

    def _run_low_coverage(self) -> Dict[str, Any]:
        """Run low coverage detector (per-band)."""
        from ran_optimizer.recommendations.coverage_gaps import (
            LowCoverageDetector, LowCoverageParams
        )

        hulls_gdf = self.data_manager.source.load_hulls()
        coverage_df = self.data_manager.source.load_coverage()
        gis_df = self.data_manager.source.load_gis()

        config = self.config.detectors.low_coverage
        params = LowCoverageParams(
            rsrp_threshold_dbm=config.rsrp_threshold_dbm,
            k_ring_steps=config.k_ring_steps,
            min_missing_neighbors=config.min_missing_neighbors,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
        )

        detector = LowCoverageDetector(params)
        results = detector.detect(hulls_gdf, coverage_df, gis_df)

        return results

    def _run_no_coverage(self) -> Any:
        """Run no coverage (gap) detector."""
        from ran_optimizer.recommendations.coverage_gaps import (
            CoverageGapDetector, CoverageGapParams
        )

        hulls_gdf = self.data_manager.source.load_hulls()

        config = self.config.detectors.no_coverage
        params = CoverageGapParams(
            cell_cluster_eps_km=config.cell_cluster_eps_km,
            cell_cluster_min_samples=config.cell_cluster_min_samples,
            k_ring_steps=config.k_ring_steps,
            min_missing_neighbors=config.min_missing_neighbors,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
        )

        detector = CoverageGapDetector(params)
        results = detector.detect(hulls_gdf)

        return results

    def _run_interference(self) -> Any:
        """Run interference detector (per-band)."""
        from ran_optimizer.recommendations.interference import (
            InterferenceDetector, InterferenceParams
        )

        coverage_df = self.data_manager.source.load_coverage()

        config = self.config.detectors.interference
        params = InterferenceParams(
            min_filtered_cells_per_grid=config.min_filtered_cells_per_grid,
            dominant_perc_grid_events=config.dominant_perc_grid_events,
            dominance_diff=config.dominance_diff,
            max_rsrp_diff=config.max_rsrp_diff,
            k=config.k_ring,
            perc_interference=config.perc_interference,
        )

        detector = InterferenceDetector(params)
        interference_cells, grid_data = detector.detect(coverage_df)

        # Convert tuple to dict for output writer
        return {
            'interference_cells': interference_cells,
            'grid_data': grid_data,
        }

    def _run_ca_imbalance(self) -> Any:
        """Run CA imbalance detector."""
        from ran_optimizer.recommendations.ca_imbalance import (
            CAImbalanceDetector, CAImbalanceParams, CAPairConfig
        )

        hulls_gdf = self.data_manager.get_hulls_with_gis()

        config = self.config.detectors.ca_imbalance

        # Build CA pair configs from pipeline config
        ca_pairs = []
        for pair in config.band_pairs:
            ca_pairs.append(CAPairConfig(
                name=pair.get('name', 'L800-L1800'),
                coverage_band=pair.get('coverage_band', 'L800'),
                capacity_band=pair.get('capacity_band', 'L1800'),
                coverage_threshold=config.coverage_threshold,
            ))

        params = CAImbalanceParams(ca_pairs=ca_pairs)
        detector = CAImbalanceDetector(params)
        results = detector.detect(hulls_gdf)

        return results

    def _run_crossed_feeders(self) -> Dict[str, pd.DataFrame]:
        """Run crossed feeder detector."""
        from ran_optimizer.recommendations.crossed_feeder import (
            CrossedFeederDetector, CrossedFeederParams
        )

        impacts_df = self.data_manager.get_impacts_for_detector('crossed_feeders')
        gis_df = self.data_manager.source.load_gis()

        config = self.config.detectors.crossed_feeders
        params = CrossedFeederParams(
            max_radius_m=config.max_radius_m,
            min_distance_m=config.min_distance_m,
            hbw_cap_deg=config.hbw_cap_deg,
            percentile=config.percentile,
        )

        detector = CrossedFeederDetector(params)
        results = detector.detect(impacts_df, gis_df)

        return results

    def _run_pci(self) -> Dict[str, pd.DataFrame]:
        """Run PCI conflict detector (hull-based + relation-based)."""
        from ran_optimizer.recommendations.pci_conflict import (
            PCIConflictDetector, PCIConflictParams
        )
        from ran_optimizer.recommendations.pci_planner import (
            PCIPlanner, PCIPlannerParams
        )

        results = {}

        # 1. Hull-based detection (geometric overlap)
        try:
            hulls_gdf = self.data_manager.get_hulls_with_gis()
            params = PCIConflictParams()
            detector = PCIConflictDetector(params)
            conflicts = detector.detect(hulls_gdf)
            results['hull_conflicts'] = pd.DataFrame(conflicts)
        except Exception as e:
            logger.warning(f"Hull-based PCI detection failed: {e}")
            results['hull_conflicts'] = pd.DataFrame()

        # 2. Relation-based detection (using cell_impacts)
        try:
            impacts_df = self.data_manager.get_impacts_for_detector(
                'pci', filter_na_pci=True
            )

            if impacts_df is not None and len(impacts_df) > 0:
                planner_params = PCIPlannerParams(
                    couple_cosectors=False,
                    max_collision_radius_m=30000.0,
                    two_hop_factor=0.25,
                )
                planner = PCIPlanner(impacts_df, planner_params)

                # Detect confusions and collisions
                results['confusion'] = planner.detect_confusions()
                results['collision'] = planner.detect_collisions()

                # Get blacklist suggestions
                blacklist_df, _ = planner.suggest_blacklists()
                results['blacklist'] = blacklist_df
            else:
                logger.warning("No impacts data available for relation-based PCI detection")
                results['confusion'] = pd.DataFrame()
                results['collision'] = pd.DataFrame()
                results['blacklist'] = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Relation-based PCI detection failed: {e}")
            results['confusion'] = pd.DataFrame()
            results['collision'] = pd.DataFrame()
            results['blacklist'] = pd.DataFrame()

        return results

    def _log_detector_result(self, detector_name: str, result: Any) -> None:
        """Log summary of detector result."""
        if result is None:
            logger.warning(f"{detector_name}_no_results")
            return

        if isinstance(result, pd.DataFrame):
            logger.info(f"{detector_name}_complete", rows=len(result))
        elif isinstance(result, dict):
            # Convert keys to strings for logging (band keys may be numeric)
            counts = {str(k): len(v) if hasattr(v, '__len__') else 'N/A' for k, v in result.items()}
            logger.info(f"{detector_name}_complete", **counts)
        else:
            logger.info(f"{detector_name}_complete", result_type=type(result).__name__)

    def _write_outputs(self) -> None:
        """Write all results to output files."""
        from ran_optimizer.pipeline_outputs import OutputWriter

        writer = OutputWriter(self.config)

        for detector_name, result in self.results.items():
            if result is None:
                continue

            try:
                writer.write(detector_name, result)
            except Exception as e:
                logger.error(
                    f"output_write_failed",
                    detector=detector_name,
                    error=str(e)
                )

    def _log_summary(self, elapsed) -> None:
        """Log pipeline execution summary."""
        summary = {
            'elapsed_seconds': elapsed.total_seconds(),
            'detectors_run': len([r for r in self.results.values() if r is not None]),
            'detectors_failed': len([r for r in self.results.values() if r is None]),
        }

        for name, result in self.results.items():
            if result is None:
                summary[f'{name}_count'] = 'FAILED'
            elif isinstance(result, pd.DataFrame):
                summary[f'{name}_count'] = len(result)
            elif isinstance(result, dict):
                summary[f'{name}_count'] = sum(
                    len(v) for v in result.values() if hasattr(v, '__len__')
                )
            else:
                summary[f'{name}_count'] = 'OK'

        logger.info("pipeline_summary", **summary)


def run_pipeline(
    config_path: Path,
    detectors: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run the production pipeline.

    Args:
        config_path: Path to pipeline configuration JSON
        detectors: Optional list of detectors to run
        dry_run: If True, validate config only without running

    Returns:
        Dictionary with results from each detector
    """
    # Load configuration
    logger.info("loading_config", path=str(config_path))
    config = load_pipeline_config(config_path)

    logger.info(
        "config_loaded",
        operator=config.operator,
        region=config.region,
        source_type=config.inputs.source_type,
        enabled_detectors=config.detectors.get_enabled_detectors()
    )

    if dry_run:
        logger.info("dry_run_complete", config_valid=True)
        return {}

    # Run pipeline
    runner = PipelineRunner(config)
    return runner.run(detectors)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='RAN Optimizer - Production Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enabled detectors
  python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json

  # Run specific detectors only
  python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json --detectors overshooters pci

  # Validate configuration without running
  python -m ran_optimizer.pipeline_runner --config config/pipeline_config.json --dry-run
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/pipeline_config.json'),
        help='Path to pipeline configuration JSON file'
    )

    parser.add_argument(
        '--detectors',
        nargs='+',
        choices=PipelineRunner.DETECTOR_ORDER,
        help='Specific detectors to run (default: all enabled)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running pipeline'
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            config_path=args.config,
            detectors=args.detectors,
            dry_run=args.dry_run
        )
        return 0
    except FileNotFoundError as e:
        logger.error("config_not_found", error=str(e))
        return 1
    except DataLoadError as e:
        logger.error("data_load_error", error=str(e))
        return 2
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
