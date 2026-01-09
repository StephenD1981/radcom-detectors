"""
Output writers for pipeline results.

Handles writing detector results to standardized output formats:
- GeoJSON for spatial data
- CSV for tabular data
- Summary CSVs for aggregated metrics
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

import pandas as pd

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.pipeline_config import PipelineConfig

logger = get_logger(__name__)


class OutputWriter:
    """
    Writes detector results to standardized output formats.

    Creates output directory structure:
    output_base/
    ├── low_coverage/
    │   ├── low_coverage_areas.geojson
    │   └── low_coverage_summary.csv
    ├── no_coverage/
    │   ├── no_coverage_areas.geojson
    │   └── no_coverage_summary.csv
    ...
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize output writer.

        Args:
            config: Pipeline configuration with output settings
        """
        self.config = config
        self.base_path = config.outputs.base_path
        self.formats = config.outputs.formats

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "output_writer_initialized",
            base_path=str(self.base_path),
            formats=self.formats
        )

    def write(self, detector_name: str, result: Any) -> Dict[str, Path]:
        """
        Write detector result to appropriate format(s).

        Args:
            detector_name: Name of the detector
            result: Result from detector (DataFrame, GeoDataFrame, or dict)

        Returns:
            Dictionary mapping output type to file path
        """
        # Create detector output directory
        output_dir = self.base_path / detector_name
        output_dir.mkdir(parents=True, exist_ok=True)

        written_files = {}

        # Dispatch to appropriate writer
        if detector_name == 'overshooters':
            written_files = self._write_overshooters(output_dir, result)
        elif detector_name == 'undershooters':
            written_files = self._write_undershooters(output_dir, result)
        elif detector_name == 'low_coverage':
            written_files = self._write_low_coverage(output_dir, result)
        elif detector_name == 'no_coverage':
            written_files = self._write_no_coverage(output_dir, result)
        elif detector_name == 'interference':
            written_files = self._write_interference(output_dir, result)
        elif detector_name == 'ca_imbalance':
            written_files = self._write_ca_imbalance(output_dir, result)
        elif detector_name == 'crossed_feeders':
            written_files = self._write_crossed_feeders(output_dir, result)
        elif detector_name == 'pci':
            written_files = self._write_pci(output_dir, result)
        else:
            logger.warning(f"unknown_detector_output", detector=detector_name)

        for output_type, path in written_files.items():
            logger.info(
                "output_written",
                detector=detector_name,
                output_type=output_type,
                path=str(path)
            )

        return written_files

    def _write_overshooters(self, output_dir: Path, result: pd.DataFrame) -> Dict[str, Path]:
        """Write overshooting detector results."""
        files = {}

        if result is None or len(result) == 0:
            return files

        # Write full grid-level data
        if self.formats.get('csv', True):
            grid_path = output_dir / 'overshooter_grids.csv'
            result.to_csv(grid_path, index=False)
            files['grids'] = grid_path

            # Write summary (aggregated by cell)
            summary = self._create_tilt_summary(result, 'overshooting')
            summary_path = output_dir / 'overshooter_summary.csv'
            summary.to_csv(summary_path, index=False)
            files['summary'] = summary_path

        return files

    def _write_undershooters(self, output_dir: Path, result: pd.DataFrame) -> Dict[str, Path]:
        """Write undershooting detector results."""
        files = {}

        if result is None or len(result) == 0:
            return files

        if self.formats.get('csv', True):
            grid_path = output_dir / 'undershooter_grids.csv'
            result.to_csv(grid_path, index=False)
            files['grids'] = grid_path

            # Write summary
            summary = self._create_tilt_summary(result, 'undershooting')
            summary_path = output_dir / 'undershooter_summary.csv'
            summary.to_csv(summary_path, index=False)
            files['summary'] = summary_path

        return files

    def _write_low_coverage(self, output_dir: Path, result: Dict) -> Dict[str, Path]:
        """Write low coverage detector results (per-band)."""
        import geopandas as gpd

        files = {}

        if result is None or len(result) == 0:
            return files

        all_clusters = []

        # Write per-band GeoJSON
        for band, gdf in result.items():
            if gdf is None or len(gdf) == 0:
                continue

            if self.formats.get('geojson', True):
                band_path = output_dir / f'low_coverage_band_{band}.geojson'
                gdf.to_file(band_path, driver='GeoJSON')
                files[f'geojson_{band}'] = band_path

            # Collect for summary
            band_df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
            band_df['band'] = band
            all_clusters.append(band_df)

        # Write combined GeoJSON
        if self.formats.get('geojson', True) and all_clusters:
            combined_gdf = gpd.GeoDataFrame(
                pd.concat([gdf for gdf in result.values() if gdf is not None and len(gdf) > 0]),
                crs="EPSG:4326"
            )
            combined_path = output_dir / 'low_coverage_areas.geojson'
            combined_gdf.to_file(combined_path, driver='GeoJSON')
            files['geojson'] = combined_path

        # Write summary CSV
        if self.formats.get('csv', True) and all_clusters:
            summary_df = pd.concat(all_clusters, ignore_index=True)
            summary_path = output_dir / 'low_coverage_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            files['summary'] = summary_path

        return files

    def _write_no_coverage(self, output_dir: Path, result: Any) -> Dict[str, Path]:
        """Write no coverage (gap) detector results."""
        import geopandas as gpd

        files = {}

        if result is None or (hasattr(result, '__len__') and len(result) == 0):
            return files

        if self.formats.get('geojson', True):
            geojson_path = output_dir / 'no_coverage_areas.geojson'
            if isinstance(result, gpd.GeoDataFrame):
                result.to_file(geojson_path, driver='GeoJSON')
            files['geojson'] = geojson_path

        if self.formats.get('csv', True):
            summary_df = pd.DataFrame(result.drop(columns='geometry', errors='ignore'))
            summary_path = output_dir / 'no_coverage_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            files['summary'] = summary_path

        return files

    def _write_interference(self, output_dir: Path, result: Any) -> Dict[str, Path]:
        """Write interference detector results."""
        import geopandas as gpd

        files = {}

        if result is None:
            return files

        # Handle new format: dict with 'interference_cells' and 'grid_data' keys
        if isinstance(result, dict) and 'interference_cells' in result:
            interference_cells = result.get('interference_cells')
            grid_data = result.get('grid_data')

            if interference_cells is not None and len(interference_cells) > 0:
                if self.formats.get('csv', True):
                    cells_path = output_dir / 'interference_cells.csv'
                    interference_cells.to_csv(cells_path, index=False)
                    files['cells'] = cells_path

            if grid_data is not None and len(grid_data) > 0:
                if self.formats.get('csv', True):
                    grids_path = output_dir / 'interference_grids.csv'
                    grid_data.to_csv(grids_path, index=False)
                    files['grids'] = grids_path

            # Write summary
            if interference_cells is not None and len(interference_cells) > 0:
                if self.formats.get('csv', True):
                    summary_path = output_dir / 'interference_summary.csv'
                    interference_cells.to_csv(summary_path, index=False)
                    files['summary'] = summary_path

            return files

        # Handle legacy format: per-band dict results
        if isinstance(result, dict):
            all_results = []
            for band, data in result.items():
                if data is None or len(data) == 0:
                    continue

                if isinstance(data, gpd.GeoDataFrame) and self.formats.get('geojson', True):
                    band_path = output_dir / f'interference_band_{band}.geojson'
                    data.to_file(band_path, driver='GeoJSON')
                    files[f'geojson_{band}'] = band_path

                df = pd.DataFrame(data.drop(columns='geometry', errors='ignore') if hasattr(data, 'drop') else data)
                df['band'] = band
                all_results.append(df)

            if all_results and self.formats.get('csv', True):
                summary_df = pd.concat(all_results, ignore_index=True)
                summary_path = output_dir / 'interference_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                files['summary'] = summary_path

        elif isinstance(result, gpd.GeoDataFrame):
            if self.formats.get('geojson', True):
                geojson_path = output_dir / 'interference_areas.geojson'
                result.to_file(geojson_path, driver='GeoJSON')
                files['geojson'] = geojson_path

            if self.formats.get('csv', True):
                summary_df = pd.DataFrame(result.drop(columns='geometry', errors='ignore'))
                summary_path = output_dir / 'interference_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                files['summary'] = summary_path

        elif isinstance(result, pd.DataFrame):
            if self.formats.get('csv', True):
                csv_path = output_dir / 'interference_summary.csv'
                result.to_csv(csv_path, index=False)
                files['summary'] = csv_path

        return files

    def _write_ca_imbalance(self, output_dir: Path, result: Any) -> Dict[str, Path]:
        """Write CA imbalance detector results."""
        import geopandas as gpd

        files = {}

        if result is None or (hasattr(result, '__len__') and len(result) == 0):
            return files

        if isinstance(result, gpd.GeoDataFrame) and self.formats.get('geojson', True):
            geojson_path = output_dir / 'ca_imbalance_hulls.geojson'
            result.to_file(geojson_path, driver='GeoJSON')
            files['geojson'] = geojson_path

        if self.formats.get('csv', True):
            if isinstance(result, gpd.GeoDataFrame):
                summary_df = pd.DataFrame(result.drop(columns='geometry', errors='ignore'))
            elif isinstance(result, pd.DataFrame):
                summary_df = result
            elif isinstance(result, list):
                summary_df = pd.DataFrame(result)
            else:
                summary_df = pd.DataFrame()

            if len(summary_df) > 0:
                summary_path = output_dir / 'ca_imbalance_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                files['summary'] = summary_path

        return files

    def _write_crossed_feeders(self, output_dir: Path, result: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """Write crossed feeder detector results."""
        files = {}

        if result is None:
            return files

        if not self.formats.get('csv', True):
            return files

        # Write relation scores
        if 'relation_scores' in result and result['relation_scores'] is not None:
            rel_path = output_dir / 'crossed_feeder_relations.csv'
            result['relation_scores'].to_csv(rel_path, index=False)
            files['relations'] = rel_path

        # Write cell scores (only flagged cells)
        if 'cell_scores' in result and result['cell_scores'] is not None:
            cell_scores = result['cell_scores']
            # Filter to only flagged cells
            if 'flagged' in cell_scores.columns:
                flagged_cells = cell_scores[cell_scores['flagged'] == True].copy()
            else:
                flagged_cells = cell_scores

            if len(flagged_cells) > 0:
                cell_path = output_dir / 'crossed_feeder_cells.csv'
                flagged_cells.to_csv(cell_path, index=False)
                files['cells'] = cell_path

        # Write site summary (only sites with flagged cells)
        if 'site_summary' in result and result['site_summary'] is not None:
            site_summary = result['site_summary']
            # Filter to only sites with flagged cells
            if 'flagged_cells' in site_summary.columns:
                flagged_sites = site_summary[site_summary['flagged_cells'] > 0].copy()
            else:
                flagged_sites = site_summary

            if len(flagged_sites) > 0:
                summary_path = output_dir / 'crossed_feeder_summary.csv'
                flagged_sites.to_csv(summary_path, index=False)
                files['summary'] = summary_path

        return files

    def _write_pci(self, output_dir: Path, result: Dict) -> Dict[str, Path]:
        """Write PCI detector results."""
        files = {}

        if result is None:
            return files

        if not self.formats.get('csv', True):
            return files

        # Write hull-based conflicts
        if 'hull_conflicts' in result and result['hull_conflicts'] is not None:
            if len(result['hull_conflicts']) > 0:
                conflicts_path = output_dir / 'pci_hull_conflicts.csv'
                result['hull_conflicts'].to_csv(conflicts_path, index=False)
                files['hull_conflicts'] = conflicts_path

        # Write confusion (if PCI planner integrated)
        if 'confusion' in result and result['confusion'] is not None:
            if len(result['confusion']) > 0:
                confusion_path = output_dir / 'pci_confusion.csv'
                result['confusion'].to_csv(confusion_path, index=False)
                files['confusion'] = confusion_path

        # Write collision
        if 'collision' in result and result['collision'] is not None:
            if len(result['collision']) > 0:
                collision_path = output_dir / 'pci_collision.csv'
                result['collision'].to_csv(collision_path, index=False)
                files['collision'] = collision_path

        # Write blacklist suggestions
        if 'blacklist' in result and result['blacklist'] is not None:
            if len(result['blacklist']) > 0:
                blacklist_path = output_dir / 'pci_blacklist_suggestions.csv'
                result['blacklist'].to_csv(blacklist_path, index=False)
                files['blacklist'] = blacklist_path

        # Write summary
        summary_data = {
            'hull_conflicts': len(result.get('hull_conflicts', [])) if result.get('hull_conflicts') is not None else 0,
            'confusion_issues': len(result.get('confusion', [])) if result.get('confusion') is not None else 0,
            'collision_issues': len(result.get('collision', [])) if result.get('collision') is not None else 0,
            'blacklist_suggestions': len(result.get('blacklist', [])) if result.get('blacklist') is not None else 0,
        }
        summary_path = output_dir / 'pci_summary.csv'
        pd.DataFrame([summary_data]).to_csv(summary_path, index=False)
        files['summary'] = summary_path

        return files

    def _create_tilt_summary(self, df: pd.DataFrame, detector_type: str) -> pd.DataFrame:
        """Create summary DataFrame for tilt recommendations."""
        if 'cell_name' not in df.columns:
            return df

        # Define aggregation based on available columns
        agg_dict = {}

        # Common columns
        if 'total_grids' in df.columns:
            agg_dict['total_grids'] = 'first'

        # Overshooting specific
        if detector_type == 'overshooting':
            if 'overshooting_grids' in df.columns:
                agg_dict['overshooting_grids'] = 'first'
            if 'percentage_overshooting' in df.columns:
                agg_dict['percentage_overshooting'] = 'first'
            if 'max_distance_m' in df.columns:
                agg_dict['max_distance_m'] = 'first'
            if 'recommended_tilt_change' in df.columns:
                agg_dict['recommended_tilt_change'] = 'first'
            if 'severity_score' in df.columns:
                agg_dict['severity_score'] = 'first'
            if 'severity_category' in df.columns:
                agg_dict['severity_category'] = 'first'

        # Undershooting specific
        elif detector_type == 'undershooting':
            if 'interference_grids' in df.columns:
                agg_dict['interference_grids'] = 'first'
            if 'interference_percentage' in df.columns:
                agg_dict['interference_percentage'] = 'first'
            if 'recommended_uptilt_deg' in df.columns:
                agg_dict['recommended_uptilt_deg'] = 'first'
            if 'coverage_increase_percentage' in df.columns:
                agg_dict['coverage_increase_percentage'] = 'first'
            if 'severity_score' in df.columns:
                agg_dict['severity_score'] = 'first'
            if 'severity_category' in df.columns:
                agg_dict['severity_category'] = 'first'

        if not agg_dict:
            # Return unique cells if no aggregation columns
            return df.drop_duplicates(subset='cell_name')

        summary = df.groupby('cell_name').agg(agg_dict).reset_index()
        return summary


def write_pipeline_metadata(
    output_dir: Path,
    config: PipelineConfig,
    results: Dict[str, Any],
    elapsed_seconds: float
) -> Path:
    """
    Write pipeline execution metadata.

    Args:
        output_dir: Output directory
        config: Pipeline configuration
        results: Results from all detectors
        elapsed_seconds: Total execution time

    Returns:
        Path to metadata file
    """
    from datetime import datetime

    metadata = {
        'execution_time': datetime.now().isoformat(),
        'elapsed_seconds': elapsed_seconds,
        'config': {
            'version': config.version,
            'operator': config.operator,
            'region': config.region,
            'source_type': config.inputs.source_type,
        },
        'results': {}
    }

    for detector_name, result in results.items():
        if result is None:
            metadata['results'][detector_name] = {'status': 'FAILED', 'count': 0}
        elif isinstance(result, pd.DataFrame):
            metadata['results'][detector_name] = {'status': 'OK', 'count': len(result)}
        elif isinstance(result, dict):
            total = sum(len(v) for v in result.values() if hasattr(v, '__len__'))
            metadata['results'][detector_name] = {'status': 'OK', 'count': total}
        else:
            metadata['results'][detector_name] = {'status': 'OK', 'count': 'N/A'}

    metadata_path = output_dir / 'pipeline_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("metadata_written", path=str(metadata_path))
    return metadata_path
