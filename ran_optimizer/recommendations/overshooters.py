"""
Overshooting cell detection for RAN optimization.

Identifies cells that are serving traffic too far from their optimal coverage area,
typically due to excessive antenna tilt or lack of nearby alternative coverage.

Algorithm based on legacy prototype code from tilt-optimisation-overshooters.ipynb
"""
import pandas as pd
import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.overshooting_config import (
    load_overshooting_config,
    get_environment_params,
    get_default_config_path,
)

logger = get_logger(__name__)


@dataclass
class OvershooterParams:
    """Parameters for overshooting cell detection algorithm."""

    # Step 1: Edge traffic threshold
    edge_traffic_percent: float = 0.15  # 15% threshold for "edge" traffic

    # Step 2: Distance filter
    min_cell_distance: float = 4000  # Minimum 4km from cell

    # Step 3: Grid bin criteria (RSRP-based competition)
    interference_threshold_db: float = 7.5  # RSRP diff threshold for competing cells
    min_cell_count_in_grid: int = 4  # Min competing cells (within interference_threshold_db)
    max_percentage_grid_events: float = 0.25  # Max 25% of grid samples from one cell
    rsrp_competition_quantile: float = 0.90  # P90 quantile for reference RSRP

    # Step 3b: Relative distance criterion
    min_relative_reach: float = 0.7  # Cell must reach ≥70% as far as furthest competitor

    # Step 4: RSRP degradation
    rsrp_degradation_db: float = 10.0  # Required RSRP degradation in dB from cell's max RSRP

    # Step 5: Final thresholds
    min_overshooting_grids: int = 30  # Min bins to flag cell as overshooting
    percentage_overshooting_grids: float = 0.10  # 10% of cell's total bins

    # Data quality filters
    rsrp_min_dbm: float = -140.0  # Minimum valid RSRP
    rsrp_max_dbm: float = -30.0  # Maximum valid RSRP
    default_rsrp_dbm: float = -140.0  # Default RSRP for missing data

    # RF propagation parameters (for tilt recommendations)
    hpbw_v_deg: float = 6.5  # Vertical half-power beamwidth
    sla_v_db: float = 30.0  # Side-lobe attenuation cap
    path_loss_exponent: float = 3.5  # Path loss exponent
    default_antenna_height_m: float = 30.0  # Default antenna height
    cell_edge_rsrp_threshold_dbm: float = -110.0  # Cell edge RSRP threshold

    # Tilt decision thresholds
    min_resolution_pct_for_1deg: float = 0.50  # Min resolution % for 1° recommendation

    # Severity score weights
    severity_weight_bins: float = 0.30
    severity_weight_percentage: float = 0.25
    severity_weight_distance: float = 0.20
    severity_weight_rsrp: float = 0.15
    severity_weight_traffic: float = 0.10

    # Severity normalization ranges
    severity_distance_min_m: float = 4000.0  # Baseline distance
    severity_distance_max_m: float = 35000.0  # Extreme distance
    severity_rsrp_best_dbm: float = -70.0  # Best RSRP
    severity_rsrp_worst_dbm: float = -120.0  # Worst RSRP

    # Severity thresholds
    severity_threshold_critical: float = 0.80
    severity_threshold_high: float = 0.60
    severity_threshold_medium: float = 0.40
    severity_threshold_low: float = 0.20

    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None,
        environment: Optional[str] = None
    ) -> 'OvershooterParams':
        """
        Create OvershooterParams from JSON configuration file.

        Args:
            config_path: Path to JSON config file. If None, uses default config path.
            environment: Environment type ('urban', 'suburban', 'rural') or None for default.
                        If specified, environment-specific parameter overrides will be applied.

        Returns:
            OvershooterParams instance with parameters from config

        Example:
            >>> # Load default parameters
            >>> params = OvershooterParams.from_config()

            >>> # Load with custom config file
            >>> params = OvershooterParams.from_config('config/custom_params.json')

            >>> # Load urban-specific parameters
            >>> params = OvershooterParams.from_config(environment='urban')
        """
        # Get config path
        if config_path is None:
            config_path = get_default_config_path()

        # Load configuration
        config = load_overshooting_config(str(config_path))

        # Get environment-specific parameters (flat top-level params)
        params_dict = get_environment_params(config, environment)

        # Flatten nested config sections into dataclass field names
        # data_quality section
        if 'data_quality' in params_dict:
            dq = params_dict['data_quality']
            params_dict['rsrp_min_dbm'] = dq.get('rsrp_min_dbm', -140.0)
            params_dict['rsrp_max_dbm'] = dq.get('rsrp_max_dbm', -30.0)
            params_dict['default_rsrp_dbm'] = dq.get('default_rsrp_dbm', -140.0)

        # rf_propagation section
        if 'rf_propagation' in params_dict:
            rf = params_dict['rf_propagation']
            params_dict['hpbw_v_deg'] = rf.get('hpbw_v_deg', 6.5)
            params_dict['sla_v_db'] = rf.get('sla_v_db', 30.0)
            params_dict['path_loss_exponent'] = rf.get('path_loss_exponent', 3.5)
            params_dict['default_antenna_height_m'] = rf.get('default_antenna_height_m', 30.0)
            params_dict['cell_edge_rsrp_threshold_dbm'] = rf.get('cell_edge_rsrp_threshold_dbm', -110.0)

        # tilt_decision section
        if 'tilt_decision' in params_dict:
            td = params_dict['tilt_decision']
            params_dict['min_resolution_pct_for_1deg'] = td.get('min_resolution_pct_for_1deg', 0.50)

        # severity_weights section
        if 'severity_weights' in params_dict:
            sw = params_dict['severity_weights']
            params_dict['severity_weight_bins'] = sw.get('overshooting_bins', 0.30)
            params_dict['severity_weight_percentage'] = sw.get('percentage_overshooting', 0.25)
            params_dict['severity_weight_distance'] = sw.get('max_distance', 0.20)
            params_dict['severity_weight_rsrp'] = sw.get('rsrp_degradation', 0.15)
            params_dict['severity_weight_traffic'] = sw.get('traffic_impact', 0.10)

        # severity_normalization section
        if 'severity_normalization' in params_dict:
            sn = params_dict['severity_normalization']
            params_dict['severity_distance_min_m'] = sn.get('distance_min_m', 4000.0)
            params_dict['severity_distance_max_m'] = sn.get('distance_max_m', 35000.0)
            params_dict['severity_rsrp_best_dbm'] = sn.get('rsrp_best_dbm', -70.0)
            params_dict['severity_rsrp_worst_dbm'] = sn.get('rsrp_worst_dbm', -120.0)

        # severity_thresholds section
        if 'severity_thresholds' in params_dict:
            st = params_dict['severity_thresholds']
            params_dict['severity_threshold_critical'] = st.get('critical', 0.80)
            params_dict['severity_threshold_high'] = st.get('high', 0.60)
            params_dict['severity_threshold_medium'] = st.get('medium', 0.40)
            params_dict['severity_threshold_low'] = st.get('low', 0.20)

        # Extract only the parameters that belong to OvershooterParams
        valid_params = {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name in params_dict:
                valid_params[field_name] = params_dict[field_name]

        logger.info(
            "Created OvershooterParams from config",
            config_file=str(config_path),
            environment=environment or 'default',
            parameters_loaded=len(valid_params),
        )

        return cls(**valid_params)


class OvershooterDetector:
    """
    Detects cells that are overshooting their optimal coverage area.

    The algorithm identifies cells serving significant traffic at the edge of
    their coverage area with degraded signal quality, suggesting the cell tilt
    should be increased (beam pointed down more).

    Expected input schemas:

    grid_df (from cell_coverage.csv):
        - grid: geohash7 identifier
        - cell_name: cell identifier
        - avg_rsrp: signal strength (dBm)
        - event_count: number of measurements
        - distance_to_cell: pre-calculated distance (meters)
        - Band: frequency band
        - Latitude, Longitude: grid coordinates

    gis_df (from cell_gis.csv):
        - cell_name: cell identifier
        - latitude, longitude: cell coordinates
        - bearing: antenna azimuth
        - tilt_elc: electrical tilt
        - tilt_mech: mechanical tilt
        - antenna_height: height (meters)
        - band: frequency band

    Example:
        >>> detector = OvershooterDetector(params)
        >>> overshooters = detector.detect(grid_df, gis_df)
        >>> print(f"Found {len(overshooters)} overshooting cells")
    """

    def __init__(self, params: OvershooterParams):
        """
        Initialize the detector with algorithm parameters.

        Args:
            params: OvershooterParams with detection thresholds
        """
        self.params = params
        logger.info(
            "Initialized OvershooterDetector",
            edge_traffic_percent=params.edge_traffic_percent,
            min_cell_distance=params.min_cell_distance,
        )

    def detect(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect overshooting cells using grid measurement data.

        Args:
            grid_df: Grid measurements (from cell_coverage.csv) with columns:
                - grid: geohash7 identifier
                - cell_name: cell identifier
                - avg_rsrp: signal strength (dBm)
                - event_count: number of measurements
                - distance_to_cell: pre-calculated distance (meters)
                - Band: frequency band
                - Latitude, Longitude: grid coordinates

            gis_df: Cell GIS data (from cell_gis.csv) with columns:
                - cell_name: cell identifier
                - latitude, longitude: cell coordinates
                - bearing: antenna azimuth
                - tilt_elc: electrical tilt
                - tilt_mech: mechanical tilt
                - antenna_height: height (meters)

        Returns:
            DataFrame with overshooting cells and metrics:
                - cell_name: Cell identifier
                - overshooting_grids: Number of overshooting bins
                - total_grids: Total bins served by cell
                - percentage_overshooting: % of bins that are overshooting
                - max_distance_m: Furthest serving distance
                - avg_edge_rsrp: Average RSRP in edge bins
                - recommended_tilt_change: Suggested tilt increase (degrees)

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(
            "Starting overshooting detection",
            grid_measurements=len(grid_df),
            cells=len(gis_df),
        )

        # Validate input data
        self._validate_inputs(grid_df, gis_df)

        # Sanitise data - remove rows with critical null values
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        # Step 1: Calculate distance from each grid bin to its serving cell
        logger.info("Step 1: Calculating grid-to-cell distances")
        grid_with_distance = self._calculate_grid_distances(grid_df, gis_df)

        # Step 2: Identify edge traffic bins
        logger.info(
            "Step 2: Identifying edge traffic bins",
            threshold=self.params.edge_traffic_percent,
        )
        edge_bins = self._identify_edge_bins(grid_with_distance)

        # Step 3: Calculate per-cell metrics
        logger.info("Step 3: Calculating per-cell edge metrics")
        cell_metrics = self._calculate_cell_metrics(grid_with_distance, edge_bins)

        # Step 4: Apply overshooting criteria
        logger.info("Step 4: Applying overshooting filters")
        overshooting_candidates, overshooting_bins = self._apply_overshooting_filters(
            cell_metrics,
            edge_bins,
            grid_with_distance,
        )

        # Step 5: Calculate tilt recommendations
        logger.info("Step 5: Calculating tilt recommendations")
        overshooters = self._calculate_tilt_recommendations(
            overshooting_candidates,
            overshooting_bins,
            gis_df,
            grid_with_distance
        )

        # Step 6: Calculate severity scores
        logger.info("Step 6: Calculating severity scores")
        overshooters = self._calculate_severity_scores(
            overshooters,
            grid_df
        )

        logger.info(
            "Overshooting detection complete",
            overshooters_found=len(overshooters),
            total_cells_analyzed=len(cell_metrics),
        )

        return overshooters

    def detect_with_grids(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect overshooting cells and return grid-level detail.

        Same as detect(), but also returns the overshooting grids DataFrame
        for downstream analysis (e.g., tier_3 metrics calculation).

        Args:
            grid_df: Grid measurements (from cell_coverage.csv)
            gis_df: Cell GIS data (from cell_gis.csv)

        Returns:
            Tuple of (overshooters_df, overshooting_grids_df):
                - overshooters_df: Cell-level recommendations
                - overshooting_grids_df: Grid-level detail with columns:
                    - cell_name: Cell identifier
                    - grid: geohash7 identifier
                    - avg_rsrp: Signal strength
                    - distance_m: Distance from cell
                    - competing_cells: Number of other cells in grid
                    - Band: Frequency band (if available)
        """
        logger.info(
            "Starting overshooting detection (with grids)",
            grid_measurements=len(grid_df),
            cells=len(gis_df),
        )

        # Validate input data
        self._validate_inputs(grid_df, gis_df)

        # Sanitise data
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        # Step 1: Calculate distance from each grid bin to its serving cell
        grid_with_distance = self._calculate_grid_distances(grid_df, gis_df)

        # Step 2: Identify edge traffic bins
        edge_bins = self._identify_edge_bins(grid_with_distance)

        # Step 3: Calculate per-cell metrics
        cell_metrics = self._calculate_cell_metrics(grid_with_distance, edge_bins)

        # Step 4: Apply overshooting criteria
        overshooting_candidates, overshooting_bins = self._apply_overshooting_filters(
            cell_metrics,
            edge_bins,
            grid_with_distance,
        )

        # Step 5: Calculate tilt recommendations
        overshooters = self._calculate_tilt_recommendations(
            overshooting_candidates,
            overshooting_bins,
            gis_df,
            grid_with_distance
        )

        # Step 6: Calculate severity scores
        overshooters = self._calculate_severity_scores(
            overshooters,
            grid_df
        )

        logger.info(
            "Overshooting detection complete (with grids)",
            overshooters_found=len(overshooters),
            overshooting_grids=len(overshooting_bins),
        )

        return overshooters, overshooting_bins

    def detect_with_environments(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
        environment_df: pd.DataFrame,
        env_params: dict,
        return_grids: bool = False,
    ):
        """
        Detect overshooting cells using per-cell environment-specific parameters.

        This runs detection ONCE, using each cell's classified environment to determine
        its edge_traffic_percent and final thresholds. Competition analysis uses all cells
        together to maintain correct cross-environment interference calculation.

        Args:
            grid_df: Grid measurements (from cell_coverage.csv)
            gis_df: Cell GIS data (from cell_gis.csv)
            environment_df: DataFrame with columns:
                - cell_name: Cell identifier
                - environment: 'URBAN', 'SUBURBAN', or 'RURAL'
            env_params: Dict mapping environment name to OvershooterParams
                e.g., {'urban': OvershooterParams(...), 'suburban': ..., 'rural': ...}
            return_grids: If True, also returns overshooting_grids DataFrame

        Returns:
            If return_grids=False: DataFrame of overshooters
            If return_grids=True: Tuple of (overshooters_df, overshooting_grids_df)
        """
        logger.info(
            "Starting environment-aware overshooting detection (single pass)",
            grid_measurements=len(grid_df),
            cells=len(gis_df),
            environments=environment_df['environment'].value_counts().to_dict(),
        )

        # Validate input data
        self._validate_inputs(grid_df, gis_df)

        # Sanitise data
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        # Ensure cell_name types match
        environment_df = environment_df.copy()
        environment_df['cell_name'] = environment_df['cell_name'].astype(str)

        # Create cell -> environment mapping
        cell_env_map = dict(zip(environment_df['cell_name'], environment_df['environment']))

        # Step 1: Calculate distance from each grid bin to its serving cell
        logger.info("Step 1: Calculating grid-to-cell distances")
        grid_with_distance = self._calculate_grid_distances(grid_df, gis_df)

        # Step 2: Identify edge traffic bins using PER-CELL edge_traffic_percent
        logger.info("Step 2: Identifying edge traffic bins (per-cell environment thresholds)")
        edge_bins = self._identify_edge_bins_by_environment(
            grid_with_distance, cell_env_map, env_params
        )

        # Step 3: Calculate per-cell metrics
        logger.info("Step 3: Calculating per-cell edge metrics")
        cell_metrics = self._calculate_cell_metrics(grid_with_distance, edge_bins)

        # Step 4: Apply overshooting criteria with per-cell thresholds
        logger.info("Step 4: Applying overshooting filters (per-cell environment thresholds)")
        overshooting_candidates, overshooting_bins = self._apply_overshooting_filters_by_environment(
            cell_metrics,
            edge_bins,
            grid_with_distance,
            cell_env_map,
            env_params,
        )

        # Step 5: Calculate tilt recommendations
        logger.info("Step 5: Calculating tilt recommendations")
        overshooters = self._calculate_tilt_recommendations(
            overshooting_candidates,
            overshooting_bins,
            gis_df,
            grid_with_distance
        )

        # Step 6: Calculate severity scores
        logger.info("Step 6: Calculating severity scores")
        overshooters = self._calculate_severity_scores(overshooters, grid_df)

        # Add environment info to results
        if len(overshooters) > 0:
            overshooters['environment'] = overshooters['cell_name'].map(cell_env_map)
            overshooters = overshooters.merge(
                environment_df[['cell_name', 'intersite_distance_km']],
                on='cell_name',
                how='left'
            )

        logger.info(
            "Environment-aware detection complete (single pass)",
            overshooters_found=len(overshooters),
            overshooting_grids=len(overshooting_bins),
        )

        if return_grids:
            # Add environment to grids
            if len(overshooting_bins) > 0:
                overshooting_bins['environment'] = overshooting_bins['cell_name'].map(cell_env_map)
            return overshooters, overshooting_bins
        return overshooters

    def _identify_edge_bins_by_environment(
        self,
        grid_df: pd.DataFrame,
        cell_env_map: dict,
        env_params: dict,
    ) -> pd.DataFrame:
        """
        Identify edge bins using per-cell edge_traffic_percent based on environment.

        Args:
            grid_df: Grid data with distance_m column
            cell_env_map: Dict mapping cell_name -> environment ('URBAN', 'SUBURBAN', 'RURAL')
            env_params: Dict mapping environment -> OvershooterParams

        Returns:
            DataFrame of edge bins
        """
        # Map each cell to its edge_traffic_percent
        def get_edge_percent(cell_name):
            env = cell_env_map.get(cell_name, 'SUBURBAN')
            params = env_params.get(env.lower(), env_params.get('suburban'))
            return params.edge_traffic_percent

        # Get unique cells and their edge_traffic_percent
        cells = grid_df['cell_name'].unique()
        cell_edge_pct = {cell: get_edge_percent(cell) for cell in cells}

        # Calculate edge threshold per cell using its environment's edge_traffic_percent
        edge_thresholds = []
        for cell_name, edge_pct in cell_edge_pct.items():
            cell_data = grid_df[grid_df['cell_name'] == cell_name]
            if len(cell_data) > 0:
                threshold = cell_data['distance_m'].quantile(1 - edge_pct)
                edge_thresholds.append({
                    'cell_name': cell_name,
                    'edge_distance_m': threshold,
                    'edge_traffic_percent': edge_pct,
                })

        edge_threshold_df = pd.DataFrame(edge_thresholds)

        # Merge threshold back to grid
        grid_with_threshold = grid_df.merge(edge_threshold_df, on='cell_name', how='left')

        # Filter to edge bins only (distance >= edge threshold)
        edge_bins = grid_with_threshold[
            grid_with_threshold['distance_m'] >= grid_with_threshold['edge_distance_m']
        ].copy()

        # Log per-environment edge bin counts
        edge_bins['_env'] = edge_bins['cell_name'].map(cell_env_map)
        env_counts = edge_bins.groupby('_env').size().to_dict()
        edge_bins = edge_bins.drop(columns=['_env'])

        logger.info(
            "Edge bins identified (per-cell environment thresholds)",
            total_bins=len(grid_df),
            edge_bins=len(edge_bins),
            edge_percentage=len(edge_bins) / len(grid_df) * 100 if len(grid_df) > 0 else 0,
            by_environment=env_counts,
        )

        return edge_bins

    def _apply_overshooting_filters_by_environment(
        self,
        cell_metrics: pd.DataFrame,
        edge_bins: pd.DataFrame,
        full_grid_df: pd.DataFrame,
        cell_env_map: dict,
        env_params: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply overshooting filters using per-cell environment-specific thresholds.

        Uses each cell's environment to determine:
        - min_cell_distance
        - min_cell_count_in_grid
        - min_relative_reach
        - min_overshooting_grids
        - percentage_overshooting_grids
        """
        # Helper to get params for a cell
        def get_params(cell_name):
            env = cell_env_map.get(cell_name, 'SUBURBAN')
            return env_params.get(env.lower(), env_params.get('suburban'))

        # Add environment to cell_metrics for per-cell filtering
        cell_metrics = cell_metrics.copy()
        cell_metrics['environment'] = cell_metrics['cell_name'].map(cell_env_map)

        # Filter 1: Cell must serve traffic beyond its environment's min_cell_distance
        def passes_distance_filter(row):
            params = get_params(row['cell_name'])
            return row['max_distance_m'] >= params.min_cell_distance

        candidates = cell_metrics[cell_metrics.apply(passes_distance_filter, axis=1)].copy()
        logger.info(
            "After distance filter (per-cell)",
            candidates=len(candidates),
            by_environment=candidates['environment'].value_counts().to_dict() if len(candidates) > 0 else {},
        )

        if len(candidates) == 0 or len(edge_bins) == 0:
            logger.info("No overshooting candidates found")
            return pd.DataFrame(), pd.DataFrame()

        # Filter 2: RSRP-based competition
        # IMPORTANT: Use full_grid_df (all cells) for P90 RSRP and competition analysis,
        # then filter back to edge_bins rows. This ensures we count ALL competing cells
        # in each grid, not just cells with edge traffic.
        band_col = 'Band' if 'Band' in full_grid_df.columns else None

        # Calculate P90 RSRP per grid from ALL cell measurements
        if band_col is not None:
            p90_rsrp_per_grid = full_grid_df.groupby(['grid', band_col])['avg_rsrp'].quantile(0.9)
            # Calculate on full grid
            full_grid_analysis = full_grid_df.copy()
            full_grid_analysis['p90_rsrp_in_grid'] = full_grid_analysis.set_index(['grid', band_col]).index.map(p90_rsrp_per_grid)
        else:
            p90_rsrp_per_grid = full_grid_df.groupby('grid')['avg_rsrp'].quantile(0.9).to_dict()
            full_grid_analysis = full_grid_df.copy()
            full_grid_analysis['p90_rsrp_in_grid'] = full_grid_analysis['grid'].map(p90_rsrp_per_grid)

        # Flag competing cells (using the most permissive interference_threshold_db)
        max_interference_threshold = max(
            p.interference_threshold_db for p in env_params.values()
        )
        full_grid_analysis['rsrp_diff'] = full_grid_analysis['p90_rsrp_in_grid'] - full_grid_analysis['avg_rsrp']
        full_grid_analysis['is_competing'] = full_grid_analysis['rsrp_diff'] <= max_interference_threshold

        # Count competing cells per grid from FULL grid data
        if band_col is not None:
            competing_counts = full_grid_analysis[full_grid_analysis['is_competing']].groupby(['grid', band_col])['cell_name'].nunique().reset_index()
            competing_counts.columns = ['grid', band_col, 'competing_cells']
            traffic = full_grid_analysis.groupby(['grid', band_col])['event_count'].sum().reset_index()
            traffic.columns = ['grid', band_col, 'grid_total_traffic']
            competing_counts = competing_counts.merge(traffic, on=['grid', band_col], how='left')
            # Apply counts back to edge_bins only (filter to candidate cells' edge measurements)
            edge_with_counts = edge_bins.merge(competing_counts, on=['grid', band_col], how='left')
        else:
            competing_counts = full_grid_analysis[full_grid_analysis['is_competing']].groupby('grid')['cell_name'].nunique().reset_index()
            competing_counts.columns = ['grid', 'competing_cells']
            traffic = full_grid_analysis.groupby('grid')['event_count'].sum().reset_index()
            traffic.columns = ['grid', 'grid_total_traffic']
            competing_counts = competing_counts.merge(traffic, on='grid', how='left')
            edge_with_counts = edge_bins.merge(competing_counts, on='grid', how='left')

        edge_with_counts['competing_cells'] = edge_with_counts['competing_cells'].fillna(0).astype(int)

        # Calculate cell traffic percentage
        total_grid_traffic = edge_with_counts['grid_total_traffic'].replace(0, np.nan).fillna(1)
        edge_with_counts['cell_traffic_pct'] = edge_with_counts['event_count'] / total_grid_traffic

        # Apply per-cell min_cell_count_in_grid and max_percentage_grid_events filters
        def passes_competition_filter(row):
            params = get_params(row['cell_name'])
            return (
                row['competing_cells'] >= params.min_cell_count_in_grid and
                row['cell_traffic_pct'] <= params.max_percentage_grid_events
            )

        competition_bins = edge_with_counts[edge_with_counts.apply(passes_competition_filter, axis=1)].copy()

        logger.info(
            "RSRP-based competition filter applied (per-cell thresholds)",
            edge_bins=len(edge_bins),
            competition_bins=len(competition_bins),
        )

        if len(competition_bins) == 0:
            return pd.DataFrame(), pd.DataFrame()

        # Filter 3b: Relative reach (per-cell threshold)
        if band_col is not None:
            grid_max_dist = full_grid_df.groupby(['grid', band_col])['distance_m'].max().reset_index()
            grid_max_dist.columns = ['grid', band_col, 'grid_max_distance']
            competition_bins = competition_bins.merge(grid_max_dist, on=['grid', band_col], how='left')
        else:
            grid_max_dist = full_grid_df.groupby('grid')['distance_m'].max().reset_index()
            grid_max_dist.columns = ['grid', 'grid_max_distance']
            competition_bins = competition_bins.merge(grid_max_dist, on='grid', how='left')

        competition_bins['relative_reach'] = (
            competition_bins['distance_m'] / competition_bins['grid_max_distance']
        )

        def passes_relative_reach(row):
            params = get_params(row['cell_name'])
            return row['relative_reach'] >= params.min_relative_reach

        overshooting_bins = competition_bins[competition_bins.apply(passes_relative_reach, axis=1)].copy()

        logger.info(
            "After relative distance filter (per-cell)",
            competition_bins=len(competition_bins),
            overshooting_bins=len(overshooting_bins),
        )

        # Filter 4: RSRP degradation (using instance params - same for all)
        cell_max_rsrp = cell_metrics[['cell_name', 'cell_max_rsrp']]
        overshooting_bins = overshooting_bins.merge(cell_max_rsrp, on='cell_name', how='left')
        overshooting_bins['edge_rsrp_threshold'] = (
            overshooting_bins['cell_max_rsrp'] - self.params.rsrp_degradation_db
        )

        bins_before = len(overshooting_bins)
        overshooting_bins = overshooting_bins[
            overshooting_bins['avg_rsrp'] <= overshooting_bins['edge_rsrp_threshold']
        ].copy()

        logger.info(
            "After RSRP degradation filter",
            overshooting_bins=len(overshooting_bins),
            bins_before=bins_before,
        )

        # Count overshooting bins per cell
        overshooting_per_cell = overshooting_bins.groupby('cell_name').agg({
            'grid': 'nunique',
        }).reset_index()
        overshooting_per_cell.columns = ['cell_name', 'overshooting_grids']

        # Merge with candidates
        candidates = candidates.merge(overshooting_per_cell, on='cell_name', how='left')
        candidates['overshooting_grids'] = candidates['overshooting_grids'].fillna(0).astype(int)
        candidates['percentage_overshooting'] = candidates['overshooting_grids'] / candidates['total_grids']

        # Filter 5: Final thresholds (per-cell)
        def passes_final_threshold(row):
            params = get_params(row['cell_name'])
            return (
                row['overshooting_grids'] >= params.min_overshooting_grids and
                row['percentage_overshooting'] >= params.percentage_overshooting_grids
            )

        overshooters = candidates[candidates.apply(passes_final_threshold, axis=1)].copy()

        # Log per-environment results
        if len(overshooters) > 0:
            env_counts = overshooters['environment'].value_counts().to_dict()
        else:
            env_counts = {}

        logger.info(
            "After final thresholds (per-cell)",
            overshooters=len(overshooters),
            by_environment=env_counts,
        )

        return overshooters, overshooting_bins

    def _validate_inputs(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame
    ) -> None:
        """
        Validate required columns exist in input data.

        Expected schemas:
            grid_df: grid, cell_name, avg_rsrp, event_count, distance_to_cell, Band
            gis_df: cell_name, latitude, longitude, bearing
        """
        # Required grid columns
        grid_required = ['grid', 'cell_name', 'avg_rsrp', 'event_count']
        grid_missing = [col for col in grid_required if col not in grid_df.columns]
        if grid_missing:
            raise ValueError(
                f"Grid data missing required columns: {grid_missing}. "
                f"Available columns: {list(grid_df.columns)}"
            )

        # Required GIS columns
        gis_required = ['cell_name', 'latitude', 'longitude']
        gis_missing = [col for col in gis_required if col not in gis_df.columns]
        if gis_missing:
            raise ValueError(
                f"GIS data missing required columns: {gis_missing}. "
                f"Available columns: {list(gis_df.columns)}"
            )

        # Validate data quality
        if len(grid_df) == 0:
            raise ValueError("Grid data is empty")
        if len(gis_df) == 0:
            raise ValueError("GIS data is empty")

        # Check for critical null values in key columns
        if grid_df['cell_name'].isna().any():
            null_count = grid_df['cell_name'].isna().sum()
            logger.warning(f"Grid data has {null_count} null cell_name values - these will be excluded")

    def _sanitise_inputs(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sanitise input data by removing rows with critical null values
        and invalid ranges.

        Returns:
            Tuple of (sanitised_grid_df, sanitised_gis_df)
        """
        grid_df = grid_df.copy()
        gis_df = gis_df.copy()
        original_grid_len = len(grid_df)
        original_gis_len = len(gis_df)

        # Remove rows with null cell_name
        grid_df = grid_df.dropna(subset=['cell_name'])
        gis_df = gis_df.dropna(subset=['cell_name'])

        # Remove rows with null/invalid RSRP (grid)
        if 'avg_rsrp' in grid_df.columns:
            grid_df = grid_df[grid_df['avg_rsrp'].notna()]
            # Filter invalid RSRP range (using config values)
            grid_df = grid_df[
                (grid_df['avg_rsrp'] >= self.params.rsrp_min_dbm) &
                (grid_df['avg_rsrp'] <= self.params.rsrp_max_dbm)
            ]

        # Remove rows with null distance (if present)
        if 'distance_to_cell' in grid_df.columns:
            grid_df = grid_df[grid_df['distance_to_cell'].notna()]
            # Filter invalid distances
            grid_df = grid_df[grid_df['distance_to_cell'] >= 0]

        # Remove GIS rows with invalid coordinates
        if 'latitude' in gis_df.columns and 'longitude' in gis_df.columns:
            gis_df = gis_df[
                (gis_df['latitude'].notna()) &
                (gis_df['longitude'].notna()) &
                (gis_df['latitude'].abs() <= 90) &
                (gis_df['longitude'].abs() <= 180)
            ]

        # Log sanitisation results
        grid_removed = original_grid_len - len(grid_df)
        gis_removed = original_gis_len - len(gis_df)

        if grid_removed > 0 or gis_removed > 0:
            logger.info(
                "Data sanitisation complete",
                grid_rows_removed=grid_removed,
                gis_rows_removed=gis_removed,
                grid_rows_remaining=len(grid_df),
                gis_rows_remaining=len(gis_df),
            )

        return grid_df, gis_df

    def _calculate_grid_distances(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate distance from each grid bin to its serving cell.

        Returns grid_df with added 'distance_m' column.
        """
        # If distance already provided (distance_to_cell from cell_coverage.csv), use it
        if 'distance_to_cell' in grid_df.columns:
            logger.info("Using pre-calculated distance_to_cell column")
            result = grid_df.copy()
            # Rename to internal column name for consistency
            result['distance_m'] = result['distance_to_cell']
            return result

        # If grid already has lat/lon, calculate distance
        if 'Latitude' in grid_df.columns and 'Longitude' in grid_df.columns:
            grid_with_coords = grid_df.copy()
        else:
            raise ValueError(
                "Grid data must include 'distance_to_cell' or 'Latitude'/'Longitude' columns."
            )

        # Merge grid with cell GIS data to get cell coordinates
        grid_with_cell = grid_with_coords.merge(
            gis_df[['cell_name', 'latitude', 'longitude']],
            on='cell_name',
            how='left',
            suffixes=('_grid', '_cell')
        )

        # Guard: drop rows where cell coordinates are null (cell not in GIS data)
        rows_before = len(grid_with_cell)
        grid_with_cell = grid_with_cell.dropna(subset=['latitude_cell', 'longitude_cell'])
        rows_dropped = rows_before - len(grid_with_cell)
        if rows_dropped > 0:
            logger.warning(
                f"Dropped {rows_dropped} grid rows with missing cell GIS coordinates "
                f"({rows_dropped / rows_before * 100:.1f}% of data)"
            )

        # Calculate distance using vectorized haversine
        grid_with_cell = grid_with_cell.copy()

        # Vectorized haversine formula
        EARTH_RADIUS_M = 6371000.0
        lat1 = np.radians(grid_with_cell['latitude_cell'].values)
        lon1 = np.radians(grid_with_cell['longitude_cell'].values)
        lat2 = np.radians(grid_with_cell['Latitude_grid'].values)
        lon2 = np.radians(grid_with_cell['Longitude_grid'].values)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        grid_with_cell['distance_m'] = EARTH_RADIUS_M * c

        # Clean up - drop cell coord columns and restore original column names
        result = grid_with_cell.drop(
            columns=['latitude_cell', 'longitude_cell'],
            errors='ignore'
        ).rename(columns={
            'Latitude_grid': 'Latitude',
            'Longitude_grid': 'Longitude'
        })

        return result

    def _identify_edge_bins(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify grid bins at the edge of each cell's coverage.

        Edge bins are those beyond edge_traffic_percent quantile of
        distance distribution for each cell.
        """
        # Calculate distance quantile per cell
        # Use (1 - edge_traffic_percent) to get the threshold above which "edge" bins lie
        # e.g., if edge_traffic_percent=0.15, we want the 85th percentile as the threshold
        # so that bins >= this threshold represent the furthest 15% of traffic
        edge_threshold = grid_df.groupby('cell_name')['distance_m'].quantile(
            1 - self.params.edge_traffic_percent
        ).reset_index()
        edge_threshold.columns = ['cell_name', 'edge_distance_m']

        # Merge threshold back to grid
        grid_with_threshold = grid_df.merge(edge_threshold, on='cell_name', how='left')

        # Filter to edge bins only (distance >= edge threshold)
        edge_bins = grid_with_threshold[
            grid_with_threshold['distance_m'] >= grid_with_threshold['edge_distance_m']
        ].copy()

        logger.info(
            "Edge bins identified",
            total_bins=len(grid_df),
            edge_bins=len(edge_bins),
            edge_percentage=len(edge_bins) / len(grid_df) * 100 if len(grid_df) > 0 else 0,
        )

        return edge_bins

    def _calculate_cell_metrics(
        self,
        grid_df: pd.DataFrame,
        edge_bins: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate per-cell metrics needed for overshooting detection.

        Returns:
            DataFrame with one row per cell:
                - cell_name
                - max_distance_m: Furthest serving distance
                - total_grids: Total number of bins served
                - edge_grids: Number of edge bins
                - avg_edge_rsrp: Average RSRP in edge bins
        """
        # Overall cell metrics
        cell_overall = grid_df.groupby('cell_name').agg({
            'distance_m': 'max',
            'grid': 'nunique',
            'avg_rsrp': 'max',  # Max RSRP across ALL grids (for degradation check)
        }).reset_index()
        cell_overall.columns = ['cell_name', 'max_distance_m', 'total_grids', 'cell_max_rsrp']

        # Edge metrics
        cell_edge = edge_bins.groupby('cell_name').agg({
            'grid': 'nunique',
            'avg_rsrp': 'mean',
        }).reset_index()
        cell_edge.columns = ['cell_name', 'edge_grids', 'avg_edge_rsrp']

        # Merge
        cell_metrics = cell_overall.merge(cell_edge, on='cell_name', how='left')

        # Fill NaN for cells with no edge bins
        cell_metrics['edge_grids'] = cell_metrics['edge_grids'].fillna(0).astype(int)
        cell_metrics['avg_edge_rsrp'] = cell_metrics['avg_edge_rsrp'].fillna(self.params.default_rsrp_dbm)

        return cell_metrics

    def _apply_overshooting_filters(
        self,
        cell_metrics: pd.DataFrame,
        edge_bins: pd.DataFrame,
        full_grid_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply filtering criteria to identify overshooting candidates.

        Filters based on:
        1. Distance criteria (min_cell_distance)
        2. Grid bin criteria (min_cell_count_in_grid, max_percentage_grid_events)
        3. Relative reach criteria (compare against all cells serving grid, not just edge)
        4. RSRP degradation criteria
        5. Final thresholds (min_overshooting_grids, percentage_overshooting_grids)

        Args:
            cell_metrics: Per-cell aggregated metrics
            edge_bins: Grid bins at the edge of each cell's coverage
            full_grid_df: Full grid data (for computing true max distance per grid)

        Returns:
            Tuple of (overshooters, overshooting_bins)
        """
        # Start with all cells
        candidates = cell_metrics.copy()

        # Filter 1: Cell must serve traffic beyond min_cell_distance
        candidates = candidates[
            candidates['max_distance_m'] >= self.params.min_cell_distance
        ]
        logger.info(
            "After distance filter",
            candidates=len(candidates),
            min_distance=self.params.min_cell_distance,
        )

        # Early return if no candidates or no edge bins
        if len(candidates) == 0 or len(edge_bins) == 0:
            logger.info("No overshooting candidates found", candidates=len(candidates), edge_bins=len(edge_bins))
            return pd.DataFrame(), pd.DataFrame()

        # Filter 2: Calculate per-grid-bin cell count and event percentage
        # Group by geohash AND frequency band to find bins served by multiple cells
        # CRITICAL: Only count cells in the same frequency band as competition

        # Check if Band column exists (from cell_coverage.csv schema)
        band_col = 'Band' if 'Band' in edge_bins.columns else None

        # RSRP-based competition counting (vectorized)
        # Step 1: Find reference RSRP per grid using configurable quantile (per band if available)
        # Using quantile instead of max to be robust against outliers
        rsrp_quantile = self.params.rsrp_competition_quantile
        if band_col is not None:
            p90_rsrp_per_grid = edge_bins.groupby(['grid', band_col])['avg_rsrp'].quantile(rsrp_quantile)
            grid_df = edge_bins.copy()
            grid_df['p90_rsrp_in_grid'] = grid_df.set_index(['grid', band_col]).index.map(p90_rsrp_per_grid)
        else:
            p90_rsrp_per_grid = edge_bins.groupby('grid')['avg_rsrp'].quantile(rsrp_quantile).to_dict()
            grid_df = edge_bins.copy()
            grid_df['p90_rsrp_in_grid'] = grid_df['grid'].map(p90_rsrp_per_grid)

        # Step 2: Calculate RSRP difference and flag competing cells
        grid_df['rsrp_diff'] = grid_df['p90_rsrp_in_grid'] - grid_df['avg_rsrp']
        grid_df['is_competing'] = grid_df['rsrp_diff'] <= self.params.interference_threshold_db

        # Step 3: Count competing cells per grid (unique cells, not rows)
        if band_col is not None:
            # Total unique cells per grid
            total_cells = grid_df.groupby(['grid', band_col])['cell_name'].nunique().reset_index()
            total_cells.columns = ['grid', band_col, 'total_cells']

            # Competing unique cells per grid (only where is_competing == True)
            competing_cells = grid_df[grid_df['is_competing']].groupby(['grid', band_col])['cell_name'].nunique().reset_index()
            competing_cells.columns = ['grid', band_col, 'competing_cells']

            # Total traffic per grid
            traffic = grid_df.groupby(['grid', band_col])['event_count'].sum().reset_index()
            traffic.columns = ['grid', band_col, 'total_grid_traffic']

            # Combine counts
            competing_counts = total_cells.merge(competing_cells, on=['grid', band_col], how='left')
            competing_counts = competing_counts.merge(traffic, on=['grid', band_col], how='left')
            competing_counts['competing_cells'] = competing_counts['competing_cells'].fillna(0).astype(int)

            # Merge into grid_df (not edge_bins) to preserve p90_rsrp_in_grid column
            edge_with_counts = grid_df.merge(
                competing_counts,
                on=['grid', band_col],
                how='left'
            )

            logger.info(
                "Band-aware RSRP-based competition filter enabled",
                band_column=band_col,
                unique_bands=edge_bins[band_col].nunique(),
            )
        else:
            # Total unique cells per grid
            total_cells = grid_df.groupby('grid')['cell_name'].nunique().reset_index()
            total_cells.columns = ['grid', 'total_cells']

            # Competing unique cells per grid (only where is_competing == True)
            competing_cells = grid_df[grid_df['is_competing']].groupby('grid')['cell_name'].nunique().reset_index()
            competing_cells.columns = ['grid', 'competing_cells']

            # Total traffic per grid
            traffic = grid_df.groupby('grid')['event_count'].sum().reset_index()
            traffic.columns = ['grid', 'total_grid_traffic']

            # Combine counts
            competing_counts = total_cells.merge(competing_cells, on='grid', how='left')
            competing_counts = competing_counts.merge(traffic, on='grid', how='left')
            competing_counts['competing_cells'] = competing_counts['competing_cells'].fillna(0).astype(int)

            # Merge into grid_df (not edge_bins) to preserve p90_rsrp_in_grid column
            edge_with_counts = grid_df.merge(competing_counts, on='grid', how='left')

            logger.warning(
                "No frequency band column found - using band-agnostic RSRP competition",
                available_columns=list(edge_bins.columns),
            )

        # Step 4: Calculate cell traffic percentage (guard against division by zero)
        edge_with_counts['cell_traffic_pct'] = (
            edge_with_counts['event_count'] /
            edge_with_counts['total_grid_traffic'].replace(0, np.nan)
        ).fillna(0)

        # Step 5: Filter using competing_cells instead of cells_in_grid
        competition_bins = edge_with_counts[
            (edge_with_counts['competing_cells'] >= self.params.min_cell_count_in_grid) &
            (edge_with_counts['cell_traffic_pct'] <= self.params.max_percentage_grid_events)
        ].copy()

        logger.info(
            "RSRP-based competition filter applied",
            band_aware=band_col is not None,
            edge_bins=len(edge_bins),
            competition_bins=len(competition_bins),
            avg_competing_cells=edge_with_counts['competing_cells'].mean(),
            avg_total_cells=edge_with_counts['total_cells'].mean() if 'total_cells' in edge_with_counts.columns else None,
        )

        # NEW: Filter 3b - Relative distance criterion
        # Calculate max distance any cell reaches to each grid bin
        # Use full_grid_df (not just edge_bins) to include all cells serving each grid
        if band_col is not None:
            grid_max_dist = full_grid_df.groupby(['grid', band_col])['distance_m'].max().reset_index()
            grid_max_dist.columns = ['grid', band_col, 'grid_max_distance']
            competition_bins = competition_bins.merge(
                grid_max_dist,
                on=['grid', band_col],
                how='left'
            )
        else:
            grid_max_dist = full_grid_df.groupby('grid')['distance_m'].max().reset_index()
            grid_max_dist.columns = ['grid', 'grid_max_distance']
            competition_bins = competition_bins.merge(
                grid_max_dist,
                on='grid',
                how='left'
            )

        # Calculate relative reach: how far is THIS cell reaching compared to furthest?
        competition_bins['relative_reach'] = (
            competition_bins['distance_m'] / competition_bins['grid_max_distance']
        )

        # Apply relative distance filter
        overshooting_bins = competition_bins[
            competition_bins['relative_reach'] >= self.params.min_relative_reach
        ].copy()

        logger.info(
            "After relative distance filter",
            competition_bins=len(competition_bins),
            overshooting_bins=len(overshooting_bins),
            min_relative_reach=self.params.min_relative_reach,
            reduction_pct=((len(competition_bins) - len(overshooting_bins)) / len(competition_bins) * 100) if len(competition_bins) > 0 else 0,
        )

        # Filter 4: RSRP degradation check
        # For each cell, check if edge RSRP is degraded compared to cell's max RSRP
        # Use cell_max_rsrp from cell_metrics (computed from ALL grids, not just edge)
        cell_max_rsrp = cell_metrics[['cell_name', 'cell_max_rsrp']]

        # Join max RSRP back to overshooting bins
        overshooting_bins = overshooting_bins.merge(
            cell_max_rsrp,
            on='cell_name',
            how='left'
        )

        # Calculate RSRP degradation threshold for each cell
        # Example: if cell_max_rsrp = -70 dBm and rsrp_degradation_db = 10
        # then edge_threshold = -70 - 10 = -80 dBm
        overshooting_bins['edge_rsrp_threshold'] = (
            overshooting_bins['cell_max_rsrp'] - self.params.rsrp_degradation_db
        )

        # Keep only bins where RSRP is degraded (more negative than threshold)
        bins_before_rsrp_filter = len(overshooting_bins)
        overshooting_bins = overshooting_bins[
            overshooting_bins['avg_rsrp'] <= overshooting_bins['edge_rsrp_threshold']
        ].copy()

        logger.info(
            "After RSRP degradation filter",
            overshooting_bins=len(overshooting_bins),
            bins_before=bins_before_rsrp_filter,
            rsrp_degradation_db=self.params.rsrp_degradation_db,
            reduction_pct=((bins_before_rsrp_filter - len(overshooting_bins)) / bins_before_rsrp_filter * 100) if bins_before_rsrp_filter > 0 else 0,
        )

        # Count overshooting bins per cell
        overshooting_per_cell = overshooting_bins.groupby('cell_name').agg({
            'grid': 'nunique',
        }).reset_index()
        overshooting_per_cell.columns = ['cell_name', 'overshooting_grids']

        # Merge with cell metrics
        candidates = candidates.merge(overshooting_per_cell, on='cell_name', how='left')
        candidates['overshooting_grids'] = candidates['overshooting_grids'].fillna(0).astype(int)

        # Calculate percentage
        candidates['percentage_overshooting'] = (
            candidates['overshooting_grids'] / candidates['total_grids']
        )

        # Filter 5: Final thresholds
        overshooters = candidates[
            (candidates['overshooting_grids'] >= self.params.min_overshooting_grids) &
            (candidates['percentage_overshooting'] >= self.params.percentage_overshooting_grids)
        ].copy()

        logger.info(
            "After final thresholds",
            overshooters=len(overshooters),
            min_grids=self.params.min_overshooting_grids,
            min_percentage=self.params.percentage_overshooting_grids,
        )

        return overshooters, overshooting_bins

    def _calculate_tilt_recommendations(
        self,
        overshooters: pd.DataFrame,
        overshooting_bins: pd.DataFrame,
        gis_df: pd.DataFrame,
        grid_with_distance: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate data-driven downtilt recommendations for overshooting cells.

        Uses 3GPP antenna patterns and log-distance path loss to predict:
        1. RSRP reduction at each overshooting grid
        2. Number of grids that will be resolved (out of reach or exit competition)
        3. New maximum serving distance from actual remaining grids (data-driven)

        Recommends 1° or 2° downtilt based on resolution effectiveness.
        """
        # Early return if no overshooters
        if len(overshooters) == 0:
            logger.info("No overshooters to calculate recommendations for")
            return pd.DataFrame()

        # RF propagation parameters (from config)
        HPBW_V_DEG = self.params.hpbw_v_deg
        SLA_V_DB = self.params.sla_v_db
        PATH_LOSS_EXP = self.params.path_loss_exponent
        DEFAULT_HEIGHT_M = self.params.default_antenna_height_m
        CELL_EDGE_RSRP_THRESHOLD = self.params.cell_edge_rsrp_threshold_dbm

        # Merge with GIS data to get current tilt and antenna height (single merge)
        # Using cell_gis.csv schema: tilt_elc, tilt_mech, antenna_height
        gis_cols = ['cell_name', 'tilt_mech', 'tilt_elc', 'latitude', 'longitude', 'antenna_height']
        gis_cols_available = [col for col in gis_cols if col in gis_df.columns]

        overshooters_with_gis = overshooters.merge(
            gis_df[gis_cols_available],
            on='cell_name',
            how='left'
        )

        # Set antenna height with default fallback
        if 'antenna_height' in overshooters_with_gis.columns:
            overshooters_with_gis['antenna_height_m'] = overshooters_with_gis['antenna_height'].fillna(DEFAULT_HEIGHT_M)
        else:
            overshooters_with_gis['antenna_height_m'] = DEFAULT_HEIGHT_M
            logger.warning(f"No antenna_height column found in GIS data, using default {DEFAULT_HEIGHT_M}m")

        # OPTIMIZATION: Vectorized calculation of RSRP reduction
        # Pre-calculate vertical attenuation for all grids at once

        def vectorized_rsrp_reduction(distance_m_arr, alpha_deg, h_m, delta_tilt_deg):
            """Vectorized RSRP reduction calculation for arrays."""
            # Elevation angle from site to grids
            theta_deg = np.degrees(np.arctan2(h_m, distance_m_arr))

            # 3GPP vertical attenuation before downtilt
            A_before = np.minimum(
                12.0 * (((theta_deg - alpha_deg) / HPBW_V_DEG) ** 2),
                SLA_V_DB
            )
            # 3GPP vertical attenuation after downtilt
            A_after = np.minimum(
                12.0 * (((theta_deg - (alpha_deg + delta_tilt_deg)) / HPBW_V_DEG) ** 2),
                SLA_V_DB
            )
            # RSRP reduction = increase in attenuation (non-negative)
            return np.maximum(A_after - A_before, 0.0)

        # For each overshooting cell, calculate tilt recommendation
        recommendations = []

        # Pre-index data for faster lookup
        overshooting_bins_indexed = overshooting_bins.set_index('cell_name')
        grid_with_distance_indexed = grid_with_distance.set_index('cell_name')

        for _, cell_row in overshooters_with_gis.iterrows():
            cell_id = cell_row['cell_name']
            current_tilt_deg = cell_row.get('tilt_elc', 0.0)
            if pd.isna(current_tilt_deg):
                current_tilt_deg = 0.0

            h_m = cell_row['antenna_height_m']
            d_max_m = cell_row['max_distance_m']

            # Get overshooting grids for this cell using index
            try:
                cell_overshoot_bins = overshooting_bins_indexed.loc[[cell_id]].copy()
            except KeyError:
                cell_overshoot_bins = pd.DataFrame()

            if len(cell_overshoot_bins) == 0:
                # No overshooting bins, shouldn't happen but handle gracefully
                recommendations.append({
                    'cell_name': cell_id,
                    'recommended_tilt_change': 0,
                    'grids_resolved_1deg': 0,
                    'grids_resolved_2deg': 0,
                    'new_max_distance_1deg_m': d_max_m,
                    'new_max_distance_2deg_m': d_max_m,
                })
                continue

            # Calculate new max distances after downtilt
            d_max_1deg, _ = self._estimate_distance_after_downtilt(
                d_max_m, current_tilt_deg, h_m, delta_tilt_deg=1.0,
                hpbw_v_deg=HPBW_V_DEG, sla_v_db=SLA_V_DB, path_loss_exp=PATH_LOSS_EXP
            )
            d_max_2deg, _ = self._estimate_distance_after_downtilt(
                d_max_m, current_tilt_deg, h_m, delta_tilt_deg=2.0,
                hpbw_v_deg=HPBW_V_DEG, sla_v_db=SLA_V_DB, path_loss_exp=PATH_LOSS_EXP
            )

            # OPTIMIZATION: Vectorized grid resolution calculation
            grid_dist_arr = cell_overshoot_bins['distance_m'].values
            grid_rsrp_arr = cell_overshoot_bins['avg_rsrp'].values
            grid_p90_rsrp_arr = cell_overshoot_bins['p90_rsrp_in_grid'].values if 'p90_rsrp_in_grid' in cell_overshoot_bins.columns else grid_rsrp_arr

            # Calculate RSRP reductions vectorized
            rsrp_reduction_1deg = vectorized_rsrp_reduction(grid_dist_arr, current_tilt_deg, h_m, 1.0)
            rsrp_reduction_2deg = vectorized_rsrp_reduction(grid_dist_arr, current_tilt_deg, h_m, 2.0)

            # New RSRP after downtilt
            new_rsrp_1deg = grid_rsrp_arr - rsrp_reduction_1deg
            new_rsrp_2deg = grid_rsrp_arr - rsrp_reduction_2deg

            # RSRP diff from P90
            rsrp_diff_1deg = grid_p90_rsrp_arr - new_rsrp_1deg
            rsrp_diff_2deg = grid_p90_rsrp_arr - new_rsrp_2deg

            # Resolution: out of reach OR exits competition
            resolved_1deg_mask = (grid_dist_arr > d_max_1deg) | (rsrp_diff_1deg > self.params.interference_threshold_db)
            resolved_2deg_mask = (grid_dist_arr > d_max_2deg) | (rsrp_diff_2deg > self.params.interference_threshold_db)

            resolved_1deg = np.sum(resolved_1deg_mask)
            resolved_2deg = np.sum(resolved_2deg_mask)

            # Decision logic: recommend tilt based on resolution effectiveness
            total_overshoot_bins = len(cell_overshoot_bins)
            resolution_1deg_pct = resolved_1deg / total_overshoot_bins if total_overshoot_bins > 0 else 0
            resolution_2deg_pct = resolved_2deg / total_overshoot_bins if total_overshoot_bins > 0 else 0

            # Use config threshold for tilt decision
            min_resolution_pct = self.params.min_resolution_pct_for_1deg
            if resolution_1deg_pct >= min_resolution_pct:
                recommended_tilt = 1
            elif resolution_2deg_pct >= min_resolution_pct:
                recommended_tilt = 2
            else:
                recommended_tilt = 2

            # OPTIMIZATION: Vectorized new max distance calculation
            try:
                all_cell_grids = grid_with_distance_indexed.loc[[cell_id]].copy()
            except KeyError:
                all_cell_grids = pd.DataFrame()

            if len(all_cell_grids) > 0:
                all_dist_arr = all_cell_grids['distance_m'].values
                all_rsrp_arr = all_cell_grids['avg_rsrp'].values

                # Calculate RSRP after downtilt for all grids
                all_reduction_1deg = vectorized_rsrp_reduction(all_dist_arr, current_tilt_deg, h_m, 1.0)
                all_reduction_2deg = vectorized_rsrp_reduction(all_dist_arr, current_tilt_deg, h_m, 2.0)

                new_rsrp_all_1deg = all_rsrp_arr - all_reduction_1deg
                new_rsrp_all_2deg = all_rsrp_arr - all_reduction_2deg

                # Grids remaining served (above threshold)
                remaining_1deg_mask = new_rsrp_all_1deg >= CELL_EDGE_RSRP_THRESHOLD
                remaining_2deg_mask = new_rsrp_all_2deg >= CELL_EDGE_RSRP_THRESHOLD

                actual_new_max_1deg = all_dist_arr[remaining_1deg_mask].max() if remaining_1deg_mask.any() else 0
                actual_new_max_2deg = all_dist_arr[remaining_2deg_mask].max() if remaining_2deg_mask.any() else 0
            else:
                actual_new_max_1deg = 0
                actual_new_max_2deg = 0

            recommendations.append({
                'cell_name': cell_id,
                'recommended_tilt_change': recommended_tilt,
                'grids_resolved_1deg': int(resolved_1deg),
                'grids_resolved_2deg': int(resolved_2deg),
                'resolution_1deg_pct': resolution_1deg_pct,
                'resolution_2deg_pct': resolution_2deg_pct,
                'new_max_distance_1deg_m': actual_new_max_1deg,
                'new_max_distance_2deg_m': actual_new_max_2deg,
            })

        # Convert to DataFrame and merge back
        recommendations_df = pd.DataFrame(recommendations)

        # Handle edge case where no overshooters found
        if len(recommendations_df) == 0:
            logger.warning("No overshooting cells found - returning empty DataFrame")
            return pd.DataFrame()

        overshooters_with_gis = overshooters_with_gis.merge(
            recommendations_df,
            on='cell_name',
            how='left'
        )

        # Add interference reduction metrics for user-requested analysis
        # Current state
        overshooters_with_gis['current_interference_grids'] = overshooters_with_gis['overshooting_grids']
        overshooters_with_gis['current_interference_pct'] = overshooters_with_gis['percentage_overshooting']

        # Removed grids based on recommended tilt (vectorized)
        overshooters_with_gis['removed_interference_grids'] = np.select(
            [
                overshooters_with_gis['recommended_tilt_change'] == 1,
                overshooters_with_gis['recommended_tilt_change'] == 2,
            ],
            [
                overshooters_with_gis['grids_resolved_1deg'],
                overshooters_with_gis['grids_resolved_2deg'],
            ],
            default=0
        )

        # New state after tilt
        overshooters_with_gis['new_interference_grids'] = (
            overshooters_with_gis['current_interference_grids'] -
            overshooters_with_gis['removed_interference_grids']
        )
        overshooters_with_gis['new_interference_pct'] = (
            overshooters_with_gis['new_interference_grids'] /
            overshooters_with_gis['total_grids']
        )

        # Reduction achieved (guard against division by zero)
        overshooters_with_gis['interference_reduction_pct'] = (
            overshooters_with_gis['removed_interference_grids'] /
            overshooters_with_gis['current_interference_grids'].replace(0, np.nan)
        ).fillna(0) * 100

        return overshooters_with_gis

    def _estimate_distance_after_downtilt(
        self,
        d_max_m: float,
        alpha_deg: float,
        h_m: float,
        delta_tilt_deg: float,
        hpbw_v_deg: float,
        sla_v_db: float,
        path_loss_exp: float
    ) -> Tuple[float, float]:
        """
        Estimate new max coverage distance after downtilt.

        Downtilt reduces antenna gain at the horizon, shortening maximum reach.
        Uses 3GPP antenna pattern and log-distance path loss model.

        Parameters:
            d_max_m: Current maximum serving distance (meters)
            alpha_deg: Current electrical downtilt (degrees)
            h_m: Antenna height above UE (meters)
            delta_tilt_deg: Tilt change (positive for downtilt)
            hpbw_v_deg: Vertical half-power beamwidth
            sla_v_db: Side-lobe attenuation cap
            path_loss_exp: Path loss exponent

        Returns:
            Tuple of (new_distance_m, reduction_pct)
        """
        if d_max_m <= 0 or h_m < 0:
            return d_max_m, 0.0

        # Elevation angle from site to current edge user (at horizon)
        theta_e_deg = math.degrees(math.atan2(h_m, d_max_m))

        # 3GPP vertical attenuation before/after downtilt
        A_before = self._vertical_attenuation(theta_e_deg, alpha_deg, hpbw_v_deg, sla_v_db)
        A_after = self._vertical_attenuation(theta_e_deg, alpha_deg + delta_tilt_deg, hpbw_v_deg, sla_v_db)

        # Gain change at horizon (dB) - downtilt increases attenuation (negative gain)
        deltaG_dB = -(A_after - A_before)

        # Translate to distance using log-distance model
        d_new_m = d_max_m * (10.0 ** (deltaG_dB / (10.0 * path_loss_exp)))

        # For downtilt, ensure distance decreases (or stays same)
        if d_new_m > d_max_m:
            d_new_m = d_max_m

        reduction_pct = (d_max_m - d_new_m) / d_max_m if d_max_m > 0 else 0.0

        return d_new_m, reduction_pct

    def _vertical_attenuation(
        self,
        theta_deg: float,
        alpha_deg: float,
        hpbw_v_deg: float,
        sla_v_db: float
    ) -> float:
        """
        3GPP parabolic attenuation in vertical plane (dB).

        Parameters:
            theta_deg: Elevation angle to user (degrees)
            alpha_deg: Antenna downtilt angle (degrees)
            hpbw_v_deg: Vertical half-power beamwidth (degrees)
            sla_v_db: Side-lobe attenuation cap (dB)

        Returns:
            Attenuation in dB
        """
        return min(
            12.0 * (((theta_deg - alpha_deg) / hpbw_v_deg) ** 2),
            sla_v_db
        )

    def _calculate_severity_scores(
        self,
        overshooters: pd.DataFrame,
        grid_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate multi-factor severity scores (0-1) for overshooting cells.

        Uses balanced weighting:
        - Overshooting bins (absolute): 30%
        - Percentage overshooting: 25%
        - Max distance: 20%
        - RSRP degradation: 15%
        - Traffic impact: 10%

        Optional band multiplier:
        - 800 MHz: 1.1x (more critical)
        - 1800 MHz: 1.0x (baseline)
        - 700 MHz: 1.05x
        - 2100 MHz: 0.9x (less critical)

        Args:
            overshooters: DataFrame with overshooting cells
            grid_df: Original grid data (for band lookup)

        Returns:
            DataFrame with added columns:
                - severity_score: 0-1 normalized score
                - severity_category: CRITICAL/HIGH/MEDIUM/LOW/MINIMAL
        """
        if len(overshooters) == 0:
            overshooters['severity_score'] = []
            overshooters['severity_category'] = []
            return overshooters

        # Calculate dataset statistics for normalization using percentiles (robust to outliers)
        bins_p5 = overshooters['overshooting_grids'].quantile(0.05)
        bins_p95 = overshooters['overshooting_grids'].quantile(0.95)
        grids_p5 = overshooters['total_grids'].quantile(0.05)
        grids_p95 = overshooters['total_grids'].quantile(0.95)

        # Guard against division by zero when all values are the same
        bins_range = max(bins_p95 - bins_p5, 1e-9)
        grids_range = max(grids_p95 - grids_p5, 1e-9)

        logger.info(
            "Calculating severity scores (percentile-based normalization)",
            bins_p5_p95=f"{bins_p5:.0f}-{bins_p95:.0f}",
            grids_p5_p95=f"{grids_p5:.0f}-{grids_p95:.0f}",
        )

        # 1. Normalize overshooting_grids (0-1) using 5th-95th percentile
        bins_score = (overshooters['overshooting_grids'] - bins_p5) / bins_range
        bins_score = bins_score.clip(0, 1)  # Clip outliers to 0-1 range

        # 2. Percentage overshooting is already 0-1
        percentage_score = overshooters['percentage_overshooting']

        # 3. Normalize max_distance_m (0-1) using config values
        dist_min = self.params.severity_distance_min_m
        dist_max = self.params.severity_distance_max_m
        distance_score = (overshooters['max_distance_m'] - dist_min) / (dist_max - dist_min)
        distance_score = distance_score.clip(0, 1)

        # 4. RSRP degradation (0-1, inverted - worse RSRP = higher score)
        # Invert so that worst_rsrp = 1.0, best_rsrp = 0.0
        rsrp_best = self.params.severity_rsrp_best_dbm
        rsrp_worst = self.params.severity_rsrp_worst_dbm
        rsrp_score = (overshooters['avg_edge_rsrp'] - rsrp_best) / (rsrp_worst - rsrp_best)
        rsrp_score = rsrp_score.clip(0, 1)

        # 5. Traffic impact (normalize by dataset using 5th-95th percentile)
        traffic_score = (overshooters['total_grids'] - grids_p5) / grids_range
        traffic_score = traffic_score.clip(0, 1)  # Clip outliers to 0-1 range

        # Weighted combination using config weights
        severity = (
            self.params.severity_weight_bins * bins_score +
            self.params.severity_weight_percentage * percentage_score +
            self.params.severity_weight_distance * distance_score +
            self.params.severity_weight_rsrp * rsrp_score +
            self.params.severity_weight_traffic * traffic_score
        )

        # Clip to 0-1 range
        overshooters['severity_score'] = severity.clip(0, 1)

        # Categorize severity using config thresholds
        severity_bins = [
            -np.inf,
            self.params.severity_threshold_low,
            self.params.severity_threshold_medium,
            self.params.severity_threshold_high,
            self.params.severity_threshold_critical,
            np.inf
        ]
        overshooters['severity_category'] = pd.cut(
            overshooters['severity_score'],
            bins=severity_bins,
            labels=['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )

        logger.info(
            "Severity scores calculated",
            avg_score=overshooters['severity_score'].mean(),
            min_score=overshooters['severity_score'].min(),
            max_score=overshooters['severity_score'].max(),
        )

        return overshooters


def detect_overshooting_cells(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    params: OvershooterParams | None = None,
) -> pd.DataFrame:
    """
    Convenience function to detect overshooting cells.

    Args:
        grid_df: Grid measurement data
        gis_df: Cell GIS data
        params: Detection parameters (uses defaults if None)

    Returns:
        DataFrame of overshooting cells with recommendations

    Example:
        >>> from ran_optimizer.data import load_grid_data, load_gis_data
        >>> grid_df = load_grid_data("data/grid.csv", "Vodafone_Ireland")
        >>> gis_df = load_gis_data("data/gis.csv", "Vodafone_Ireland")
        >>>
        >>> overshooters = detect_overshooting_cells(grid_df, gis_df)
        >>> print(f"Found {len(overshooters)} overshooting cells")
        >>> print(overshooters[['cell_name', 'overshooting_grids', 'recommended_tilt_change']])
    """
    if params is None:
        params = OvershooterParams()

    detector = OvershooterDetector(params)
    return detector.detect(grid_df, gis_df)
