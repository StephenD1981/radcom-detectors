"""
Undershooting cell detection for RAN optimization.

Identifies cells with insufficient coverage in areas with low interference,
which are candidates for up-tilting to extend their reach.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import math

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.overshooting_config import (
    load_overshooting_config,
    get_environment_params,
    get_default_config_path,
)

logger = get_logger(__name__)


@dataclass
class UndershooterParams:
    """
    Parameters for undershooting cell detection.

    Undershooting detection identifies cells that aren't covering their intended
    area adequately - candidates for up-tilting to extend reach.
    """
    # Traffic filter
    min_cell_event_count: int = 200

    # Distance filter
    max_cell_distance: float = 15000.0  # 15km

    # Interference thresholds
    interference_threshold_db: float = 7.5
    max_cell_grid_count: int = 4
    max_interference_percentage: float = 0.20
    rsrp_competition_quantile: float = 0.90  # P90 for reference RSRP

    # Coverage increase thresholds
    min_coverage_increase_1deg: float = 0.05
    min_coverage_increase_2deg: float = 0.10
    min_distance_gain_1deg_m: float = 50.0
    min_distance_gain_2deg_m: float = 100.0
    min_new_grids_1deg: int = 5
    min_new_grids_2deg: int = 10

    # RF propagation parameters
    hpbw_v_deg: float = 6.5
    sla_v_db: float = 30.0
    path_loss_exponent: float = 3.5

    # Severity score weights
    severity_weight_coverage: float = 0.30
    severity_weight_new_grids: float = 0.25
    severity_weight_low_interference: float = 0.20
    severity_weight_distance: float = 0.15
    severity_weight_traffic: float = 0.10

    # Severity normalization
    severity_max_coverage_increase_pct: float = 0.50  # 50% increase = score of 1.0

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
    ) -> 'UndershooterParams':
        """
        Create UndershooterParams from JSON configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to undershooting_params.json. If None, uses default config.
        environment : str, optional
            Environment type: 'urban', 'suburban', or 'rural'.
            If None, uses default parameters.

        Returns
        -------
        UndershooterParams
            Parameter object with values loaded from config.

        Examples
        --------
        >>> # Load default parameters
        >>> params = UndershooterParams.from_config()

        >>> # Load urban-specific parameters
        >>> params_urban = UndershooterParams.from_config(environment='urban')

        >>> # Load from custom config file
        >>> params_custom = UndershooterParams.from_config('config/operator_xyz.json')
        """
        if config_path is None:
            config_path = str(Path(get_default_config_path()).parent / 'undershooting_params.json')

        logger.info("Loading configuration", config_file=config_path)
        config = load_overshooting_config(str(config_path))

        logger.info(
            "Configuration loaded",
            version=config.get('version', 'unknown'),
            description=config.get('description', 'No description')
        )

        params_dict = get_environment_params(config, environment)

        # Flatten nested config sections into dataclass field names
        # severity_weights section
        if 'severity_weights' in params_dict:
            sw = params_dict['severity_weights']
            params_dict['severity_weight_coverage'] = sw.get('coverage_increase', 0.30)
            params_dict['severity_weight_new_grids'] = sw.get('new_grids', 0.25)
            params_dict['severity_weight_low_interference'] = sw.get('low_interference', 0.20)
            params_dict['severity_weight_distance'] = sw.get('distance_gain', 0.15)
            params_dict['severity_weight_traffic'] = sw.get('traffic_potential', 0.10)

        # severity_normalization section
        if 'severity_normalization' in params_dict:
            sn = params_dict['severity_normalization']
            params_dict['severity_max_coverage_increase_pct'] = sn.get('max_coverage_increase_pct', 0.50)

        # severity_thresholds section
        if 'severity_thresholds' in params_dict:
            st = params_dict['severity_thresholds']
            params_dict['severity_threshold_critical'] = st.get('critical', 0.80)
            params_dict['severity_threshold_high'] = st.get('high', 0.60)
            params_dict['severity_threshold_medium'] = st.get('medium', 0.40)
            params_dict['severity_threshold_low'] = st.get('low', 0.20)

        # Extract only parameters belonging to UndershooterParams
        valid_params = {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name in params_dict:
                valid_params[field_name] = params_dict[field_name]

        logger.info(
            "Created UndershooterParams from config",
            config_file=config_path,
            environment=environment or 'default',
            parameters_loaded=len(valid_params)
        )

        return cls(**valid_params)


class UndershooterDetector:
    """
    Detector for cells with insufficient coverage (undershooting).

    Identifies cells that aren't reaching far enough and could benefit from
    up-tilting to extend their coverage, with acceptable interference impact.

    Parameters
    ----------
    params : UndershooterParams
        Detection parameters

    Examples
    --------
    >>> from ran_optimizer.recommendations import UndershooterDetector, UndershooterParams
    >>>
    >>> # Create detector with default parameters
    >>> params = UndershooterParams()
    >>> detector = UndershooterDetector(params)
    >>>
    >>> # Run detection
    >>> undershooters = detector.detect(grid_df, gis_df)
    """

    def __init__(self, params: UndershooterParams):
        """Initialize detector with parameters."""
        self.params = params
        logger.info(
            "Initialized UndershooterDetector",
            max_cell_distance=params.max_cell_distance,
            min_cell_event_count=params.min_cell_event_count,
            max_interference_percentage=params.max_interference_percentage
        )

    def _sanitise_inputs(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sanitise input DataFrames by removing invalid rows.

        Removes:
        - Rows with null cell_name
        - Rows with invalid coordinates
        - Rows with null/negative distance values (grid_df)
        - Rows with null RSRP values (grid_df)
        """
        # Sanitise grid_df
        grid_clean = grid_df.copy()
        initial_grid_rows = len(grid_clean)

        # Remove nulls in critical columns
        grid_clean = grid_clean.dropna(subset=['cell_name', 'grid', 'avg_rsrp', 'distance_to_cell'])

        # Remove invalid distances
        grid_clean = grid_clean[grid_clean['distance_to_cell'] > 0]

        # Ensure cell_name is string
        grid_clean['cell_name'] = grid_clean['cell_name'].astype(str)

        if len(grid_clean) < initial_grid_rows:
            logger.info(
                "Sanitised grid_df",
                removed_rows=initial_grid_rows - len(grid_clean),
                remaining_rows=len(grid_clean)
            )

        # Sanitise gis_df
        gis_clean = gis_df.copy()
        initial_gis_rows = len(gis_clean)

        # Remove nulls in critical columns
        gis_clean = gis_clean.dropna(subset=['cell_name', 'latitude', 'longitude'])

        # Ensure cell_name is string
        gis_clean['cell_name'] = gis_clean['cell_name'].astype(str)

        if len(gis_clean) < initial_gis_rows:
            logger.info(
                "Sanitised gis_df",
                removed_rows=initial_gis_rows - len(gis_clean),
                remaining_rows=len(gis_clean)
            )

        return grid_clean, gis_clean

    def detect(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect undershooting cells.

        Parameters
        ----------
        grid_df : pd.DataFrame
            Grid-level coverage data with columns:
            - cell_name: Cell identifier
            - grid: Grid identifier (geohash7)
            - distance_to_cell: Distance from cell to grid (meters)
            - event_count: Traffic volume in grid
            - avg_rsrp: Average RSRP measurement (dBm)

        gis_df : pd.DataFrame
            Cell GIS data with columns:
            - cell_name: Cell identifier
            - latitude, longitude: Cell location
            - azimuth_deg: Cell bearing
            - mechanical_tilt, electrical_tilt: Tilt values (degrees)
            - height: Antenna height (meters)

        Returns
        -------
        pd.DataFrame
            Undershooting cells with recommended tilt adjustments.
            Columns include:
            - cell_name: Cell identifier
            - max_distance_m: Current maximum serving distance
            - total_grids: Number of grids served
            - interference_grids: Number of high-competition grids
            - interference_percentage: Fraction of interfering grids
            - total_traffic: Total traffic volume (sum of event_count)
            - tilt_mech, tilt_elc: Current mechanical and electrical tilt
            - recommended_uptilt_deg: Recommended uptilt (1 or 2 degrees)
            - new_max_distance_m: Predicted distance after uptilt
            - coverage_increase_percentage: Expected coverage increase
            - new_coverage_grids: Estimated new grids covered
        """
        # Sanitise inputs
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        logger.info(
            "Starting undershooting detection",
            cells=gis_df['cell_name'].nunique(),
            grid_measurements=len(grid_df)
        )

        # Step 1: Filter to cells with short range
        logger.info("Step 1: Filtering cells by maximum serving distance")
        candidates = self._filter_by_distance(grid_df)

        if len(candidates) == 0:
            logger.warning("No cells found with max distance <= threshold")
            return pd.DataFrame()

        # Step 2: Calculate interference metrics
        logger.info("Step 2: Calculating interference metrics")
        candidates = self._calculate_interference(candidates, grid_df)

        # Step 3: Filter by traffic and interference
        logger.info("Step 3: Filtering by traffic and interference thresholds")
        candidates = self._filter_candidates(candidates)

        if len(candidates) == 0:
            logger.warning("No candidates passed traffic/interference filters")
            return pd.DataFrame()

        # Step 4: Merge with GIS data
        logger.info("Step 4: Merging with cell GIS data")
        candidates = candidates.merge(
            gis_df[['cell_name', 'latitude', 'longitude', 'bearing',
                   'tilt_mech', 'tilt_elc', 'antenna_height']],
            on='cell_name',
            how='inner'
        )

        # Step 5: Calculate uptilt impact
        logger.info("Step 5: Calculating uptilt impact (1° and 2°)")
        candidates = self._calculate_uptilt_impact(candidates)

        # Step 6: Select best uptilt recommendation
        logger.info("Step 6: Selecting optimal uptilt recommendations")
        undershooters = self._select_recommendations(candidates, grid_df)

        logger.info(
            "Undershooting detection complete",
            undershooters_found=len(undershooters),
            total_cells_analyzed=gis_df['cell_name'].nunique()
        )

        return undershooters

    def detect_with_grids(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect undershooting cells and return grid-level interference data.

        Same as detect(), but also returns the interference grids DataFrame
        for downstream analysis (e.g., tier_3 metrics calculation).

        Args:
            grid_df: Grid measurements DataFrame
            gis_df: Cell GIS data DataFrame

        Returns:
            Tuple of (undershooters_df, interference_grids_df):
                - undershooters_df: Cell-level recommendations
                - interference_grids_df: Grid-level detail with interference flags
        """
        # Sanitise inputs
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        logger.info(
            "Starting undershooting detection (with grids)",
            cells=gis_df['cell_name'].nunique(),
            grid_measurements=len(grid_df)
        )

        # Step 1: Filter to cells with short range
        candidates = self._filter_by_distance(grid_df)

        if len(candidates) == 0:
            logger.warning("No cells found with max distance <= threshold")
            return pd.DataFrame(), pd.DataFrame()

        # Step 2: Calculate interference metrics and get interference grids
        candidates, interference_grids_df = self._calculate_interference_with_grids(candidates, grid_df)

        # Step 3: Filter by traffic and interference
        candidates = self._filter_candidates(candidates)

        if len(candidates) == 0:
            logger.warning("No candidates passed traffic/interference filters")
            return pd.DataFrame(), interference_grids_df

        # Step 4: Merge with GIS data
        candidates = candidates.merge(
            gis_df[['cell_name', 'latitude', 'longitude', 'bearing',
                   'tilt_mech', 'tilt_elc', 'antenna_height']],
            on='cell_name',
            how='inner'
        )

        # Step 5: Calculate uptilt impact
        candidates = self._calculate_uptilt_impact(candidates)

        # Step 6: Select best uptilt recommendation
        undershooters = self._select_recommendations(candidates, grid_df)

        logger.info(
            "Undershooting detection complete (with grids)",
            undershooters_found=len(undershooters),
            interference_grids=len(interference_grids_df)
        )

        return undershooters, interference_grids_df

    def detect_with_environments(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
        environment_df: pd.DataFrame,
        env_params: dict,
    ) -> pd.DataFrame:
        """
        Detect undershooting cells using per-cell environment-specific parameters.

        This runs detection ONCE, using each cell's classified environment to determine
        its thresholds (max_cell_distance, max_interference_percentage, etc.).

        Args:
            grid_df: Grid measurements DataFrame
            gis_df: Cell GIS data DataFrame
            environment_df: DataFrame with cell_name and environment columns
            env_params: Dict mapping environment name to UndershooterParams

        Returns:
            DataFrame with undershooting cells and recommendations
        """
        # Sanitise inputs
        grid_df, gis_df = self._sanitise_inputs(grid_df, gis_df)

        # Build cell -> environment mapping
        environment_df = environment_df.copy()
        environment_df['cell_name'] = environment_df['cell_name'].astype(str)
        cell_env_map = dict(zip(environment_df['cell_name'], environment_df['environment'].str.lower()))

        logger.info(
            "Starting environment-aware undershooting detection (single pass)",
            cells=gis_df['cell_name'].nunique(),
            grid_measurements=len(grid_df),
            environments={k.upper(): sum(1 for v in cell_env_map.values() if v == k)
                         for k in ['urban', 'suburban', 'rural']},
        )

        # Helper to get params for a cell
        def get_params(cell_name: str) -> UndershooterParams:
            env = cell_env_map.get(str(cell_name), 'suburban')
            return env_params.get(env, env_params['suburban'])

        # Step 1: Calculate cell statistics
        cell_stats = grid_df.groupby('cell_name').agg({
            'distance_to_cell': 'max',
            'event_count': 'sum',
            'grid': 'count'
        }).reset_index()
        cell_stats.columns = ['cell_name', 'max_distance_m', 'total_traffic', 'total_grids']

        # Step 2: Filter by distance using per-cell thresholds (vectorized)
        # Build environment -> max_cell_distance lookup
        env_max_distance = {env: p.max_cell_distance for env, p in env_params.items()}
        # Map cell_name -> environment -> max_cell_distance
        cell_envs = cell_stats['cell_name'].map(cell_env_map).fillna('suburban')
        cell_max_distance = cell_envs.map(env_max_distance)
        candidates = cell_stats[cell_stats['max_distance_m'] <= cell_max_distance].copy()

        # Add environment to candidates
        candidates['environment'] = candidates['cell_name'].map(cell_env_map).fillna('suburban')

        logger.info(
            "After distance filter (per-cell)",
            candidates=len(candidates),
            by_environment=candidates['environment'].value_counts().to_dict(),
        )

        if len(candidates) == 0:
            logger.warning("No cells found with max distance <= threshold")
            return pd.DataFrame()

        # Step 3: Calculate interference (per-cell thresholds applied inside)
        candidates = self._calculate_interference_by_environment(
            candidates, grid_df, cell_env_map, env_params
        )

        # Step 4: Filter by traffic and interference using per-cell thresholds (vectorized)
        # Build environment -> threshold lookups
        env_min_traffic = {env: p.min_cell_event_count for env, p in env_params.items()}
        env_max_interference = {env: p.max_interference_percentage for env, p in env_params.items()}
        # Map thresholds to each cell
        cand_envs = candidates['environment']
        min_traffic_threshold = cand_envs.map(env_min_traffic)
        max_interference_threshold = cand_envs.map(env_max_interference)
        # Apply vectorized filter
        filter_mask = (
            (candidates['total_traffic'] >= min_traffic_threshold) &
            (candidates['interference_percentage'] <= max_interference_threshold)
        )
        filtered = candidates[filter_mask].copy()

        logger.info(
            "After traffic/interference filter (per-cell)",
            candidates=len(filtered),
            by_environment=filtered['environment'].value_counts().to_dict() if len(filtered) > 0 else {},
        )

        if len(filtered) == 0:
            logger.warning("No candidates passed traffic/interference filters")
            return pd.DataFrame()

        # Step 5: Merge with GIS data
        filtered = filtered.merge(
            gis_df[['cell_name', 'latitude', 'longitude', 'bearing',
                   'tilt_mech', 'tilt_elc', 'antenna_height']],
            on='cell_name',
            how='inner'
        )

        # Step 6: Calculate uptilt impact with environment-specific path loss exponents
        filtered = self._calculate_uptilt_impact_by_environment(
            filtered, cell_env_map, env_params
        )

        # Step 7: Select recommendations using per-cell thresholds
        undershooters = self._select_recommendations_by_environment(
            filtered, grid_df, cell_env_map, env_params
        )

        # Add environment metadata
        if len(undershooters) > 0:
            undershooters['environment'] = undershooters['cell_name'].map(cell_env_map).fillna('suburban')
            undershooters['environment'] = undershooters['environment'].str.upper()

            # Merge with intersite distance if available
            if 'intersite_distance_km' in environment_df.columns:
                undershooters = undershooters.merge(
                    environment_df[['cell_name', 'intersite_distance_km']],
                    on='cell_name',
                    how='left'
                )

        logger.info(
            "Environment-aware undershooting detection complete (single pass)",
            undershooters_found=len(undershooters),
            by_environment=undershooters['environment'].value_counts().to_dict() if len(undershooters) > 0 else {},
        )

        return undershooters

    def _calculate_interference_by_environment(
        self,
        candidates: pd.DataFrame,
        grid_df: pd.DataFrame,
        cell_env_map: dict,
        env_params: dict,
    ) -> pd.DataFrame:
        """
        Calculate interference metrics using per-cell max_cell_grid_count thresholds.
        """
        candidate_ids = candidates['cell_name'].unique()
        candidate_grids = grid_df[grid_df['cell_name'].isin(candidate_ids)].copy()
        grid_df_temp = grid_df.copy()

        # Calculate P90 RSRP per grid (band-aware if available)
        rsrp_quantile = self.params.rsrp_competition_quantile
        if 'Band' in grid_df.columns:
            p90_rsrp = grid_df_temp.groupby(['grid', 'Band'])['avg_rsrp'].quantile(rsrp_quantile)
            grid_df_temp['p90_rsrp_in_grid'] = grid_df_temp.set_index(['grid', 'Band']).index.map(p90_rsrp)
        else:
            p90_rsrp = grid_df_temp.groupby('grid')['avg_rsrp'].quantile(rsrp_quantile).to_dict()
            grid_df_temp['p90_rsrp_in_grid'] = grid_df_temp['grid'].map(p90_rsrp)

        # Calculate RSRP diff and competing flag
        grid_df_temp['rsrp_diff'] = grid_df_temp['p90_rsrp_in_grid'] - grid_df_temp['avg_rsrp']
        grid_df_temp['is_competing'] = grid_df_temp['rsrp_diff'] <= self.params.interference_threshold_db

        # Count competing cells per grid
        if 'Band' in grid_df.columns:
            competing_per_grid = grid_df_temp.groupby(['grid', 'Band'])['is_competing'].sum().reset_index()
            competing_per_grid.columns = ['grid', 'Band', 'competing_cells']
        else:
            competing_per_grid = grid_df_temp.groupby('grid')['is_competing'].sum().reset_index()
            competing_per_grid.columns = ['grid', 'competing_cells']

        # For each candidate cell, determine which of its grids are "interfering"
        # based on that cell's environment-specific max_cell_grid_count
        if 'Band' in grid_df.columns:
            candidate_grids = candidate_grids.merge(competing_per_grid, on=['grid', 'Band'], how='left')
        else:
            candidate_grids = candidate_grids.merge(competing_per_grid, on='grid', how='left')

        candidate_grids['competing_cells'] = candidate_grids['competing_cells'].fillna(0)

        # Mark interfering grids per-cell based on environment threshold (vectorized)
        # Build environment -> max_cell_grid_count lookup
        env_max_grid_count = {env: p.max_cell_grid_count for env, p in env_params.items()}
        # Map cell_name -> environment -> max_cell_grid_count
        grid_envs = candidate_grids['cell_name'].map(cell_env_map).fillna('suburban')
        grid_max_count = grid_envs.map(env_max_grid_count)
        candidate_grids['is_interfering'] = candidate_grids['competing_cells'] > grid_max_count

        # Aggregate per cell
        interference_stats = candidate_grids.groupby('cell_name').agg(
            interference_grids=('is_interfering', 'sum'),
            total_grids_check=('grid', 'count')
        ).reset_index()

        # Merge with candidates
        candidates = candidates.merge(
            interference_stats[['cell_name', 'interference_grids']],
            on='cell_name',
            how='left'
        )
        candidates['interference_grids'] = candidates['interference_grids'].fillna(0).astype(int)
        candidates['interference_percentage'] = (
            candidates['interference_grids'] / candidates['total_grids']
        ).fillna(0.0)

        return candidates

    def _select_recommendations_by_environment(
        self,
        candidates: pd.DataFrame,
        grid_df: pd.DataFrame,
        cell_env_map: dict,
        env_params: dict,
    ) -> pd.DataFrame:
        """
        Select uptilt recommendations using per-cell environment thresholds (vectorized).

        Cells with physical constraints (e.g., already at 0° tilt) are still included
        in the output with a flag indicating the constraint, as they may indicate
        physical build issues that should be investigated.
        """
        # Build environment-based threshold lookups
        env_min_cov_2deg = {env: p.min_coverage_increase_2deg for env, p in env_params.items()}
        env_min_dist_2deg = {env: p.min_distance_gain_2deg_m for env, p in env_params.items()}
        env_min_cov_1deg = {env: p.min_coverage_increase_1deg for env, p in env_params.items()}
        env_min_dist_1deg = {env: p.min_distance_gain_1deg_m for env, p in env_params.items()}

        # Map thresholds to each cell based on environment
        cell_envs = candidates['cell_name'].map(cell_env_map).fillna('suburban')
        min_cov_2deg = cell_envs.map(env_min_cov_2deg)
        min_dist_2deg = cell_envs.map(env_min_dist_2deg)
        min_cov_1deg = cell_envs.map(env_min_cov_1deg)
        min_dist_1deg = cell_envs.map(env_min_dist_1deg)

        # Check criteria (vectorized)
        dist_gain = candidates['new_distance_2deg_m'] - candidates['max_distance_m']
        dist_gain_1deg = candidates['new_distance_1deg_m'] - candidates['max_distance_m']

        meets_2deg = (
            (candidates['coverage_increase_2deg_pct'] >= min_cov_2deg) &
            (dist_gain >= min_dist_2deg)
        )
        meets_1deg = (
            (candidates['coverage_increase_1deg_pct'] >= min_cov_1deg) &
            (dist_gain_1deg >= min_dist_1deg)
        )

        # Identify cells with physical constraints (can't uptilt)
        has_constraint = candidates['uptilt_constraint'].notna() if 'uptilt_constraint' in candidates.columns else pd.Series(False, index=candidates.index)

        # Apply recommendations using np.select
        conditions = [meets_2deg, meets_1deg]
        candidates['recommended_uptilt_deg'] = np.select(conditions, [2, 1], default=0)
        candidates['new_max_distance_m'] = np.select(
            conditions,
            [candidates['new_distance_2deg_m'], candidates['new_distance_1deg_m']],
            default=candidates['max_distance_m']
        )
        candidates['coverage_increase_percentage'] = np.select(
            conditions,
            [candidates['coverage_increase_2deg_pct'], candidates['coverage_increase_1deg_pct']],
            default=0.0
        )

        # Add physical constraint note for cells that can't uptilt
        # These cells are undershooting but can't be fixed via tilt - may indicate physical build issue
        if 'uptilt_constraint' in candidates.columns:
            candidates['physical_constraint_note'] = candidates['uptilt_constraint'].apply(
                lambda x: "Cell at minimum tilt (0°) - investigate physical build" if x == 'MIN_TILT_REACHED' else None
            )
        else:
            candidates['physical_constraint_note'] = None

        # Include cells with valid recommendations OR physical constraints
        # (physical constraints indicate potential build issues worth investigating)
        has_recommendation = candidates['recommended_uptilt_deg'] > 0
        undershooters = candidates[has_recommendation | has_constraint].copy()

        if len(undershooters) == 0:
            return pd.DataFrame()

        # Log constraint info
        constrained_count = has_constraint.sum()
        if constrained_count > 0:
            logger.info(
                "Cells with physical constraints included in output",
                constrained_cells=constrained_count,
                note="These cells are undershooting but can't uptilt - investigate physical build"
            )

        # Add coverage expansion metrics
        undershooters = self._add_coverage_metrics(undershooters)

        # Calculate severity scores
        undershooters = self._calculate_severity_scores(undershooters, grid_df)

        # Select final columns (including new physical_constraint_note)
        output_cols = [
            'cell_name', 'max_distance_m', 'total_grids', 'interference_grids',
            'interference_percentage', 'total_traffic', 'tilt_mech', 'tilt_elc',
            'recommended_uptilt_deg', 'new_max_distance_m', 'coverage_increase_percentage',
            'current_coverage_grids', 'current_distance_m', 'distance_increase_m',
            'new_coverage_grids', 'total_coverage_after_uptilt',
            'severity_score', 'severity_category', 'physical_constraint_note'
        ]
        available_cols = [c for c in output_cols if c in undershooters.columns]

        return undershooters[available_cols].sort_values('severity_score', ascending=False)

    def _calculate_interference_with_grids(
        self,
        candidates: pd.DataFrame,
        grid_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate interference metrics and return interfering grids.

        Delegates to _calculate_interference with return_grids=True.

        Returns:
            Tuple of (candidates_with_metrics, interference_grids_df)
        """
        return self._calculate_interference(candidates, grid_df, return_grids=True)

    def _filter_by_distance(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to cells with maximum serving distance below threshold."""
        # Calculate max distance per cell
        cell_stats = grid_df.groupby('cell_name').agg({
            'distance_to_cell': 'max',
            'event_count': 'sum',
            'grid': 'count'
        }).reset_index()

        cell_stats.columns = ['cell_name', 'max_distance_m', 'total_traffic', 'total_grids']

        # Filter by distance
        candidates = cell_stats[
            cell_stats['max_distance_m'] <= self.params.max_cell_distance
        ].copy()

        logger.info(
            "After distance filter",
            candidates=len(candidates),
            max_distance=self.params.max_cell_distance
        )

        return candidates

    def _calculate_interference(
        self,
        candidates: pd.DataFrame,
        grid_df: pd.DataFrame,
        return_grids: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Calculate interference metrics for candidate cells using RSRP-based competition analysis.

        For each grid (per frequency band), counts cells within interference_threshold_db
        of the strongest cell. Only grids with competing cells > max_cell_grid_count are
        flagged as interfering.

        IMPORTANT: Competition is calculated per-band. A 700 MHz cell only competes with
        other 700 MHz cells, not with 2100 MHz cells in the same grid.

        This approach correctly identifies actual interference (cells competing for UE)
        rather than just cell density (which includes cells too weak to compete).

        Args:
            candidates: DataFrame with candidate cells
            grid_df: Grid measurements DataFrame
            return_grids: If True, also return DataFrame of interfering grids

        Returns:
            If return_grids=False: candidates DataFrame with interference metrics
            If return_grids=True: Tuple of (candidates, interference_grids_df)
        """
        # Get list of candidate cell IDs
        candidate_ids = candidates['cell_name'].unique()

        # Filter grid data to only candidate cells
        candidate_grids = grid_df[grid_df['cell_name'].isin(candidate_ids)].copy()

        grid_df_temp = grid_df.copy()

        # BAND-AWARE: Calculate competition per grid per band
        rsrp_quantile = self.params.rsrp_competition_quantile
        if 'Band' in grid_df.columns:
            # Step 1: Find reference RSRP (configurable quantile) per grid PER BAND
            p90_rsrp_per_grid_band = grid_df_temp.groupby(['grid', 'Band'])['avg_rsrp'].quantile(rsrp_quantile)
            grid_df_temp['p90_rsrp_in_grid'] = grid_df_temp.set_index(['grid', 'Band']).index.map(p90_rsrp_per_grid_band)

            # Step 2: Calculate RSRP difference from strongest in same band
            grid_df_temp['rsrp_diff'] = grid_df_temp['p90_rsrp_in_grid'] - grid_df_temp['avg_rsrp']

            # Step 3: Flag cells within threshold
            grid_df_temp['is_competing'] = grid_df_temp['rsrp_diff'] <= self.params.interference_threshold_db

            # Step 4: Count competing cells per grid PER BAND
            competing_per_grid = grid_df_temp.groupby(['grid', 'Band'])['is_competing'].sum()

            # Step 5: Flag grids (per band) with excessive competition
            interfering_grids = competing_per_grid[competing_per_grid > self.params.max_cell_grid_count].index.tolist()
            interfering_grids_set = set(interfering_grids)

            # Step 6: Count interfering grids per candidate cell (must match both grid AND band)
            candidate_grids['grid_band_key'] = list(zip(candidate_grids['grid'], candidate_grids['Band']))
            candidate_grids['is_interfering'] = candidate_grids['grid_band_key'].isin(interfering_grids_set)

            logger.info(
                "Band-aware interference calculation enabled",
                unique_bands=grid_df_temp['Band'].nunique(),
                bands_found=grid_df_temp['Band'].unique().tolist()[:10],
            )
        else:
            # Fallback: band-agnostic calculation
            # Step 1: Find reference RSRP (configurable quantile) per grid
            p90_rsrp_per_grid = grid_df_temp.groupby('grid')['avg_rsrp'].quantile(rsrp_quantile).to_dict()
            grid_df_temp['p90_rsrp_in_grid'] = grid_df_temp['grid'].map(p90_rsrp_per_grid)

            # Step 2: Calculate RSRP difference from strongest
            grid_df_temp['rsrp_diff'] = grid_df_temp['p90_rsrp_in_grid'] - grid_df_temp['avg_rsrp']

            # Step 3: Flag cells within threshold
            grid_df_temp['is_competing'] = grid_df_temp['rsrp_diff'] <= self.params.interference_threshold_db

            # Step 4: Count competing cells per grid
            competing_per_grid = grid_df_temp.groupby('grid')['is_competing'].sum()

            # Step 5: Flag grids with excessive competition
            interfering_grids = competing_per_grid[competing_per_grid > self.params.max_cell_grid_count].index.tolist()
            interfering_grids_set = set(interfering_grids)

            # Step 6: Count interfering grids per candidate cell
            candidate_grids['is_interfering'] = candidate_grids['grid'].isin(interfering_grids_set)

            logger.warning(
                "No Band column found - using band-agnostic interference calculation",
                available_columns=list(grid_df.columns)[:20],
            )

        # Step 7: Aggregate interference stats per cell
        interference_stats = candidate_grids.groupby('cell_name').agg(
            interference_grids=('is_interfering', 'sum'),
            total_grids_check=('grid', 'count')
        ).reset_index()

        # Merge interference stats with candidates
        candidates = candidates.merge(
            interference_stats[['cell_name', 'interference_grids']],
            on='cell_name',
            how='left'
        )

        # Fill NaN with 0 (cells with no interference data)
        candidates['interference_grids'] = candidates['interference_grids'].fillna(0).astype(int)

        # Calculate interference percentage
        candidates['interference_percentage'] = (
            candidates['interference_grids'] / candidates['total_grids']
        ).fillna(0.0)

        logger.info(
            "Interference metrics calculated (RSRP-based)",
            band_aware='Band' in grid_df.columns,
            cells_with_interference=len(candidates[candidates['interference_percentage'] > 0]),
            avg_interference_pct=candidates['interference_percentage'].mean(),
            max_interference_pct=candidates['interference_percentage'].max(),
            interference_threshold_db=self.params.interference_threshold_db,
            max_competing_cells=self.params.max_cell_grid_count
        )

        if return_grids:
            interference_grids_df = candidate_grids[candidate_grids['is_interfering']].copy()
            return candidates, interference_grids_df
        return candidates

    def _filter_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Filter candidates by traffic and interference thresholds."""
        filtered = candidates[
            (candidates['total_traffic'] >= self.params.min_cell_event_count) &
            (candidates['interference_percentage'] <= self.params.max_interference_percentage)
        ].copy()

        logger.info(
            "After traffic/interference filter",
            candidates=len(filtered),
            min_traffic=self.params.min_cell_event_count,
            max_interference=self.params.max_interference_percentage
        )

        return filtered

    def _calculate_uptilt_impact(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Calculate predicted impact of 1° and 2° uptilt."""
        # Calculate for 1 degree uptilt
        uptilt_1deg = candidates.apply(
            lambda row: self._estimate_distance_after_tilt(
                d_max_m=row['max_distance_m'],
                alpha_deg=row['tilt_elc'] + row['tilt_mech'],
                h_m=row['antenna_height'],
                delta_tilt_deg=-1.0  # negative = uptilt
            ),
            axis=1,
            result_type='expand'
        )

        candidates['new_distance_1deg_m'] = uptilt_1deg[0]
        candidates['coverage_increase_1deg_pct'] = uptilt_1deg[1]

        # Calculate for 2 degrees uptilt
        uptilt_2deg = candidates.apply(
            lambda row: self._estimate_distance_after_tilt(
                d_max_m=row['max_distance_m'],
                alpha_deg=row['tilt_elc'] + row['tilt_mech'],
                h_m=row['antenna_height'],
                delta_tilt_deg=-2.0  # negative = uptilt
            ),
            axis=1,
            result_type='expand'
        )

        candidates['new_distance_2deg_m'] = uptilt_2deg[0]
        candidates['coverage_increase_2deg_pct'] = uptilt_2deg[1]

        return candidates

    def _calculate_uptilt_impact_by_environment(
        self,
        candidates: pd.DataFrame,
        cell_env_map: dict,
        env_params: dict,
    ) -> pd.DataFrame:
        """
        Calculate predicted impact of 1° and 2° uptilt using environment-specific path loss exponents.

        Args:
            candidates: DataFrame with cell info including tilt and antenna height
            cell_env_map: Dict mapping cell_name to environment ('urban', 'suburban', 'rural')
            env_params: Dict mapping environment name to UndershooterParams

        Returns:
            DataFrame with uptilt impact columns added
        """
        def get_path_loss_exponent(cell_name: str) -> float:
            """Get path loss exponent for a cell based on its environment."""
            env = cell_env_map.get(str(cell_name), 'suburban')
            params = env_params.get(env, env_params['suburban'])
            return params.path_loss_exponent

        # Calculate for 1 degree uptilt
        uptilt_1deg = candidates.apply(
            lambda row: self._estimate_distance_after_tilt_with_ple(
                d_max_m=row['max_distance_m'],
                alpha_deg=row['tilt_elc'] + row['tilt_mech'],
                h_m=row['antenna_height'],
                delta_tilt_deg=-1.0,  # negative = uptilt
                path_loss_exponent=get_path_loss_exponent(row['cell_name'])
            ),
            axis=1,
            result_type='expand'
        )

        candidates['new_distance_1deg_m'] = uptilt_1deg[0]
        candidates['coverage_increase_1deg_pct'] = uptilt_1deg[1]
        candidates['uptilt_constraint'] = uptilt_1deg[2]

        # Calculate for 2 degrees uptilt
        uptilt_2deg = candidates.apply(
            lambda row: self._estimate_distance_after_tilt_with_ple(
                d_max_m=row['max_distance_m'],
                alpha_deg=row['tilt_elc'] + row['tilt_mech'],
                h_m=row['antenna_height'],
                delta_tilt_deg=-2.0,  # negative = uptilt
                path_loss_exponent=get_path_loss_exponent(row['cell_name'])
            ),
            axis=1,
            result_type='expand'
        )

        candidates['new_distance_2deg_m'] = uptilt_2deg[0]
        candidates['coverage_increase_2deg_pct'] = uptilt_2deg[1]

        logger.info(
            "Calculated uptilt impact with environment-specific path loss exponents",
            cells_with_constraint=candidates['uptilt_constraint'].notna().sum(),
        )

        return candidates

    def _estimate_distance_after_tilt_with_ple(
        self,
        d_max_m: float,
        alpha_deg: float,
        h_m: float,
        delta_tilt_deg: float,
        path_loss_exponent: float
    ) -> Tuple[float, float, Optional[str]]:
        """
        Estimate new max coverage distance after changing tilt.

        Uses 3GPP antenna pattern and log-distance path loss model with
        environment-specific path loss exponent.

        Parameters
        ----------
        d_max_m : float
            Current maximum serving distance (meters)
        alpha_deg : float
            Current total downtilt (mechanical + electrical, degrees)
        h_m : float
            Antenna height above UE (meters)
        delta_tilt_deg : float
            Tilt change (negative for uptilt, positive for downtilt)
        path_loss_exponent : float
            Path loss exponent (varies by environment: urban=4.0, suburban=3.5, rural=3.0)

        Returns
        -------
        Tuple[float, float, Optional[str]]
            - new_distance_m: Predicted new maximum distance
            - increase_pct: Fractional increase (e.g., 0.15 = 15% increase)
            - constraint: None if no constraint, or string describing physical limitation
        """
        constraint = None

        # Check if cell is already at minimum tilt (can't uptilt further)
        if alpha_deg <= 0 and delta_tilt_deg < 0:
            # Cell at 0° or negative tilt - can't uptilt, flag as physical build constraint
            constraint = "MIN_TILT_REACHED"
            return d_max_m, 0.0, constraint

        if d_max_m <= 0 or h_m < 0:
            return d_max_m, 0.0, constraint

        # Elevation angle from site to current edge user
        theta_e_deg = math.degrees(math.atan2(h_m, d_max_m))

        # 3GPP vertical attenuation before/after
        A_before = self._vertical_attenuation(theta_e_deg, alpha_deg)
        A_after = self._vertical_attenuation(theta_e_deg, alpha_deg + delta_tilt_deg)

        # Gain change at edge direction (dB)
        deltaG_dB = -(A_after - A_before)

        # Translate to distance using log-distance model with environment-specific exponent
        d_new_m = d_max_m * (10.0 ** (deltaG_dB / (10.0 * path_loss_exponent)))

        # For uptilt, don't allow decrease
        if d_new_m < d_max_m:
            d_new_m = d_max_m

        increase_pct = (d_new_m - d_max_m) / d_max_m

        return d_new_m, increase_pct, constraint

    def _estimate_distance_after_tilt(
        self,
        d_max_m: float,
        alpha_deg: float,
        h_m: float,
        delta_tilt_deg: float
    ) -> Tuple[float, float]:
        """
        Estimate new max coverage distance after changing tilt.

        Uses 3GPP antenna pattern and log-distance path loss model.

        Parameters
        ----------
        d_max_m : float
            Current maximum serving distance (meters)
        alpha_deg : float
            Current electrical downtilt (degrees)
        h_m : float
            Antenna height above UE (meters)
        delta_tilt_deg : float
            Tilt change (negative for uptilt, positive for downtilt)

        Returns
        -------
        new_distance_m : float
            Predicted new maximum distance
        increase_pct : float
            Fractional increase (e.g., 0.15 = 15% increase)
        """
        if alpha_deg == 0 and delta_tilt_deg < 0:
            # Cannot uptilt from 0 degrees
            return d_max_m, 0.0

        if d_max_m <= 0 or h_m < 0:
            return d_max_m, 0.0

        # Elevation angle from site to current edge user
        theta_e_deg = math.degrees(math.atan2(h_m, d_max_m))

        # 3GPP vertical attenuation before/after
        A_before = self._vertical_attenuation(theta_e_deg, alpha_deg)
        A_after = self._vertical_attenuation(theta_e_deg, alpha_deg + delta_tilt_deg)

        # Gain change at edge direction (dB)
        deltaG_dB = -(A_after - A_before)

        # Translate to distance using log-distance model
        d_new_m = d_max_m * (10.0 ** (deltaG_dB / (10.0 * self.params.path_loss_exponent)))

        # For uptilt, don't allow decrease
        if d_new_m < d_max_m:
            d_new_m = d_max_m

        increase_pct = (d_new_m - d_max_m) / d_max_m

        return d_new_m, increase_pct

    def _vertical_attenuation(self, theta_deg: float, alpha_deg: float) -> float:
        """3GPP parabolic attenuation in vertical plane (dB)."""
        return min(
            12.0 * (((theta_deg - alpha_deg) / self.params.hpbw_v_deg) ** 2),
            self.params.sla_v_db
        )

    def _apply_uptilt_recommendations(
        self,
        candidates: pd.DataFrame,
        params: 'UndershooterParams',
    ) -> pd.DataFrame:
        """
        Apply uptilt recommendation logic using vectorized operations.

        Args:
            candidates: DataFrame with uptilt impact columns
            params: Parameters containing thresholds

        Returns:
            DataFrame with recommendation columns added
        """
        # Check 2-degree criteria (vectorized)
        meets_2deg = (
            (candidates['coverage_increase_2deg_pct'] >= params.min_coverage_increase_2deg) &
            (candidates['new_distance_2deg_m'] - candidates['max_distance_m'] >= params.min_distance_gain_2deg_m)
        )

        # Check 1-degree criteria (vectorized)
        meets_1deg = (
            (candidates['coverage_increase_1deg_pct'] >= params.min_coverage_increase_1deg) &
            (candidates['new_distance_1deg_m'] - candidates['max_distance_m'] >= params.min_distance_gain_1deg_m)
        )

        # Assign recommendations using np.select (priority: 2deg > 1deg > none)
        conditions = [meets_2deg, meets_1deg]

        candidates['recommended_uptilt_deg'] = np.select(conditions, [2, 1], default=0)

        candidates['new_max_distance_m'] = np.select(
            conditions,
            [candidates['new_distance_2deg_m'], candidates['new_distance_1deg_m']],
            default=candidates['max_distance_m']
        )

        candidates['coverage_increase_percentage'] = np.select(
            conditions,
            [candidates['coverage_increase_2deg_pct'], candidates['coverage_increase_1deg_pct']],
            default=0.0
        )

        return candidates

    def _add_coverage_metrics(self, undershooters: pd.DataFrame) -> pd.DataFrame:
        """
        Add coverage expansion metrics to undershooters DataFrame.

        Args:
            undershooters: DataFrame with recommendation columns

        Returns:
            DataFrame with coverage metrics added
        """
        undershooters['current_coverage_grids'] = undershooters['total_grids']
        undershooters['current_distance_m'] = undershooters['max_distance_m']
        undershooters['distance_increase_m'] = (
            undershooters['new_max_distance_m'] - undershooters['max_distance_m']
        )
        undershooters['new_coverage_grids'] = (
            undershooters['total_grids'] * undershooters['coverage_increase_percentage']
        ).round().astype(int)
        undershooters['total_coverage_after_uptilt'] = (
            undershooters['current_coverage_grids'] + undershooters['new_coverage_grids']
        )
        return undershooters

    def _calculate_severity_scores(
        self,
        undershooters: pd.DataFrame,
        grid_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate multi-factor severity scores (0-1) for undershooting cells.

        Uses balanced weighting focused on coverage improvement potential:
        - Coverage increase %: 30% (primary metric - how much gain from uptilt)
        - New grids gained: 25% (absolute coverage expansion)
        - Low interference %: 20% (cells with low interference are better candidates)
        - Distance gain: 15% (how much further the cell would reach)
        - Traffic potential: 10% (total grids - indicates demand in area)

        Optional band multiplier:
        - 800 MHz: 1.1x (more critical to fix - larger coverage impact)
        - 700 MHz: 1.05x
        - 1800 MHz: 1.0x (baseline)
        - 2100 MHz: 0.9x (less critical)

        Args:
            undershooters: DataFrame with undershooting cells
            grid_df: Original grid data (for band lookup)

        Returns:
            DataFrame with added columns:
                - severity_score: 0-1 normalized score
                - severity_category: CRITICAL/HIGH/MEDIUM/LOW/MINIMAL
        """
        if len(undershooters) == 0:
            undershooters['severity_score'] = []
            undershooters['severity_category'] = []
            return undershooters

        # Calculate dataset statistics for normalization using percentiles (robust to outliers)
        new_grids_p5 = undershooters['new_coverage_grids'].quantile(0.05)
        new_grids_p95 = undershooters['new_coverage_grids'].quantile(0.95)
        distance_gain_p5 = undershooters['distance_increase_m'].quantile(0.05)
        distance_gain_p95 = undershooters['distance_increase_m'].quantile(0.95)
        total_grids_p5 = undershooters['total_grids'].quantile(0.05)
        total_grids_p95 = undershooters['total_grids'].quantile(0.95)

        logger.info(
            "Calculating undershooting severity scores (percentile-based normalization)",
            new_grids_p5_p95=f"{new_grids_p5:.0f}-{new_grids_p95:.0f}",
            distance_gain_p5_p95=f"{distance_gain_p5:.0f}-{distance_gain_p95:.0f}m",
        )

        # 1. Coverage increase percentage is already 0-1 (typically 0.05-0.5)
        # Normalize using config value (e.g., 50% increase = 1.0)
        max_coverage_pct = self.params.severity_max_coverage_increase_pct
        coverage_score = (undershooters['coverage_increase_percentage'] / max_coverage_pct).clip(0, 1)

        # 2. Normalize new_coverage_grids (0-1) using 5th-95th percentile
        if new_grids_p95 > new_grids_p5:
            new_grids_score = (undershooters['new_coverage_grids'] - new_grids_p5) / (new_grids_p95 - new_grids_p5)
        else:
            new_grids_score = pd.Series(0.5, index=undershooters.index)
        new_grids_score = new_grids_score.clip(0, 1)

        # 3. Low interference score (inverted - lower interference = higher score)
        # interference_percentage is 0-1, so invert it
        low_interference_score = 1.0 - undershooters['interference_percentage']

        # 4. Distance gain normalized using percentiles
        if distance_gain_p95 > distance_gain_p5:
            distance_score = (undershooters['distance_increase_m'] - distance_gain_p5) / (distance_gain_p95 - distance_gain_p5)
        else:
            distance_score = pd.Series(0.5, index=undershooters.index)
        distance_score = distance_score.clip(0, 1)

        # 5. Traffic potential (total grids in area)
        if total_grids_p95 > total_grids_p5:
            traffic_score = (undershooters['total_grids'] - total_grids_p5) / (total_grids_p95 - total_grids_p5)
        else:
            traffic_score = pd.Series(0.5, index=undershooters.index)
        traffic_score = traffic_score.clip(0, 1)

        # Weighted combination using config weights
        severity = (
            self.params.severity_weight_coverage * coverage_score +
            self.params.severity_weight_new_grids * new_grids_score +
            self.params.severity_weight_low_interference * low_interference_score +
            self.params.severity_weight_distance * distance_score +
            self.params.severity_weight_traffic * traffic_score
        )

        # Clip to 0-1 range
        undershooters['severity_score'] = severity.clip(0, 1)

        # Categorize severity using config thresholds (vectorized with np.select)
        conditions = [
            undershooters['severity_score'] >= self.params.severity_threshold_critical,
            undershooters['severity_score'] >= self.params.severity_threshold_high,
            undershooters['severity_score'] >= self.params.severity_threshold_medium,
            undershooters['severity_score'] >= self.params.severity_threshold_low,
        ]
        choices = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        undershooters['severity_category'] = np.select(conditions, choices, default='MINIMAL')

        logger.info(
            "Undershooting severity scores calculated",
            avg_score=undershooters['severity_score'].mean(),
            min_score=undershooters['severity_score'].min(),
            max_score=undershooters['severity_score'].max(),
        )

        return undershooters

    def _select_recommendations(self, candidates: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
        """Select best uptilt recommendation for each candidate."""
        # Apply uptilt recommendations (vectorized)
        candidates = self._apply_uptilt_recommendations(candidates, self.params)

        # Filter to only cells with valid recommendations
        undershooters = candidates[candidates['recommended_uptilt_deg'] > 0].copy()

        # Add coverage expansion metrics
        undershooters = self._add_coverage_metrics(undershooters)

        # Calculate severity scores
        undershooters = self._calculate_severity_scores(undershooters, grid_df)

        # Select final columns
        output_cols = [
            'cell_name', 'max_distance_m', 'total_grids', 'interference_grids',
            'interference_percentage', 'total_traffic', 'tilt_mech', 'tilt_elc',
            'recommended_uptilt_deg', 'new_max_distance_m', 'coverage_increase_percentage',
            'current_coverage_grids', 'current_distance_m', 'distance_increase_m',
            'new_coverage_grids', 'total_coverage_after_uptilt',
            'severity_score', 'severity_category'
        ]

        return undershooters[output_cols].sort_values('severity_score', ascending=False)


def detect_undershooting_cells(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    params: Optional[UndershooterParams] = None
) -> pd.DataFrame:
    """
    Convenience function to detect undershooting cells.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid-level coverage data
    gis_df : pd.DataFrame
        Cell GIS data
    params : UndershooterParams, optional
        Detection parameters. If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        Detected undershooting cells with recommendations

    Examples
    --------
    >>> from ran_optimizer.recommendations import detect_undershooting_cells
    >>> undershooters = detect_undershooting_cells(grid_df, gis_df)
    """
    if params is None:
        params = UndershooterParams()

    detector = UndershooterDetector(params)
    return detector.detect(grid_df, gis_df)


def detect_undershooting_with_environment_awareness(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run undershooting detection with environment-specific parameters.

    This function runs detection ONCE using per-cell environment-specific parameters.
    Each cell's thresholds (max_cell_distance, max_interference_percentage, etc.) are
    determined by its classified environment (URBAN, SUBURBAN, RURAL).

    Interference analysis uses all cells together to maintain correct cross-environment
    calculations.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid measurements DataFrame
    gis_df : pd.DataFrame
        Cell GIS data DataFrame
    environment_df : pd.DataFrame
        DataFrame with columns:
        - cell_name: Cell identifier
        - environment: Environment type ('URBAN', 'SUBURBAN', 'RURAL')
        - intersite_distance_km: Mean intersite distance (optional)
    config_path : str, optional
        Optional path to configuration file

    Returns
    -------
    pd.DataFrame
        Undershooting cells from all environments, including:
        - All standard undershooting metrics
        - environment: Cell environment type
        - intersite_distance_km: Mean intersite distance

    Examples
    --------
    >>> undershooters = detect_undershooting_with_environment_awareness(
    ...     grid_df, gis_df, environment_df
    ... )
    >>> print(f"Found {len(undershooters)} undershooters")
    >>> print(undershooters.groupby('environment').size())
    """
    logger.info(
        "Starting environment-aware undershooting detection (single pass)",
        total_cells=len(gis_df),
        environments=environment_df['environment'].value_counts().to_dict(),
    )

    # Validate environment_df
    required_cols = ['cell_name', 'environment']
    missing_cols = set(required_cols) - set(environment_df.columns)
    if missing_cols:
        raise ValueError(f"environment_df missing required columns: {missing_cols}")

    # Ensure cell_name types match
    environment_df = environment_df.copy()
    environment_df['cell_name'] = environment_df['cell_name'].astype(str)

    # Load parameters for each environment
    env_params = {}
    for env in ['urban', 'suburban', 'rural']:
        env_params[env] = UndershooterParams.from_config(
            config_path=config_path,
            environment=env
        )
        logger.info(
            f"{env.upper()} parameters loaded",
            max_cell_distance=env_params[env].max_cell_distance,
            min_cell_event_count=env_params[env].min_cell_event_count,
            max_interference_percentage=env_params[env].max_interference_percentage,
            min_coverage_increase_1deg=env_params[env].min_coverage_increase_1deg,
        )

    # Create detector with default params (used for shared calculations)
    detector = UndershooterDetector(env_params['suburban'])

    # Run detection ONCE with per-cell environment parameters
    undershooters = detector.detect_with_environments(
        grid_df, gis_df, environment_df, env_params
    )

    if len(undershooters) == 0:
        logger.warning("No undershooters found in any environment")
        return pd.DataFrame()

    logger.info(
        "Environment-aware detection complete (single pass)",
        total_undershooters=len(undershooters),
        by_environment=undershooters['environment'].value_counts().to_dict() if 'environment' in undershooters.columns else {},
    )

    return undershooters


def compare_undershooting_detection_approaches(
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    config_path: Optional[str] = None,
) -> dict:
    """
    Compare standard vs environment-aware undershooting detection approaches.

    Runs both detection methods and returns results for comparison.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid measurements DataFrame
    gis_df : pd.DataFrame
        Cell GIS data DataFrame
    environment_df : pd.DataFrame
        Environment classification DataFrame
    config_path : str, optional
        Optional path to configuration file

    Returns
    -------
    dict
        Dictionary with keys:
        - 'standard': Results using default parameters for all cells
        - 'environment_aware': Results using environment-specific parameters
        - 'comparison': Summary statistics comparing the two approaches

    Examples
    --------
    >>> results = compare_undershooting_detection_approaches(grid_df, gis_df, environment_df)
    >>> print(results['comparison'])
    """
    logger.info("Running detection comparison: standard vs environment-aware")

    # 1. Standard detection (default parameters for all)
    logger.info("=" * 80)
    logger.info("Running STANDARD detection (single parameter set)")
    logger.info("=" * 80)

    params_standard = UndershooterParams.from_config(config_path=config_path)
    detector_standard = UndershooterDetector(params_standard)
    standard_results = detector_standard.detect(grid_df, gis_df)

    # Add environment info to standard results
    if len(standard_results) > 0:
        env_info = environment_df[['cell_name', 'environment', 'intersite_distance_km']]
        standard_results = standard_results.merge(env_info, on='cell_name', how='left')

    # 2. Environment-aware detection
    logger.info("=" * 80)
    logger.info("Running ENVIRONMENT-AWARE detection (per-environment parameters)")
    logger.info("=" * 80)

    env_aware_results = detect_undershooting_with_environment_awareness(
        grid_df, gis_df, environment_df, config_path
    )

    # 3. Generate comparison statistics
    logger.info("=" * 80)
    logger.info("Generating comparison statistics")
    logger.info("=" * 80)

    comparison = _generate_undershooting_comparison_stats(
        standard_results,
        env_aware_results,
        environment_df
    )

    return {
        'standard': standard_results,
        'environment_aware': env_aware_results,
        'comparison': comparison,
    }


def _generate_undershooting_comparison_stats(
    standard_df: pd.DataFrame,
    env_aware_df: pd.DataFrame,
    environment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate comparison statistics between detection approaches."""

    stats = []

    # Overall stats
    stats.append({
        'metric': 'Total Undershooters',
        'standard': len(standard_df),
        'environment_aware': len(env_aware_df),
        'difference': len(env_aware_df) - len(standard_df),
        'pct_change': ((len(env_aware_df) - len(standard_df)) / len(standard_df) * 100) if len(standard_df) > 0 else 0,
    })

    # Stats by environment
    for env in ['URBAN', 'SUBURBAN', 'RURAL']:
        # Count cells in this environment
        env_cell_count = len(environment_df[environment_df['environment'] == env])

        std_count = len(standard_df[standard_df['environment'] == env]) if len(standard_df) > 0 and 'environment' in standard_df.columns else 0
        env_count = len(env_aware_df[env_aware_df['environment'] == env]) if len(env_aware_df) > 0 else 0

        stats.append({
            'metric': f'{env} Undershooters',
            'standard': std_count,
            'environment_aware': env_count,
            'difference': env_count - std_count,
            'pct_change': ((env_count - std_count) / std_count * 100) if std_count > 0 else 0,
        })

        # Detection rate (% of cells in environment flagged as undershooters)
        std_rate = (std_count / env_cell_count * 100) if env_cell_count > 0 else 0
        env_rate = (env_count / env_cell_count * 100) if env_cell_count > 0 else 0

        stats.append({
            'metric': f'{env} Detection Rate (%)',
            'standard': round(std_rate, 2),
            'environment_aware': round(env_rate, 2),
            'difference': round(env_rate - std_rate, 2),
            'pct_change': None,
        })

    comparison_df = pd.DataFrame(stats)

    logger.info("Comparison statistics generated", rows=len(comparison_df))

    return comparison_df
