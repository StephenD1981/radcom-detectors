"""
Daily resolution recommendations export for RAN optimization.

Generates a unified export file combining overshooting and undershooting
recommendations with tier_3 metrics for integration with network management systems.
"""
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DailyResolutionConfig:
    """Configuration for daily resolution recommendations export."""

    # Tilt constraints
    min_tilt: float = 0.0
    max_tilt: float = 15.0

    # Cycle defaults
    default_cycle_status: str = "PENDING"
    default_conditions: str = "AUTO_GENERATED"

    # Output filename
    output_filename: str = "daily_overshooter_resolution_recommendations.csv"


def calculate_tier3_metrics_for_overshooting(
    cell_name: str,
    cell_band: int,
    overshooting_grids: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> dict:
    """
    Calculate tier_3 metrics for an overshooting cell.

    Tier_3 metrics capture:
    - tier_3_sectors_count: Count of DISTINCT same-band cells (excluding target cell)
      that exist in the overshooting_grids
    - tier_3_traffic_total: Traffic of the OFFENDING CELL in overshooting_grids
    - tier3_traffic_perc: (traffic in overshooting_grids) / (total cell traffic) * 100

    Parameters
    ----------
    cell_name : str
        The cell identifier for the recommendation
    cell_band : int
        The frequency band of the recommendation cell
    overshooting_grids : pd.DataFrame
        Grid-level data for overshooting grids (geohash7, cell_name, etc.)
    grid_df : pd.DataFrame
        Full grid data with traffic and other metrics

    Returns
    -------
    dict
        Dictionary with tier_3 metrics
    """
    # Normalize cell_name types
    cell_name_str = str(cell_name)

    # Detect column names in overshooting_grids
    og_cell_name_col = 'cell_name' if 'cell_name' in overshooting_grids.columns else 'cilac'
    og_geohash_col = 'geohash7' if 'geohash7' in overshooting_grids.columns else 'grid'

    # Detect column names in grid_df
    grid_cell_name_col = None
    for col in ['cell_name', 'cilac', 'CILAC']:
        if col in grid_df.columns:
            grid_cell_name_col = col
            break
    if grid_cell_name_col is None:
        grid_cell_name_col = 'cell_name'

    geohash_col = 'geohash7' if 'geohash7' in grid_df.columns else 'grid'
    traffic_col = 'total_traffic' if 'total_traffic' in grid_df.columns else 'event_count'
    drops_col = None
    for col in ['drops', 'drop_count', 'dropped_calls', 'total_drops']:
        if col in grid_df.columns:
            drops_col = col
            break

    # Detect band column
    band_col = None
    for col in ['band', 'Band', 'frequency_band', 'freq_band']:
        if col in grid_df.columns:
            band_col = col
            break

    # Get the overshooting grids for this cell
    cell_overshooting_grids = overshooting_grids[
        overshooting_grids[og_cell_name_col].astype(str) == cell_name_str
    ][og_geohash_col].unique()

    if len(cell_overshooting_grids) == 0:
        return {
            'tier_3_sectors_count': 0,
            'tier_3_cells_count': 0,
            'tier_3_traffic_total': 0,
            'tier_3_drops_total': 0,
            'tier3_traffic_perc': 0.0,
            'tier3_drops_perc': 0.0,
        }

    # Get all grid data for these overshooting grids (all cells serving these grids)
    grids_data = grid_df[grid_df[geohash_col].isin(cell_overshooting_grids)].copy()

    if len(grids_data) == 0:
        return {
            'tier_3_sectors_count': 0,
            'tier_3_cells_count': 0,
            'tier_3_traffic_total': 0,
            'tier_3_drops_total': 0,
            'tier3_traffic_perc': 0.0,
            'tier3_drops_perc': 0.0,
        }

    # Filter to same-band cells only (if band column exists)
    if band_col is not None and cell_band is not None:
        grids_data_same_band = grids_data[grids_data[band_col] == cell_band]
    else:
        grids_data_same_band = grids_data

    # TIER_3_SECTORS_COUNT: Count DISTINCT same-band cells (excluding the target cell)
    other_cells = grids_data_same_band[
        grids_data_same_band[grid_cell_name_col].astype(str) != cell_name_str
    ]
    tier_3_sectors_count = other_cells[grid_cell_name_col].nunique()
    tier_3_cells_count = tier_3_sectors_count

    # TIER_3_TRAFFIC_TOTAL: Traffic of THE OFFENDING CELL in overshooting grids
    target_cell_in_grids = grids_data[
        grids_data[grid_cell_name_col].astype(str) == cell_name_str
    ]
    tier_3_traffic_total = int(target_cell_in_grids[traffic_col].sum()) if traffic_col in target_cell_in_grids.columns else 0

    # Get total traffic of the cell (across all grids)
    all_cell_traffic = grid_df[
        grid_df[grid_cell_name_col].astype(str) == cell_name_str
    ]
    total_cell_traffic = all_cell_traffic[traffic_col].sum() if traffic_col in all_cell_traffic.columns else 0

    # TIER3_TRAFFIC_PERC: (traffic in overshooting_grids) / (total cell traffic) * 100
    tier3_traffic_perc = (tier_3_traffic_total / total_cell_traffic * 100) if total_cell_traffic > 0 else 0.0

    # Drops - offending cell's drops in overshooting grids vs total cell drops
    tier_3_drops_total = 0
    tier3_drops_perc = 0.0
    if drops_col is not None and drops_col in target_cell_in_grids.columns:
        # Drops of the offending cell in overshooting grids
        tier_3_drops_total = int(target_cell_in_grids[drops_col].sum())
        # Total drops of this cell across ALL grids
        total_cell_drops = all_cell_traffic[drops_col].sum() if drops_col in all_cell_traffic.columns else 0
        tier3_drops_perc = (tier_3_drops_total / total_cell_drops * 100) if total_cell_drops > 0 else 0.0

    return {
        'tier_3_sectors_count': int(tier_3_sectors_count),
        'tier_3_cells_count': int(tier_3_cells_count),
        'tier_3_traffic_total': int(tier_3_traffic_total),
        'tier_3_drops_total': int(tier_3_drops_total),
        'tier3_traffic_perc': round(tier3_traffic_perc, 2),
        'tier3_drops_perc': round(tier3_drops_perc, 2),
    }


def calculate_tier3_metrics_for_undershooting(
    cell_name: str,
    cell_band: int,
    interference_grids: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> dict:
    """
    Calculate tier_3 metrics for an undershooting cell.

    Tier_3 metrics capture information about same-band cells (excluding the
    recommendation cell) that appear in the interference grids.

    Parameters
    ----------
    cell_name : str
        The cell identifier for the recommendation
    cell_band : int
        The frequency band of the recommendation cell
    interference_grids : pd.DataFrame
        Grid-level data for interference grids (geohash7, is_interfering, etc.)
    grid_df : pd.DataFrame
        Full grid data with traffic and other metrics

    Returns
    -------
    dict
        Dictionary with tier_3 metrics
    """
    # For undershooting, interference_grids may be identified differently
    # Use the same logic as overshooting but with interference grids
    return calculate_tier3_metrics_for_overshooting(
        cell_name, cell_band, interference_grids, grid_df
    )


def get_cell_band(
    cell_name: str,
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
) -> Optional[int]:
    """Get the frequency band for a cell."""
    # Detect grid_df cell_name column
    grid_cell_name_col = None
    for col in ['cell_name', 'cilac', 'CILAC']:
        if col in grid_df.columns:
            grid_cell_name_col = col
            break

    if grid_cell_name_col is None:
        grid_cell_name_col = 'cell_name'  # Default fallback

    band_col = None
    for col in ['band', 'Band', 'frequency_band', 'freq_band']:
        if col in grid_df.columns:
            band_col = col
            break

    if band_col is not None:
        cell_bands = grid_df[grid_df[grid_cell_name_col].astype(str) == str(cell_name)][band_col]
        if len(cell_bands) > 0:
            return cell_bands.iloc[0]

    # Try gis_df
    gis_cell_name_col = None
    for col in ['cell_name', 'cilac', 'CILAC']:
        if col in gis_df.columns:
            gis_cell_name_col = col
            break

    if gis_cell_name_col is None:
        return None

    for col in ['band', 'Band', 'frequency_band', 'freq_band']:
        if col in gis_df.columns:
            cell_bands = gis_df[gis_df[gis_cell_name_col].astype(str) == str(cell_name)][col]
            if len(cell_bands) > 0:
                return cell_bands.iloc[0]

    return None


def get_cell_gis_info(
    cell_name: str,
    gis_df: pd.DataFrame,
) -> dict:
    """Get GIS info for a cell (tilts, name, etc.)."""
    # Detect cell_name column
    gis_cell_name_col = None
    for col in ['cell_name', 'cilac', 'CILAC']:
        if col in gis_df.columns:
            gis_cell_name_col = col
            break

    if gis_cell_name_col is None:
        return {
            'cell_name': str(cell_name),
            'electrical_tilt': 0.0,
            'mechanical_tilt': 0.0,
        }

    cell_row = gis_df[gis_df[gis_cell_name_col].astype(str) == str(cell_name)]

    if len(cell_row) == 0:
        return {
            'cell_name': str(cell_name),
            'electrical_tilt': 0.0,
            'mechanical_tilt': 0.0,
        }

    cell_row = cell_row.iloc[0]

    # Get cell name
    cell_name = cell_name
    for col in ['Cell_Name', 'CellName', 'cell_name', 'name', 'Name']:
        if col in gis_df.columns:
            cell_name = cell_row.get(col, cell_name)
            if pd.notna(cell_name) and str(cell_name).strip():
                break

    # Get tilts
    electrical_tilt = 0.0
    for col in ['TiltE', 'electrical_tilt', 'ElectricalTilt']:
        if col in gis_df.columns:
            val = cell_row.get(col)
            if pd.notna(val):
                electrical_tilt = float(val)
            break

    mechanical_tilt = 0.0
    for col in ['TiltM', 'mechanical_tilt', 'MechanicalTilt']:
        if col in gis_df.columns:
            val = cell_row.get(col)
            if pd.notna(val):
                mechanical_tilt = float(val)
            break

    return {
        'cell_name': str(cell_name),
        'electrical_tilt': electrical_tilt,
        'mechanical_tilt': mechanical_tilt,
    }


def generate_daily_resolution_recommendations(
    overshooting_df: pd.DataFrame,
    undershooting_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    overshooting_grids_df: Optional[pd.DataFrame] = None,
    interference_grids_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    analysis_date: Optional[date] = None,
    config: Optional[DailyResolutionConfig] = None,
) -> pd.DataFrame:
    """
    Generate daily resolution recommendations combining overshooting and undershooting.

    Creates a unified CSV with the following fields:
    - analysisdate: Date of analysis
    - cell_name: Human-readable cell name
    - cilac: Cell identifier
    - parameter: Tilt parameter to modify (Electrical_tilt or Manual_tilt)
    - category: 'overshooter' or 'undershooter'
    - parameter_new_value: Recommended new tilt value
    - cycle_start_date, cycle_end_date, cycle_status: Cycle management fields
    - conditions: Any conditions for the recommendation
    - current_tilt: Current total tilt (electrical + mechanical)
    - min_tilt, max_tilt: Tilt constraints
    - tier_3_sectors_count: Count of same-band cells in problematic grids
    - tier_3_cells_count: Same as sectors_count (cell-level analysis)
    - tier_3_traffic_total: Total traffic from tier_3 cells
    - tier_3_drops_total: Total drops from tier_3 cells
    - tier3_traffic_perc: Percentage of traffic from tier_3 cells
    - tier3_drops_perc: Percentage of drops from tier_3 cells

    Parameters
    ----------
    overshooting_df : pd.DataFrame
        Overshooting cell recommendations
    undershooting_df : pd.DataFrame
        Undershooting cell recommendations
    grid_df : pd.DataFrame
        Full grid data with traffic/drops
    gis_df : pd.DataFrame
        Cell GIS data with tilts
    overshooting_grids_df : pd.DataFrame, optional
        Grid-level detail for overshooting cells (for tier_3 calculation)
    interference_grids_df : pd.DataFrame, optional
        Grid-level detail for undershooting interference (for tier_3 calculation)
    output_dir : Path, optional
        Output directory (if provided, saves CSV)
    analysis_date : date, optional
        Analysis date (defaults to today)
    config : DailyResolutionConfig, optional
        Configuration options

    Returns
    -------
    pd.DataFrame
        Combined recommendations with all required fields
    """
    if config is None:
        config = DailyResolutionConfig()

    if analysis_date is None:
        analysis_date = date.today()

    logger.info(
        "Generating daily resolution recommendations",
        overshooting_count=len(overshooting_df) if overshooting_df is not None else 0,
        undershooting_count=len(undershooting_df) if undershooting_df is not None else 0,
        analysis_date=analysis_date.isoformat(),
    )

    recommendations = []

    # Process overshooting cells
    if overshooting_df is not None and len(overshooting_df) > 0:
        for _, row in overshooting_df.iterrows():
            cell_name = str(row['cell_name'])

            # Get GIS info
            gis_info = get_cell_gis_info(cell_name, gis_df)

            # Get cell band
            cell_band = get_cell_band(cell_name, grid_df, gis_df)

            # Calculate tier_3 metrics
            if overshooting_grids_df is not None:
                tier3 = calculate_tier3_metrics_for_overshooting(
                    cell_name, cell_band, overshooting_grids_df, grid_df
                )
            else:
                tier3 = {
                    'tier_3_sectors_count': 0,
                    'tier_3_cells_count': 0,
                    'tier_3_traffic_total': 0,
                    'tier_3_drops_total': 0,
                    'tier3_traffic_perc': 0.0,
                    'tier3_drops_perc': 0.0,
                }

            # Get recommended tilt change (downtilt for overshooters)
            recommended_change = row.get('recommended_downtilt_deg', row.get('recommended_tilt_change', 1))
            current_elec_tilt = gis_info['electrical_tilt']
            current_mech_tilt = gis_info['mechanical_tilt']
            current_total_tilt = current_elec_tilt + current_mech_tilt

            # For overshooters, we downtilt (increase tilt)
            # Parameter is always Electrical_tilt for overshooters (downtilt)
            parameter = 'Electrical_tilt'
            new_tilt_value = current_elec_tilt + recommended_change

            # Enforce constraints
            new_tilt_value = min(max(new_tilt_value, config.min_tilt), config.max_tilt)

            recommendations.append({
                'analysisdate': analysis_date.isoformat(),
                'cell_name': gis_info['cell_name'],
                'cilac': cell_name,
                'parameter': parameter,
                'category': 'overshooter',
                'parameter_new_value': round(new_tilt_value, 1),
                'cycle_start_date': analysis_date.isoformat(),
                'cycle_end_date': '',
                'cycle_status': config.default_cycle_status,
                'conditions': config.default_conditions,
                'current_tilt': round(current_total_tilt, 1),
                'min_tilt': config.min_tilt,
                'max_tilt': config.max_tilt,
                **tier3,
            })

    # Process undershooting cells
    if undershooting_df is not None and len(undershooting_df) > 0:
        for _, row in undershooting_df.iterrows():
            cell_name = str(row['cell_name'])

            # Get GIS info
            gis_info = get_cell_gis_info(cell_name, gis_df)

            # Get cell band
            cell_band = get_cell_band(cell_name, grid_df, gis_df)

            # Calculate tier_3 metrics
            if interference_grids_df is not None:
                tier3 = calculate_tier3_metrics_for_undershooting(
                    cell_name, cell_band, interference_grids_df, grid_df
                )
            else:
                tier3 = {
                    'tier_3_sectors_count': 0,
                    'tier_3_cells_count': 0,
                    'tier_3_traffic_total': 0,
                    'tier_3_drops_total': 0,
                    'tier3_traffic_perc': 0.0,
                    'tier3_drops_perc': 0.0,
                }

            # Get recommended tilt change (uptilt for undershooters)
            recommended_change = row.get('recommended_uptilt_deg', 1)
            current_elec_tilt = gis_info['electrical_tilt']
            current_mech_tilt = gis_info['mechanical_tilt']
            current_total_tilt = current_elec_tilt + current_mech_tilt

            # For undershooters, we uptilt (decrease tilt)
            # Parameter logic: Electrical_tilt unless electrical_tilt is already 0
            if current_elec_tilt <= 0:
                parameter = 'Manual_tilt'
                new_tilt_value = current_mech_tilt - recommended_change
            else:
                parameter = 'Electrical_tilt'
                new_tilt_value = current_elec_tilt - recommended_change

            # Enforce constraints
            new_tilt_value = min(max(new_tilt_value, config.min_tilt), config.max_tilt)

            recommendations.append({
                'analysisdate': analysis_date.isoformat(),
                'cell_name': gis_info['cell_name'],
                'cilac': cell_name,
                'parameter': parameter,
                'category': 'undershooter',
                'parameter_new_value': round(new_tilt_value, 1),
                'cycle_start_date': analysis_date.isoformat(),
                'cycle_end_date': '',
                'cycle_status': config.default_cycle_status,
                'conditions': config.default_conditions,
                'current_tilt': round(current_total_tilt, 1),
                'min_tilt': config.min_tilt,
                'max_tilt': config.max_tilt,
                **tier3,
            })

    # Create DataFrame
    if len(recommendations) == 0:
        logger.warning("No recommendations to export")
        return pd.DataFrame()

    result_df = pd.DataFrame(recommendations)

    # Ensure column order
    column_order = [
        'analysisdate',
        'cell_name',
        'cilac',
        'parameter',
        'category',
        'parameter_new_value',
        'cycle_start_date',
        'cycle_end_date',
        'cycle_status',
        'conditions',
        'current_tilt',
        'min_tilt',
        'max_tilt',
        'tier_3_sectors_count',
        'tier_3_cells_count',
        'tier_3_traffic_total',
        'tier_3_drops_total',
        'tier3_traffic_perc',
        'tier3_drops_perc',
    ]

    result_df = result_df[column_order]

    logger.info(
        "Daily resolution recommendations generated",
        total_recommendations=len(result_df),
        overshooters=len(result_df[result_df['category'] == 'overshooter']),
        undershooters=len(result_df[result_df['category'] == 'undershooter']),
    )

    # Save if output_dir provided
    if output_dir is not None:
        output_path = output_dir / config.output_filename
        result_df.to_csv(output_path, index=False)
        logger.info("Daily resolution recommendations saved", path=str(output_path))

    return result_df
