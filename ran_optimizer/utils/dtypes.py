"""
Data type specifications for CSV file loading.

This module defines explicit dtypes for all data sources to:
1. Eliminate DtypeWarning messages
2. Improve loading performance
3. Ensure consistent data types
4. Reduce memory usage
"""

import numpy as np

# Cell Coverage Dataset (cell_coverage.csv)
# ~1.95M rows, 55 columns, 1.6 GB
CELL_COVERAGE_DTYPES = {
    # Grid identifiers
    'grid_cell': str,
    'grid': str,  # geohash7

    # Cell identifiers
    'cell_name': str,
    'cilac': np.int64,

    # RSRP/RSRQ/SINR metrics (dBm/dB)
    'avg_rsrp': np.float32,
    'avg_rsrq': np.float32,
    'avg_sinr': np.float32,

    # Event counts
    'event_count': np.int32,
    'grid_event_count': np.int32,
    'cell_event_count': np.int32,

    # Aggregated metrics
    'avg_rsrp_grid': np.float32,
    'avg_rsrq_grid': np.float32,
    'avg_sinr_grid': np.float32,
    'avg_rsrp_cell': np.float32,
    'avg_rsrq_cell': np.float32,
    'avg_sinr_cell': np.float32,

    # Percentage metrics
    'perc_cell_events': np.float32,
    'perc_grid_events': np.float32,
    'perc_cell_max_dist': np.float32,
    'perc_dist_reduct_1_dt': np.float32,
    'perc_dist_reduct_2_dt': np.float32,
    'perc_dist_inc_1_ut': np.float32,
    'perc_dist_inc_2_ut': np.float32,

    # Distance metrics (meters)
    'distance_to_cell': np.float32,
    'grid_max_distance_to_cell': np.float32,
    'grid_min_distance_to_cell': np.float32,
    'cell_max_distance_to_cell': np.float32,
    'max_dist_1_dt': np.float32,
    'max_dist_2_dt': np.float32,
    'max_dist_1_ut': np.float32,
    'max_dist_2_ut': np.float32,

    # Angle metrics (degrees)
    'cell_angle_to_grid': np.float32,
    'grid_bearing_diff': np.float32,
    'Bearing': np.float32,

    # Cell configuration
    'Scr_Freq': np.int32,
    'UARFCN': 'Int32',  # Nullable integer (3G/UMTS field - all NA for LTE)
    'TiltE': np.float32,  # Electrical tilt
    'TiltM': np.float32,  # Mechanical tilt
    'SiteID': str,
    'City': str,
    'Height': np.float32,  # meters
    'TAC': str,  # Tracking Area Code (can be mixed int/str)
    'Band': np.int16,  # Frequency band (700, 800, 1800, 2100)
    'Vendor': str,
    'MaxTransPwr': np.float32,
    'FreqMHz': np.float32,
    'HBW': np.float32,  # Horizontal beamwidth

    # Geographic data
    'geometry': str,  # WKT geometry string
    'Latitude': np.float64,  # High precision required
    'Longitude': np.float64,  # High precision required
    'RF_Team': str,

    # Predicted RSRP after tilt changes
    'avg_rsrp_1_degree_downtilt': np.float32,
    'avg_rsrp_2_degree_downtilt': np.float32,
    'avg_rsrp_1_degree_uptilt': np.float32,
    'avg_rsrp_2_degree_uptilt': np.float32,
}


# Overshooting Results (overshooting_cells_full_dataset.csv)
OVERSHOOTING_RESULTS_DTYPES = {
    'cell_name': np.int64,
    'max_distance_m': np.float32,
    'total_grids': np.int32,
    'overshooting_grids': np.int32,
    'percentage_overshooting': np.float32,
    'recommended_tilt_change': np.float32,
    'new_max_distance_1deg_m': np.float32,
    'new_max_distance_2deg_m': np.float32,
    'coverage_reduction_1deg_pct': np.float32,
    'coverage_reduction_2deg_pct': np.float32,
    'severity_score': np.float32,
    'severity_category': str,
    # Add other columns as needed
}


# Undershooting Results (undershooting_cells_full_dataset.csv)
UNDERSHOOTING_RESULTS_DTYPES = {
    'cell_name': np.int64,
    'current_coverage_grids': np.int32,
    'new_coverage_grids': np.int32,
    'total_coverage_after_uptilt': np.int32,
    'current_distance_m': np.float32,
    'distance_increase_m': np.float32,
    'new_max_distance_m': np.float32,
    'coverage_increase_percentage': np.float32,
    'recommended_uptilt_deg': np.float32,
    'interference_percentage': np.float32,
    # Add other columns as needed
}


def load_cell_coverage_csv(file_path: str, **kwargs):
    """
    Load cell_coverage.csv with proper dtypes to eliminate warnings.

    Args:
        file_path: Path to cell_coverage.csv
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        pd.DataFrame: Loaded data with proper dtypes

    Example:
        >>> from ran_optimizer.utils.dtypes import load_cell_coverage_csv
        >>> df = load_cell_coverage_csv('data/cell_coverage.csv')
    """
    import pandas as pd

    # Merge user-provided kwargs with dtypes
    read_kwargs = {'dtype': CELL_COVERAGE_DTYPES}
    read_kwargs.update(kwargs)

    return pd.read_csv(file_path, **read_kwargs)


def get_memory_usage_report(df):
    """
    Generate memory usage report for a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        dict: Memory usage statistics
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    return {
        'total_memory_mb': round(memory_mb, 2),
        'rows': len(df),
        'columns': len(df.columns),
        'memory_per_row_kb': round((memory_mb * 1024) / len(df), 2) if len(df) > 0 else 0,
    }


# Memory optimization tips
DTYPE_OPTIMIZATION_NOTES = """
Memory Optimization Guidelines:
================================

1. Integer Types:
   - Use np.int8 for small integers (-128 to 127)
   - Use np.int16 for medium integers (-32K to 32K)
   - Use np.int32 for large integers (-2B to 2B)
   - Use np.int64 only when necessary

2. Float Types:
   - Use np.float32 for most metrics (sufficient precision)
   - Use np.float64 only for coordinates (lat/lon) requiring high precision

3. String Types:
   - Use 'category' dtype for columns with limited unique values
   - Use str for truly unique identifiers

4. Expected Memory Usage:
   - cell_coverage.csv: ~800 MB with optimized dtypes (vs 1.6 GB)
   - 50% memory reduction possible with proper dtype selection
"""
