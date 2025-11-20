"""
Data loading functions with validation.

Provides functions to load and validate grid and GIS data from CSV files,
with comprehensive error handling and data quality reporting.
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
from pydantic import ValidationError

from ran_optimizer.data.schemas import GridMeasurement, CellGIS
from ran_optimizer.utils.exceptions import DataLoadError, DataValidationError
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_grid_data(
    file_path: Path,
    validate: bool = True,
    sample_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and validate grid measurement data from CSV.

    Args:
        file_path: Path to grid CSV file
        validate: If True, validate all rows against GridMeasurement schema
        sample_rows: If specified, only load first N rows (for testing)

    Returns:
        DataFrame with validated grid measurements

    Raises:
        DataLoadError: If file not found or cannot be read
        DataValidationError: If validation fails for too many rows

    Example:
        >>> from pathlib import Path
        >>> df = load_grid_data(Path("data/grid/bins_enrichment.csv"))
        >>> print(f"Loaded {len(df)} measurements")
    """
    # Check file exists
    if not file_path.exists():
        raise DataLoadError(f"Grid file not found: {file_path}")

    logger.info("loading_grid_data", file=str(file_path), sample_rows=sample_rows)

    try:
        # Load CSV
        df = pd.read_csv(file_path, nrows=sample_rows)
        logger.info("grid_data_loaded", rows=len(df), columns=len(df.columns))

    except Exception as e:
        raise DataLoadError(f"Failed to read grid CSV: {e}") from e

    # Validate if requested
    if validate:
        original_row_count = len(df)
        df, validation_errors = _validate_dataframe(df, GridMeasurement)

        if validation_errors:
            error_rate = len(validation_errors) / original_row_count
            logger.warning(
                "grid_validation_errors",
                total_rows=original_row_count,
                invalid_rows=len(validation_errors),
                error_rate=f"{error_rate:.2%}"
            )

            # Fail if >10% of rows are invalid
            if error_rate > 0.10:
                raise DataValidationError(
                    f"Grid data validation failed: {len(validation_errors)} invalid rows",
                    invalid_rows=len(validation_errors),
                    details={
                        'total_rows': original_row_count,
                        'error_rate': error_rate,
                        'sample_errors': validation_errors[:5]  # First 5 errors
                    }
                )

    return df


def load_gis_data(
    file_path: Path,
    validate: bool = True,
    filter_on_air: bool = True
) -> pd.DataFrame:
    """
    Load and validate cell GIS data from CSV.

    Args:
        file_path: Path to GIS CSV file
        validate: If True, validate all rows against CellGIS schema
        filter_on_air: If True, filter to only on-air cells

    Returns:
        DataFrame with validated cell GIS data

    Raises:
        DataLoadError: If file not found or cannot be read
        DataValidationError: If validation fails for too many rows

    Example:
        >>> from pathlib import Path
        >>> df = load_gis_data(Path("data/gis/cell_config.csv"))
        >>> print(f"Loaded {len(df)} cells")
    """
    # Check file exists
    if not file_path.exists():
        raise DataLoadError(f"GIS file not found: {file_path}")

    logger.info("loading_gis_data", file=str(file_path))

    try:
        # Load CSV
        df = pd.read_csv(file_path)
        logger.info("gis_data_loaded", rows=len(df), columns=len(df.columns))

    except Exception as e:
        raise DataLoadError(f"Failed to read GIS CSV: {e}") from e

    # Filter to on-air cells if requested
    if filter_on_air and 'on_air' in df.columns:
        original_count = len(df)
        df = df[df['on_air'] == True].copy()
        logger.info(
            "filtered_on_air_cells",
            original=original_count,
            filtered=len(df),
            removed=original_count - len(df)
        )

    # Validate if requested
    if validate:
        original_row_count = len(df)
        df, validation_errors = _validate_dataframe(df, CellGIS)

        if validation_errors:
            error_rate = len(validation_errors) / original_row_count
            logger.warning(
                "gis_validation_errors",
                total_rows=original_row_count,
                invalid_rows=len(validation_errors),
                error_rate=f"{error_rate:.2%}"
            )

            # Fail if >5% of rows are invalid (stricter for GIS data)
            if error_rate > 0.05:
                raise DataValidationError(
                    f"GIS data validation failed: {len(validation_errors)} invalid rows",
                    invalid_rows=len(validation_errors),
                    details={
                        'total_rows': original_row_count,
                        'error_rate': error_rate,
                        'sample_errors': validation_errors[:5]
                    }
                )

    return df


def _validate_dataframe(
    df: pd.DataFrame,
    schema_class: type[GridMeasurement] | type[CellGIS]
) -> tuple[pd.DataFrame, List[dict]]:
    """
    Validate DataFrame rows against Pydantic schema.

    Args:
        df: DataFrame to validate
        schema_class: Pydantic model class (GridMeasurement or CellGIS)

    Returns:
        Tuple of (validated_df, list_of_validation_errors)
        Invalid rows are dropped from the DataFrame.

    Example:
        >>> df, errors = _validate_dataframe(df, GridMeasurement)
        >>> print(f"Validation errors: {len(errors)}")
    """
    validation_errors = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            # Convert row to dict and validate
            row_dict = row.to_dict()
            schema_class(**row_dict)
            valid_indices.append(idx)

        except ValidationError as e:
            # Record validation error
            validation_errors.append({
                'row_index': idx,
                'errors': e.errors(),
                'row_sample': {k: row_dict.get(k) for k in list(row_dict.keys())[:5]}
            })

    # Return only valid rows
    validated_df = df.loc[valid_indices].copy()

    return validated_df, validation_errors


def get_data_summary(df: pd.DataFrame, data_type: str = "grid") -> dict:
    """
    Generate summary statistics for loaded data.

    Args:
        df: DataFrame with grid or GIS data
        data_type: Type of data ('grid' or 'gis')

    Returns:
        Dictionary with summary statistics

    Example:
        >>> df = load_grid_data(Path("data/grid.csv"))
        >>> summary = get_data_summary(df, "grid")
        >>> print(f"RSRP range: {summary['rsrp_min']} to {summary['rsrp_max']}")
    """
    summary = {
        'total_rows': len(df),
        'columns': list(df.columns),
    }

    if data_type == "grid":
        # Grid-specific stats
        summary.update({
            'rsrp_min': df['rsrp'].min() if 'rsrp' in df.columns else None,
            'rsrp_max': df['rsrp'].max() if 'rsrp' in df.columns else None,
            'rsrp_mean': df['rsrp'].mean() if 'rsrp' in df.columns else None,
            'unique_cells': df['cell_id'].nunique() if 'cell_id' in df.columns else None,
            'unique_geohashes': df['geohash7'].nunique() if 'geohash7' in df.columns else None,
        })

    elif data_type == "gis":
        # GIS-specific stats
        summary.update({
            'unique_sites': df['site_name'].nunique() if 'site_name' in df.columns else None,
            'unique_cells': df['cell_id'].nunique() if 'cell_id' in df.columns else None,
            'avg_height': df['height_m'].mean() if 'height_m' in df.columns else None,
            'avg_mechanical_tilt': df['mechanical_tilt'].mean() if 'mechanical_tilt' in df.columns else None,
            'on_air_cells': df['on_air'].sum() if 'on_air' in df.columns else None,
        })

    return summary
