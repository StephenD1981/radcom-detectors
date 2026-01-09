"""
Data loading functions.

Provides functions to load grid and GIS data from CSV files.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

from ran_optimizer.utils.exceptions import DataLoadError
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_grid_data(
    file_path: Path,
    sample_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load grid measurement data from CSV.

    Args:
        file_path: Path to grid CSV file
        sample_rows: If specified, only load first N rows (for testing)

    Returns:
        DataFrame with grid measurements

    Raises:
        DataLoadError: If file not found or cannot be read

    Example:
        >>> from pathlib import Path
        >>> df = load_grid_data(Path("data/input-data/cell_coverage.csv"))
        >>> print(f"Loaded {len(df)} measurements")
    """
    if not file_path.exists():
        raise DataLoadError(f"Grid file not found: {file_path}")

    logger.info("loading_grid_data", file=str(file_path), sample_rows=sample_rows)

    try:
        df = pd.read_csv(file_path, nrows=sample_rows)
        logger.info("grid_data_loaded", rows=len(df), columns=len(df.columns))
    except Exception as e:
        raise DataLoadError(f"Failed to read grid CSV: {e}") from e

    return df


def load_gis_data(
    file_path: Path,
    filter_on_air: bool = True
) -> pd.DataFrame:
    """
    Load cell GIS data from CSV.

    Args:
        file_path: Path to GIS CSV file
        filter_on_air: If True, filter to only on-air cells

    Returns:
        DataFrame with cell GIS data

    Raises:
        DataLoadError: If file not found or cannot be read

    Example:
        >>> from pathlib import Path
        >>> df = load_gis_data(Path("data/input-data/cell_gis.csv"))
        >>> print(f"Loaded {len(df)} cells")
    """
    if not file_path.exists():
        raise DataLoadError(f"GIS file not found: {file_path}")

    logger.info("loading_gis_data", file=str(file_path))

    try:
        df = pd.read_csv(file_path)
        logger.info("gis_data_loaded", rows=len(df), columns=len(df.columns))
    except Exception as e:
        raise DataLoadError(f"Failed to read GIS CSV: {e}") from e

    if filter_on_air and 'on_air' in df.columns:
        original_count = len(df)
        df = df[df['on_air'] == True].copy()
        logger.info(
            "filtered_on_air_cells",
            original=original_count,
            filtered=len(df),
            removed=original_count - len(df)
        )

    return df


def load_cell_hulls(file_path: Path) -> 'gpd.GeoDataFrame':
    """
    Load convex hull polygons for cells from CSV file.

    Args:
        file_path: Path to cell_hulls.csv file with WKT geometry

    Returns:
        GeoDataFrame with columns: cell_name, geometry (POLYGON), area_km2

    Raises:
        DataLoadError: If file not found or cannot be read
        ValueError: If geometry column is invalid

    Example:
        >>> from pathlib import Path
        >>> hulls = load_cell_hulls(Path("data/input-data/cell_hulls.csv"))
        >>> print(f"Loaded {len(hulls)} convex hulls")
    """
    if not file_path.exists():
        raise DataLoadError(f"Cell hulls file not found: {file_path}")

    logger.info("loading_cell_hulls", file=str(file_path))

    try:
        import geopandas as gpd
        from shapely import wkt

        df = pd.read_csv(file_path)
        logger.info("cell_hulls_loaded", rows=len(df), columns=len(df.columns))

        required_cols = ['cell_name', 'geometry']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Cell hulls file missing required columns: {missing_cols}")

        try:
            geometries = df['geometry'].apply(wkt.loads)
        except Exception as e:
            raise ValueError(f"Failed to parse WKT geometry: {e}") from e

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

        logger.info(
            "cell_hulls_parsed",
            valid_hulls=len(gdf),
            has_area=('area_km2' in gdf.columns)
        )

        return gdf

    except Exception as e:
        if isinstance(e, (DataLoadError, ValueError)):
            raise
        raise DataLoadError(f"Failed to load cell hulls: {e}") from e
