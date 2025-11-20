"""
Data adapters for different operator data formats.

Provides column mapping adapters to convert operator-specific data formats
to the standard GridMeasurement and CellGIS schemas.
"""
import pandas as pd
from typing import Dict, Optional


class DataAdapter:
    """Base class for data format adapters."""

    # Column mappings: {schema_field: data_column}
    GRID_COLUMN_MAP: Dict[str, str] = {}
    GIS_COLUMN_MAP: Dict[str, str] = {}

    @classmethod
    def adapt_grid_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt grid data to standard schema format.

        Args:
            df: DataFrame with operator-specific column names

        Returns:
            DataFrame with standardized column names
        """
        # Rename columns according to mapping
        df_adapted = df.copy()

        # Rename columns that exist
        rename_map = {v: k for k, v in cls.GRID_COLUMN_MAP.items() if v in df.columns}
        df_adapted = df_adapted.rename(columns=rename_map)

        # Apply any custom transformations
        df_adapted = cls._transform_grid_data(df_adapted)

        return df_adapted

    @classmethod
    def adapt_gis_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt GIS data to standard schema format.

        Args:
            df: DataFrame with operator-specific column names

        Returns:
            DataFrame with standardized column names
        """
        # Rename columns according to mapping
        df_adapted = df.copy()

        # Rename columns that exist
        rename_map = {v: k for k, v in cls.GIS_COLUMN_MAP.items() if v in df.columns}
        df_adapted = df_adapted.rename(columns=rename_map)

        # Apply any custom transformations
        df_adapted = cls._transform_gis_data(df_adapted)

        return df_adapted

    @classmethod
    def _transform_grid_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add custom grid transformations."""
        return df

    @classmethod
    def _transform_gis_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add custom GIS transformations."""
        return df


class VodafoneIrelandAdapter(DataAdapter):
    """Adapter for Vodafone Ireland data format."""

    GRID_COLUMN_MAP = {
        'geohash7': 'grid',
        'cell_id': 'global_cell_id',
        'rsrp': 'avg_rsrp',
        'rsrq': 'avg_rsrq',
        'total_traffic': 'eventCount',
        # Note: cell_pci not available in VF data
    }

    GIS_COLUMN_MAP = {
        'cell_id': 'CILAC',
        'site_name': 'SiteID',
        'sector_id': 'SectorID',
        'cell_pci': 'Scr_Freq',  # Scr_Freq is actually PCI (0-503), not frequency
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'azimuth_deg': 'Bearing',
        'mechanical_tilt': 'TiltM',
        'electrical_tilt': 'TiltE',
        'height_m': 'Height',
        'on_air': 'AdminCellState',
        'technology': 'Tech',
        # frequency_mhz not available in VF data
    }

    @classmethod
    def _transform_grid_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Custom transformations for VF Ireland grid data."""
        # Convert cell_id to string if needed
        if 'cell_id' in df.columns:
            df['cell_id'] = df['cell_id'].astype(str)

        # Set cell_pci to 0 as placeholder (not available in VF data)
        df['cell_pci'] = 0

        return df

    @classmethod
    def _transform_gis_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Custom transformations for VF Ireland GIS data."""
        # IMPORTANT: Apply transformations AFTER renaming
        # The adapter rename happens in adapt_gis_data() before calling this

        df = df.copy()  # Avoid SettingWithCopyWarning

        # Convert cell_id to string (CILAC is integer in source)
        if 'cell_id' in df.columns:
            df['cell_id'] = df['cell_id'].astype(str)

        # Convert sector_id to string (SectorID is integer in source)
        if 'sector_id' in df.columns:
            df['sector_id'] = df['sector_id'].astype(str)

        # site_name is coordinate-based, keep as is
        if 'site_name' in df.columns:
            df['site_name'] = df['site_name'].astype(str)

        # Convert on_air to boolean (1 = True, 0 = False)
        if 'on_air' in df.columns:
            df['on_air'] = df['on_air'] == 1

        # Replace NaN in electrical_tilt with 0 (default value)
        if 'electrical_tilt' in df.columns:
            df['electrical_tilt'] = df['electrical_tilt'].fillna(0.0)

        # Replace NaN in mechanical_tilt with 0 (default value)
        if 'mechanical_tilt' in df.columns:
            df['mechanical_tilt'] = df['mechanical_tilt'].fillna(0.0)

        return df


class DISHAdapter(DataAdapter):
    """Adapter for DISH Network data format (future)."""

    GRID_COLUMN_MAP = {
        'geohash7': 'geohash7',
        'cell_id': 'cell_id',
        'cell_pci': 'pci',
        'rsrp': 'rsrp',
        'rsrq': 'rsrq',
        'sinr': 'sinr',
        # DISH data likely matches schema closely
    }

    GIS_COLUMN_MAP = {
        'cell_id': 'cell_id',
        'site_name': 'site_name',
        'latitude': 'lat',
        'longitude': 'lon',
        'azimuth_deg': 'azimuth',
        'mechanical_tilt': 'mechanical_tilt',
        'electrical_tilt': 'electrical_tilt',
        'height_m': 'height',
        # To be updated when DISH data format is confirmed
    }


def get_adapter(operator: str) -> type[DataAdapter]:
    """
    Get the appropriate data adapter for an operator.

    Args:
        operator: Operator name (e.g., 'Vodafone_Ireland', 'DISH')

    Returns:
        DataAdapter class

    Example:
        >>> adapter = get_adapter('Vodafone_Ireland')
        >>> df_grid = adapter.adapt_grid_data(raw_df)
    """
    adapters = {
        'Vodafone_Ireland': VodafoneIrelandAdapter,
        'DISH': DISHAdapter,
    }

    return adapters.get(operator, DataAdapter)
