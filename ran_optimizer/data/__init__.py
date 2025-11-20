"""
Data loading and validation module.

Provides Pydantic schemas for input data validation and
functions to load and validate grid and GIS data.
"""
from ran_optimizer.data.schemas import GridMeasurement, CellGIS
from ran_optimizer.data.loaders import load_grid_data, load_gis_data
from ran_optimizer.data.adapters import (
    get_adapter,
    VodafoneIrelandAdapter,
    DISHAdapter,
)

__all__ = [
    'GridMeasurement',
    'CellGIS',
    'load_grid_data',
    'load_gis_data',
    'get_adapter',
    'VodafoneIrelandAdapter',
    'DISHAdapter',
]
