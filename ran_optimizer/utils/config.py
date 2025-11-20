"""
Configuration management using Pydantic for validation.

This module provides type-safe configuration loading and validation
for the RAN Optimizer system.
"""
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import yaml
import os


class DataPaths(BaseModel):
    """Data input/output paths."""
    grid: Path
    gis: Path
    output_base: Path

    @validator('grid', 'gis', pre=True)
    def expand_env_vars(cls, v):
        """Expand environment variables in paths."""
        if isinstance(v, str):
            # Replace ${VAR} with environment variable
            for var in ['DATA_ROOT', 'HOME', 'PWD']:
                if f'${{{var}}}' in v:
                    v = v.replace(f'${{{var}}}', os.environ.get(var, ''))
            return Path(v)
        return v

    @validator('output_base', pre=True)
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist."""
        if isinstance(v, str):
            # Expand env vars first
            for var in ['DATA_ROOT', 'HOME', 'PWD']:
                if f'${{{var}}}' in v:
                    v = v.replace(f'${{{var}}}', os.environ.get(var, ''))
            v = Path(v)
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v


class OvershooterParams(BaseModel):
    """Parameters for overshooting detection algorithm."""
    enabled: bool = True
    edge_traffic_percent: float = Field(0.1, ge=0.0, le=1.0, description="Percentage of edge traffic to analyze")
    min_cell_distance: float = Field(5000.0, gt=0.0, description="Minimum cell range (meters)")
    percent_max_distance: float = Field(0.7, ge=0.0, le=1.0, description="Percentage of max distance threshold")
    min_cell_count_in_grid: int = Field(3, ge=1, description="Minimum cells visible in grid")
    max_percentage_grid_events: float = Field(0.25, ge=0.0, le=1.0, description="Max traffic share in grid")
    rsrp_offset: float = Field(0.8, ge=0.0, le=1.0, description="RSRP degradation tolerance")
    min_overshooting_grids: int = Field(50, ge=1, description="Minimum overshooting grids threshold")
    percentage_overshooting_grids: float = Field(0.05, ge=0.0, le=1.0, description="Percentage of overshooting grids")


class UndershooterParams(BaseModel):
    """Parameters for undershooting detection algorithm."""
    enabled: bool = True
    min_cell_max_distance: float = Field(3000.0, gt=0.0, description="Short-range cell threshold (meters)")
    min_grid_count: int = Field(100, ge=1, description="Minimum grids per cell")
    uptilt_scenarios: list = Field(default_factory=lambda: [1, 2], description="Uptilt degrees to test")
    min_predicted_rsrp: float = Field(-110.0, ge=-144.0, le=-44.0, description="Minimum acceptable RSRP (dBm)")


class InterferenceParams(BaseModel):
    """Parameters for interference detection algorithm."""
    enabled: bool = True
    min_filtered_cells_per_grid: int = Field(4, ge=2, description="Minimum cells for interference consideration")
    min_cell_event_count: int = Field(25, ge=1, description="Minimum samples per cell")
    perc_grid_events: float = Field(0.05, ge=0.0, le=1.0, description="Minimum traffic share")
    dominant_perc_grid_events: float = Field(0.3, ge=0.0, le=1.0, description="Dominance threshold")
    dominance_diff: float = Field(10.0, gt=0.0, description="RSRP difference for dominance (dB)")
    max_rsrp_diff: float = Field(5.0, gt=0.0, description="RSRP clustering width (dB)")
    k_ring: int = Field(3, ge=1, le=5, description="Geohash ring size for spatial clustering")
    perc_interference: float = Field(0.33, ge=0.0, le=1.0, description="Spatial clustering threshold")


class CrossedFeederParams(BaseModel):
    """Parameters for crossed feeder detection algorithm."""
    enabled: bool = True
    min_perc_grid_events: float = Field(0.1, ge=0.0, le=1.0, description="Minimum traffic share in grid")
    min_angular_deviation: float = Field(90.0, ge=0.0, le=180.0, description="Minimum angular misalignment (degrees)")
    top_percent_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Top percentage to flag")


class ProcessingParams(BaseModel):
    """Processing configuration."""
    chunk_size: int = Field(100000, ge=1000, description="Rows per processing chunk")
    n_workers: int = Field(4, ge=1, le=32, description="Number of parallel workers")
    timeout_minutes: int = Field(60, ge=1, description="Processing timeout")
    cache_intermediate: bool = Field(True, description="Cache intermediate results")


class OperatorConfig(BaseModel):
    """Complete configuration for an operator region."""
    operator: str = Field(..., description="Operator name (e.g., DISH, Vodafone_Ireland)")
    region: str = Field(..., description="Region name (e.g., Denver, Cork)")
    data: DataPaths
    features: Dict[str, Any] = Field(default_factory=dict)
    processing: ProcessingParams = Field(default_factory=ProcessingParams)

    def __init__(self, **data):
        """Initialize and parse feature configurations."""
        super().__init__(**data)

        # Parse feature configs with proper types
        if 'overshooters' in self.features:
            if isinstance(self.features['overshooters'], dict):
                self.features['overshooters'] = OvershooterParams(**self.features['overshooters'])

        if 'undershooters' in self.features:
            if isinstance(self.features['undershooters'], dict):
                self.features['undershooters'] = UndershooterParams(**self.features['undershooters'])

        if 'interference' in self.features:
            if isinstance(self.features['interference'], dict):
                self.features['interference'] = InterferenceParams(**self.features['interference'])

        if 'crossed_feeders' in self.features:
            if isinstance(self.features['crossed_feeders'], dict):
                self.features['crossed_feeders'] = CrossedFeederParams(**self.features['crossed_feeders'])

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


def load_config(config_path: Path) -> OperatorConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated OperatorConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        ValidationError: If config validation fails

    Example:
        >>> config = load_config(Path("config/operators/dish_denver.yaml"))
        >>> print(config.operator, config.region)
        DISH Denver
        >>> print(config.features['overshooters'].min_cell_distance)
        5000.0
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return OperatorConfig(**config_dict)


def get_default_config() -> OperatorConfig:
    """
    Get default configuration template.

    Returns:
        Default OperatorConfig
    """
    return OperatorConfig(
        operator="UNKNOWN",
        region="UNKNOWN",
        data=DataPaths(
            grid=Path("./data/input-data/operator/region/grid.csv"),
            gis=Path("./data/input-data/operator/region/gis.csv"),
            output_base=Path("./data/output-data/operator/region")
        ),
        features={
            'overshooters': OvershooterParams(),
            'crossed_feeders': CrossedFeederParams(),
        },
        processing=ProcessingParams()
    )
