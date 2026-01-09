"""
Pipeline configuration for production deployment.

Supports JSON configuration with:
- Multiple data sources (CSV, PostgreSQL)
- All 8 detector configurations
- Output format specifications
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class CSVSourceConfig(BaseModel):
    """Configuration for CSV file data source."""
    base_path: Path
    files: Dict[str, str] = Field(
        default_factory=lambda: {
            "coverage": "cell_coverage.csv",
            "gis": "cell_gis.csv",
            "hulls": "cell_hulls.csv",
            "impacts": "cell_impacts.csv"
        }
    )

    @field_validator('base_path', mode='before')
    @classmethod
    def expand_path(cls, v):
        if isinstance(v, str):
            v = _expand_env_vars(v)
            return Path(v)
        return v

    def get_file_path(self, file_key: str) -> Path:
        """Get full path to a data file."""
        if file_key not in self.files:
            raise KeyError(f"Unknown file key: {file_key}. Available: {list(self.files.keys())}")
        return self.base_path / self.files[file_key]


class PostgresSourceConfig(BaseModel):
    """Configuration for PostgreSQL data source."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    tables: Dict[str, str] = Field(
        default_factory=lambda: {
            "coverage": "cell_coverage",
            "gis": "cell_gis",
            "hulls": "cell_hulls",
            "impacts": "cell_impacts"
        }
    )

    @field_validator('host', 'database', 'username', 'password', mode='before')
    @classmethod
    def expand_env(cls, v):
        if isinstance(v, str):
            return _expand_env_vars(v)
        return v

    def get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class InputConfig(BaseModel):
    """Configuration for data inputs."""
    source_type: Literal["csv", "postgres", "mixed"] = "csv"
    csv: CSVSourceConfig = Field(default_factory=lambda: CSVSourceConfig(base_path=Path(".")))
    postgres: PostgresSourceConfig = Field(default_factory=PostgresSourceConfig)


class OutputConfig(BaseModel):
    """Configuration for outputs."""
    base_path: Path
    formats: Dict[str, bool] = Field(
        default_factory=lambda: {
            "geojson": True,
            "csv": True
        }
    )

    @field_validator('base_path', mode='before')
    @classmethod
    def expand_and_create_path(cls, v):
        if isinstance(v, str):
            v = _expand_env_vars(v)
            v = Path(v)
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v


class DetectorConfig(BaseModel):
    """Base configuration for a detector."""
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class LowCoverageConfig(DetectorConfig):
    """Low coverage detector configuration."""
    rsrp_threshold_dbm: float = -115.0
    k_ring_steps: int = 3
    min_missing_neighbors: int = 40
    hdbscan_min_cluster_size: int = 10


class NoCoverageConfig(DetectorConfig):
    """No coverage (gap) detector configuration."""
    cell_cluster_eps_km: float = 5.0
    cell_cluster_min_samples: int = 3
    k_ring_steps: int = 3
    min_missing_neighbors: int = 40
    hdbscan_min_cluster_size: int = 10


class InterferenceConfig(DetectorConfig):
    """Interference detector configuration."""
    min_filtered_cells_per_grid: int = 4
    dominant_perc_grid_events: float = 0.3
    dominance_diff: float = 5.0
    max_rsrp_diff: float = 5.0
    k_ring: int = 3
    perc_interference: float = 0.33


class OvershooterConfig(DetectorConfig):
    """Overshooting detector configuration."""
    edge_traffic_percent: float = 0.15
    min_cell_distance: float = 4000.0
    interference_threshold_db: float = 7.5
    min_cell_count_in_grid: int = 4
    max_percentage_grid_events: float = 0.25
    min_relative_reach: float = 0.70
    rsrp_degradation_db: float = 10.0
    min_overshooting_grids: int = 30
    percentage_overshooting_grids: float = 0.10


class UndershooterConfig(DetectorConfig):
    """Undershooting detector configuration."""
    max_cell_distance: float = 15000.0
    min_cell_event_count: int = 10
    max_interference_percentage: float = 0.10
    interference_threshold_db: float = 7.5
    max_cell_grid_count: int = 4


class CAImbalanceConfig(DetectorConfig):
    """CA imbalance detector configuration."""
    coverage_threshold: float = 0.70
    band_pairs: list = Field(
        default_factory=lambda: [
            {"name": "L800-L1800", "coverage_band": "L800", "capacity_band": "L1800"},
            {"name": "L700-L2100", "coverage_band": "L700", "capacity_band": "L2100"}
        ]
    )


class CrossedFeederConfig(DetectorConfig):
    """Crossed feeder detector configuration."""
    max_radius_m: float = 30000.0
    min_distance_m: float = 500.0
    hbw_cap_deg: float = 60.0
    percentile: float = 0.95


class PCIConfig(DetectorConfig):
    """PCI conflict detector configuration."""
    max_collision_radius_km: float = 30.0
    two_hop_factor: float = 0.25
    min_active_neighbors_after_blacklist: int = 2


class DetectorsConfig(BaseModel):
    """Configuration for all detectors."""
    low_coverage: LowCoverageConfig = Field(default_factory=LowCoverageConfig)
    no_coverage: NoCoverageConfig = Field(default_factory=NoCoverageConfig)
    interference: InterferenceConfig = Field(default_factory=InterferenceConfig)
    overshooters: OvershooterConfig = Field(default_factory=OvershooterConfig)
    undershooters: UndershooterConfig = Field(default_factory=UndershooterConfig)
    ca_imbalance: CAImbalanceConfig = Field(default_factory=CAImbalanceConfig)
    crossed_feeders: CrossedFeederConfig = Field(default_factory=CrossedFeederConfig)
    pci: PCIConfig = Field(default_factory=PCIConfig)

    def get_enabled_detectors(self) -> list:
        """Return list of enabled detector names."""
        enabled = []
        for name in ['low_coverage', 'no_coverage', 'interference', 'overshooters',
                     'undershooters', 'ca_imbalance', 'crossed_feeders', 'pci']:
            if getattr(self, name).enabled:
                enabled.append(name)
        return enabled


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    chunk_size: int = Field(100000, ge=1000)
    n_workers: int = Field(4, ge=1, le=32)
    timeout_minutes: int = Field(60, ge=1)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: str = "1.0"
    operator: str = "unknown"
    region: str = "unknown"
    inputs: InputConfig = Field(default_factory=InputConfig)
    outputs: OutputConfig
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    @model_validator(mode='after')
    def validate_source_config(self):
        """Validate that the selected source type is properly configured."""
        if self.inputs.source_type == "csv":
            if not self.inputs.csv.base_path.exists():
                raise ValueError(f"CSV base path does not exist: {self.inputs.csv.base_path}")
        elif self.inputs.source_type == "postgres":
            if not self.inputs.postgres.enabled:
                raise ValueError("PostgreSQL source selected but not enabled")
            if not self.inputs.postgres.database:
                raise ValueError("PostgreSQL database name required")
        return self


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in string values."""
    pattern = r'\$\{([^}]+)\}'

    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, '')

    return re.sub(pattern, replace_var, value)


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """
    Load pipeline configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        ValidationError: If config validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return PipelineConfig(**config_dict)


def create_default_config(
    input_path: Path,
    output_path: Path,
    operator: str = "vodafone_ireland",
    region: str = "cork"
) -> PipelineConfig:
    """
    Create a default pipeline configuration.

    Args:
        input_path: Base path for input CSV files
        output_path: Base path for outputs
        operator: Operator name
        region: Region name

    Returns:
        Default PipelineConfig
    """
    return PipelineConfig(
        version="1.0",
        operator=operator,
        region=region,
        inputs=InputConfig(
            source_type="csv",
            csv=CSVSourceConfig(base_path=input_path)
        ),
        outputs=OutputConfig(base_path=output_path),
        detectors=DetectorsConfig(),
        processing=ProcessingConfig()
    )


def save_pipeline_config(config: PipelineConfig, config_path: Path) -> None:
    """
    Save pipeline configuration to JSON file.

    Args:
        config: PipelineConfig to save
        config_path: Path to save JSON file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump(mode='json')

    # Convert Path objects to strings
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    config_dict = convert_paths(config_dict)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
