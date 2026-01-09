# Phase 1: Foundation - Detailed Implementation Plan

## Overview

**Duration**: 4 weeks
**Goal**: Establish production-grade infrastructure foundation
**Team**: Tech Lead (1.0 FTE) + Senior Python Developer (1.0 FTE)
**Success Criteria**: Installable package, version control, configuration management, proper logging

---

## Pre-Phase 1: Assessment & Setup (Week 0)

### Day 1-2: Environment Setup
- [ ] Review all 7 documentation files in `./obsidian/`
- [ ] Audit current codebase structure
- [ ] Document all dependencies (create requirements snapshot)
- [ ] Identify critical vs non-critical code paths
- [ ] Setup development environment standard (Python 3.11+, virtual env)

### Day 3-5: Decision Points
- [ ] **Decision 1**: Keep or archive legacy `code/` scripts?
  - **Recommendation**: Archive (move to `legacy/`) but don't delete
  - **Rationale**: `code-opt-data-sources/` is superior version

- [ ] **Decision 2**: Migrate notebooks immediately or gradually?
  - **Recommendation**: Gradual (keep notebooks, extract logic to modules)
  - **Rationale**: Notebooks useful for visualization/debugging

- [ ] **Decision 3**: Which features to prioritize?
  - **Recommendation**: Overshooters + Crossed Feeders (proven, 70%+ precision)
  - **Defer**: Interference (needs optimization), PCI (incomplete)

### Day 5: Kickoff Meeting
- [ ] Present Phase 1 plan to stakeholders
- [ ] Confirm scope and priorities
- [ ] Establish communication channels (Slack, stand-ups)
- [ ] Setup weekly demo schedule

---

## Week 1: Version Control & Project Structure

### Workstream 1.1: Git Repository Setup (Days 1-2)

**Owner**: Tech Lead

#### Tasks:
```bash
# Day 1 Morning: Initialize Git
cd /path/to/5-radcom-recommendations
git init
git branch -M main

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data files (do NOT commit large CSVs)
data/input-data/**/*.csv
data/output-data/**/*.csv
*.parquet

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Secrets
.env
*.key
credentials.json
EOF

git add .gitignore
git commit -m "Initial commit: Add .gitignore"
```

#### Day 1 Afternoon: Branch Strategy
```bash
# Create develop branch
git checkout -b develop

# Document branching model
cat > BRANCHING_STRATEGY.md << 'EOF'
# Git Branching Strategy

## Branches
- `main`: Production-ready code (protected)
- `develop`: Integration branch
- `feature/*`: Feature development
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

## Workflow
1. Create feature branch from develop: `git checkout -b feature/package-structure develop`
2. Commit changes with descriptive messages
3. Push and create Pull Request to develop
4. After review, merge to develop
5. Periodically, merge develop to main (releases)

## Commit Message Format
<type>: <subject>

Types: feat, fix, docs, refactor, test, chore

Examples:
- feat: Add grid enrichment module
- fix: Correct RSRP calculation in rf_models.py
- docs: Update API documentation
- refactor: Split grid_cell_functions into focused modules
EOF

git add BRANCHING_STRATEGY.md
git commit -m "docs: Add branching strategy"
```

#### Day 2: Repository Documentation
```bash
# Create basic README (simplified version)
cat > README.md << 'EOF'
# RAN Optimization System

Radio Access Network optimization tool for automated antenna tilt recommendations.

## Quick Start
```bash
# Clone repository
git clone <repo-url>
cd ran-optimizer

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Documentation
See `./obsidian/` folder for comprehensive documentation:
- [README](./obsidian/README.md) - Documentation index
- [PROJECT_OVERVIEW](./obsidian/PROJECT_OVERVIEW.md) - System overview
- [PRODUCTION_READINESS_PLAN](./obsidian/PRODUCTION_READINESS_PLAN.md) - Implementation roadmap

## Project Structure
```
ran-optimizer/
├── ran_optimizer/          # Main package (NEW)
├── code-opt-data-sources/  # Data source generation (KEEP)
├── legacy/                 # Archived scripts (MOVED)
├── explore/                # Jupyter notebooks (KEEP)
├── data/                   # Data files (gitignored)
├── tests/                  # Test suite (NEW)
├── config/                 # Configuration files (NEW)
└── docs/                   # Documentation
```

## Development
- Python 3.11+
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines
- Run tests: `pytest tests/`
- Format code: `black ran_optimizer/`
EOF

git add README.md
git commit -m "docs: Add project README"
```

**Deliverables**:
- ✅ Git repository initialized
- ✅ .gitignore configured
- ✅ Branching strategy documented
- ✅ Basic README created

---

### Workstream 1.2: Package Structure (Days 3-5)

**Owner**: Senior Python Developer

#### Day 3: Create Package Skeleton

```bash
# Create feature branch
git checkout -b feature/package-structure develop

# Create package structure
mkdir -p ran_optimizer/{core,data,recommendations,pipeline,utils}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p config/{operators,defaults}
mkdir -p legacy

# Create __init__.py files
touch ran_optimizer/__init__.py
touch ran_optimizer/core/__init__.py
touch ran_optimizer/data/__init__.py
touch ran_optimizer/recommendations/__init__.py
touch ran_optimizer/pipeline/__init__.py
touch ran_optimizer/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ran-optimizer",
    version="0.1.0",
    author="RADCOM Team",
    description="RAN optimization and recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "legacy*", "explore*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ran-optimize=ran_optimizer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF

# Create requirements.txt (from current project)
cat > requirements.txt << 'EOF'
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
geopandas>=0.13.0
shapely>=2.0.0
pyproj>=3.5.0
geohash2>=1.1

# Data processing
scipy>=1.10.0
scikit-learn>=1.2.0

# Configuration
pydantic>=2.0.0
pyyaml>=6.0

# Logging
structlog>=23.0.0

# Visualization (optional)
matplotlib>=3.7.0
folium>=0.14.0
contextily>=1.3.0
EOF

# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pylint>=2.17.0

# Development
ipython>=8.0.0
jupyter>=1.0.0
pre-commit>=3.0.0
EOF

git add setup.py requirements.txt requirements-dev.txt
git commit -m "feat: Add package setup files"
```

#### Day 4: Package Version and Main __init__

```python
# ran_optimizer/__init__.py
"""
RAN Optimizer - Radio Access Network Optimization System

Provides automated recommendations for:
- Antenna tilt adjustments (overshooters/undershooters)
- Interference mitigation
- Crossed feeder detection
- Coverage gap identification
"""

__version__ = "0.1.0"
__author__ = "RADCOM Team"

# Import key classes for convenience
from ran_optimizer.core import (
    calculate_distance,
    estimate_distance_after_tilt,
)

from ran_optimizer.data import (
    load_grid_data,
    load_gis_data,
    validate_grid_data,
    validate_gis_data,
)

__all__ = [
    "__version__",
    "calculate_distance",
    "estimate_distance_after_tilt",
    "load_grid_data",
    "load_gis_data",
    "validate_grid_data",
    "validate_gis_data",
]
```

#### Day 5: Archive Legacy Code

```bash
# Move legacy code
mv code/create-bin-cell-enrichment-*.py legacy/
echo "# Legacy Scripts\n\nArchived grid enrichment scripts. Replaced by ran_optimizer.pipeline.enrichment.\n\nDO NOT USE IN PRODUCTION." > legacy/README.md

git add legacy/
git commit -m "refactor: Archive legacy enrichment scripts"

# Push feature branch
git push -u origin feature/package-structure

# Create Pull Request (in GitHub/GitLab UI)
# Title: "feat: Create ran_optimizer package structure"
# Description: "Initial package structure with setup.py, requirements, and archived legacy code"
```

**Deliverables**:
- ✅ Package structure created
- ✅ setup.py configured
- ✅ requirements.txt defined
- ✅ Legacy code archived
- ✅ Installable package (pip install -e .)

---

## Week 2: Configuration Management

### Workstream 2.1: Configuration System (Days 1-3)

**Owner**: Senior Python Developer

#### Day 1: Configuration Schema

```python
# ran_optimizer/utils/config.py
"""
Configuration management using Pydantic for validation.
"""
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import yaml


class DataPaths(BaseModel):
    """Data input/output paths."""
    grid: Path
    gis: Path
    output_base: Path

    @validator('grid', 'gis', 'output_base')
    def path_must_exist_or_creatable(cls, v):
        """Validate paths."""
        if isinstance(v, str):
            v = Path(v)
        # For input paths, they should exist
        # For output paths, create if needed
        return v


class OvershooterParams(BaseModel):
    """Parameters for overshooting detection."""
    enabled: bool = True
    edge_traffic_percent: float = Field(0.1, ge=0.0, le=1.0)
    min_cell_distance: float = Field(5000.0, gt=0.0)
    percent_max_distance: float = Field(0.7, ge=0.0, le=1.0)
    min_cell_count_in_grid: int = Field(3, ge=1)
    max_percentage_grid_events: float = Field(0.25, ge=0.0, le=1.0)
    rsrp_offset: float = Field(0.8, ge=0.0, le=1.0)
    min_overshooting_grids: int = Field(50, ge=1)
    percentage_overshooting_grids: float = Field(0.05, ge=0.0, le=1.0)


class InterferenceParams(BaseModel):
    """Parameters for interference detection."""
    enabled: bool = True
    min_filtered_cells_per_grid: int = Field(4, ge=2)
    min_cell_event_count: int = Field(25, ge=1)
    perc_grid_events: float = Field(0.05, ge=0.0, le=1.0)
    dominant_perc_grid_events: float = Field(0.3, ge=0.0, le=1.0)
    dominance_diff: float = Field(10.0, gt=0.0)
    max_rsrp_diff: float = Field(5.0, gt=0.0)
    k_ring: int = Field(3, ge=1, le=5)
    perc_interference: float = Field(0.33, ge=0.0, le=1.0)


class CrossedFeederParams(BaseModel):
    """Parameters for crossed feeder detection."""
    enabled: bool = True
    min_perc_grid_events: float = Field(0.1, ge=0.0, le=1.0)
    min_angular_deviation: float = Field(90.0, ge=0.0, le=180.0)
    top_percent_threshold: float = Field(0.05, ge=0.0, le=1.0)


class ProcessingParams(BaseModel):
    """Processing configuration."""
    chunk_size: int = Field(100000, ge=1000)
    n_workers: int = Field(4, ge=1, le=32)
    timeout_minutes: int = Field(60, ge=1)
    cache_intermediate: bool = True


class OperatorConfig(BaseModel):
    """Complete configuration for an operator."""
    operator: str
    region: str
    data: DataPaths
    features: Dict[str, Any] = Field(default_factory=dict)
    processing: ProcessingParams = Field(default_factory=ProcessingParams)

    def __init__(self, **data):
        super().__init__(**data)
        # Parse feature configs
        if 'overshooters' in self.features:
            self.features['overshooters'] = OvershooterParams(**self.features.get('overshooters', {}))
        if 'interference' in self.features:
            self.features['interference'] = InterferenceParams(**self.features.get('interference', {}))
        if 'crossed_feeders' in self.features:
            self.features['crossed_feeders'] = CrossedFeederParams(**self.features.get('crossed_feeders', {}))


def load_config(config_path: Path) -> OperatorConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated OperatorConfig

    Raises:
        ValidationError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Expand environment variables in paths
    if 'data' in config_dict:
        for key in ['grid', 'gis', 'output_base']:
            if key in config_dict['data']:
                path_str = config_dict['data'][key]
                # Simple env var expansion
                if '${' in path_str:
                    import os
                    for env_var in ['DATA_ROOT', 'HOME']:
                        path_str = path_str.replace(f'${{{env_var}}}', os.environ.get(env_var, ''))
                config_dict['data'][key] = path_str

    return OperatorConfig(**config_dict)
```

#### Day 2: Create Configuration Files

```yaml
# config/operators/dish_denver.yaml
operator: DISH
region: Denver

data:
  grid: "${DATA_ROOT}/data/input-data/dish/grid/denver/bins_enrichment_dn.csv"
  gis: "${DATA_ROOT}/data/input-data/dish/gis/gis.csv"
  output_base: "${DATA_ROOT}/data/output-data/dish/denver"

features:
  overshooters:
    enabled: true
    edge_traffic_percent: 0.1
    min_cell_distance: 5000
    percent_max_distance: 0.7
    min_cell_count_in_grid: 3
    max_percentage_grid_events: 0.25
    rsrp_offset: 0.8
    min_overshooting_grids: 50
    percentage_overshooting_grids: 0.05

  crossed_feeders:
    enabled: true
    min_perc_grid_events: 0.1
    min_angular_deviation: 90
    top_percent_threshold: 0.05

  interference:
    enabled: false  # Disabled until optimized
    min_filtered_cells_per_grid: 4
    min_cell_event_count: 25
    perc_grid_events: 0.05
    dominant_perc_grid_events: 0.3
    dominance_diff: 10.0
    max_rsrp_diff: 5.0
    k_ring: 3
    perc_interference: 0.33

processing:
  chunk_size: 100000
  n_workers: 4
  timeout_minutes: 60
  cache_intermediate: true
```

```yaml
# config/operators/vf_ireland_cork.yaml
operator: Vodafone_Ireland
region: Cork

data:
  grid: "${DATA_ROOT}/data/input-data/vf-ie/grid/grid-cell-data-150m.csv"
  gis: "${DATA_ROOT}/data/input-data/vf-ie/gis/cork-gis.csv"
  output_base: "${DATA_ROOT}/data/output-data/vf-ie/cork"

features:
  overshooters:
    enabled: true
    # VF-IE specific tuning
    edge_traffic_percent: 0.15  # Slightly more conservative
    min_cell_distance: 4000     # Smaller cells in urban area
    percent_max_distance: 0.7
    min_cell_count_in_grid: 3
    max_percentage_grid_events: 0.25
    rsrp_offset: 0.8
    min_overshooting_grids: 30  # Lower threshold for smaller dataset
    percentage_overshooting_grids: 0.05

  crossed_feeders:
    enabled: true
    min_perc_grid_events: 0.1
    min_angular_deviation: 90
    top_percent_threshold: 0.05

processing:
  chunk_size: 50000  # Smaller chunks for smaller dataset
  n_workers: 2
  timeout_minutes: 30
  cache_intermediate: true
```

```yaml
# config/defaults/default.yaml
# Default configuration template
operator: UNKNOWN
region: UNKNOWN

data:
  grid: "${DATA_ROOT}/data/input-data/operator/region/grid.csv"
  gis: "${DATA_ROOT}/data/input-data/operator/region/gis.csv"
  output_base: "${DATA_ROOT}/data/output-data/operator/region"

features:
  overshooters:
    enabled: true
    edge_traffic_percent: 0.1
    min_cell_distance: 5000
    percent_max_distance: 0.7
    min_cell_count_in_grid: 3
    max_percentage_grid_events: 0.25
    rsrp_offset: 0.8
    min_overshooting_grids: 50
    percentage_overshooting_grids: 0.05

processing:
  chunk_size: 100000
  n_workers: 4
  timeout_minutes: 60
  cache_intermediate: true
```

#### Day 3: Configuration Usage Example

```python
# tests/unit/test_config.py
"""Tests for configuration management."""
import pytest
from pathlib import Path
from ran_optimizer.utils.config import load_config, OperatorConfig


def test_load_valid_config():
    """Test loading a valid configuration."""
    config = load_config(Path("config/defaults/default.yaml"))
    assert isinstance(config, OperatorConfig)
    assert config.operator == "UNKNOWN"
    assert config.features['overshooters'].enabled is True


def test_config_validation():
    """Test configuration validation."""
    # Invalid edge_traffic_percent (must be 0-1)
    with pytest.raises(ValueError):
        OperatorConfig(
            operator="TEST",
            region="TEST",
            data={
                "grid": "./test.csv",
                "gis": "./gis.csv",
                "output_base": "./output"
            },
            features={
                "overshooters": {
                    "edge_traffic_percent": 1.5  # Invalid!
                }
            }
        )


# Example usage in main code
if __name__ == "__main__":
    # Load configuration
    config = load_config(Path("config/operators/dish_denver.yaml"))

    print(f"Operator: {config.operator}")
    print(f"Region: {config.region}")
    print(f"Grid data: {config.data.grid}")
    print(f"Overshooters enabled: {config.features['overshooters'].enabled}")
    print(f"Min cell distance: {config.features['overshooters'].min_cell_distance}m")
```

**Deliverables**:
- ✅ Pydantic configuration schema
- ✅ YAML configs for DISH Denver and VF-IE Cork
- ✅ Default configuration template
- ✅ Configuration loader with validation
- ✅ Unit tests for config

---

### Workstream 2.2: Logging System (Days 4-5)

**Owner**: Tech Lead

#### Day 4: Structured Logging Setup

```python
# ran_optimizer/utils/logging_config.py
"""
Structured logging configuration using structlog.
"""
import sys
import logging
import structlog
from pathlib import Path
from typing import Optional


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = True
):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_output: If True, output JSON logs; else human-readable
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Processors for structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add JSON or console renderer
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str):
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger
    """
    return structlog.get_logger(name)


# Example usage
if __name__ == "__main__":
    configure_logging(log_level="DEBUG", json_output=False)
    logger = get_logger(__name__)

    logger.info("application_started", operator="DISH", region="Denver")
    logger.debug("processing_grid_data", grid_count=881498)

    try:
        raise ValueError("Example error")
    except ValueError as e:
        logger.error("processing_failed",
                     error=str(e),
                     exc_info=True)
```

#### Day 5: Custom Exception Hierarchy

```python
# ran_optimizer/utils/exceptions.py
"""
Custom exception hierarchy for RAN Optimizer.
"""


class RANOptimizerError(Exception):
    """Base exception for all RAN Optimizer errors."""
    pass


class ConfigurationError(RANOptimizerError):
    """Configuration-related errors."""
    pass


class DataValidationError(RANOptimizerError):
    """Data validation errors."""

    def __init__(self, message: str, invalid_rows: int = 0, details: dict = None):
        super().__init__(message)
        self.invalid_rows = invalid_rows
        self.details = details or {}


class DataLoadError(RANOptimizerError):
    """Data loading errors."""
    pass


class ProcessingError(RANOptimizerError):
    """Processing pipeline errors."""

    def __init__(self, message: str, stage: str, details: dict = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class AlgorithmError(RANOptimizerError):
    """Algorithm execution errors."""
    pass


# Example usage with logging
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def example_function():
    try:
        # Some processing
        raise DataValidationError(
            "Invalid RSRP values found",
            invalid_rows=150,
            details={"min_rsrp": -200, "max_rsrp": -40}
        )
    except DataValidationError as e:
        logger.error("validation_failed",
                     error=str(e),
                     invalid_rows=e.invalid_rows,
                     details=e.details,
                     exc_info=True)
        raise  # Re-raise for caller to handle
```

**Deliverables**:
- ✅ Structured logging configured (structlog)
- ✅ Custom exception hierarchy
- ✅ Logging usage examples
- ✅ JSON and console output support

---

## Week 3: Data Foundation

### Workstream 3.1: Data Schemas (Days 1-3)

**Owner**: Data Engineer (0.5 FTE starts)

```python
# ran_optimizer/data/schemas.py
"""
Pydantic schemas for data validation.
"""
from typing import Optional
from pydantic import BaseModel, Field, validator


class GridMeasurement(BaseModel):
    """Schema for grid measurement data."""
    grid: str = Field(..., regex=r'^[0-9a-z]{7}$')
    global_cell_id: int = Field(..., gt=0)
    avg_rsrp: float = Field(..., ge=-144, le=-44)
    avg_rsrq: Optional[float] = Field(None, ge=-24, le=0)
    avg_sinr: Optional[float] = Field(None, ge=-20, le=30)
    eventCount: int = Field(..., ge=1, alias="event_count")

    class Config:
        allow_population_by_field_name = True

    @validator('avg_rsrp')
    def rsrp_reasonable(cls, v):
        if v > -50:
            raise ValueError(f'RSRP {v} dBm unrealistically high')
        if v < -140:
            raise ValueError(f'RSRP {v} dBm unrealistically low')
        return v


class CellGIS(BaseModel):
    """Schema for cell GIS data."""
    Name: str
    CILAC: int = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)
    Bearing: float = Field(..., ge=0, lt=360)
    TiltE: float = Field(..., ge=0, le=20)
    TiltM: float = Field(..., ge=0, le=20)
    Height: float = Field(..., gt=0, le=200)
    HBW: float = Field(..., ge=30, le=120)
    Band: str
    FreqMHz: float = Field(..., gt=0)

    @validator('Bearing')
    def bearing_valid(cls, v):
        if v < 0 or v >= 360:
            raise ValueError(f'Bearing must be [0, 360), got {v}')
        return v

    @validator('TiltE', 'TiltM')
    def tilt_combined_check(cls, v, values):
        if 'TiltE' in values and 'TiltM' in values:
            total = values.get('TiltE', 0) + values.get('TiltM', 0) + v
            if total > 20:
                raise ValueError(f'Total tilt {total}° exceeds 20°')
        return v
```

```python
# ran_optimizer/data/loaders.py
"""
Data loading utilities with validation.
"""
import pandas as pd
from pathlib import Path
from typing import Optional
from ran_optimizer.data.schemas import GridMeasurement, CellGIS
from ran_optimizer.utils.exceptions import DataLoadError, DataValidationError
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_grid_data(
    file_path: Path,
    validate: bool = True,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and optionally validate grid measurement data.

    Args:
        file_path: Path to grid CSV file
        validate: If True, validate data against schema
        sample_size: If provided, validate only this many random rows

    Returns:
        DataFrame with grid measurements

    Raises:
        DataLoadError: If file cannot be loaded
        DataValidationError: If validation fails
    """
    logger.info("loading_grid_data", file_path=str(file_path))

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise DataLoadError(f"Failed to load grid data: {e}")

    logger.info("grid_data_loaded", rows=len(df), columns=len(df.columns))

    if validate:
        logger.info("validating_grid_data", sample_size=sample_size or "all")
        validate_grid_data(df, sample_size=sample_size)

    return df


def validate_grid_data(
    df: pd.DataFrame,
    sample_size: Optional[int] = None
) -> None:
    """
    Validate grid data against schema.

    Args:
        df: Grid data DataFrame
        sample_size: Number of rows to validate (None = all)

    Raises:
        DataValidationError: If validation fails
    """
    if sample_size:
        df_to_validate = df.sample(n=min(sample_size, len(df)))
    else:
        df_to_validate = df

    errors = []
    for idx, row in df_to_validate.iterrows():
        try:
            GridMeasurement(**row.to_dict())
        except Exception as e:
            errors.append({
                'row': idx,
                'error': str(e)
            })

    if errors:
        logger.warning("validation_errors_found",
                       error_count=len(errors),
                       sample_errors=errors[:5])
        raise DataValidationError(
            f"Found {len(errors)} validation errors",
            invalid_rows=len(errors),
            details={'errors': errors[:10]}  # First 10 errors
        )

    logger.info("validation_passed", rows_validated=len(df_to_validate))


def load_gis_data(
    file_path: Path,
    validate: bool = True
) -> pd.DataFrame:
    """Load and validate GIS data."""
    logger.info("loading_gis_data", file_path=str(file_path))

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise DataLoadError(f"Failed to load GIS data: {e}")

    logger.info("gis_data_loaded", rows=len(df))

    if validate:
        validate_gis_data(df)

    return df


def validate_gis_data(df: pd.DataFrame) -> None:
    """Validate GIS data against schema."""
    errors = []
    for idx, row in df.iterrows():
        try:
            CellGIS(**row.to_dict())
        except Exception as e:
            errors.append({'row': idx, 'error': str(e)})

    if errors:
        logger.warning("gis_validation_errors",
                       error_count=len(errors),
                       sample_errors=errors[:5])
        raise DataValidationError(
            f"Found {len(errors)} GIS validation errors",
            invalid_rows=len(errors),
            details={'errors': errors[:10]}
        )

    logger.info("gis_validation_passed", rows_validated=len(df))
```

**Deliverables**:
- ✅ Pydantic schemas for grid and GIS data
- ✅ Data loaders with validation
- ✅ Error handling with custom exceptions
- ✅ Logging integration

---

### Workstream 3.2: Initial Migration (Days 4-5)

**Owner**: Senior Python Developer

#### Task: Extract First Core Function

```python
# ran_optimizer/core/geometry.py
"""
Geometric calculations for RAN optimization.

Extracted from code-opt-data-sources/utils/grid_cell_functions.py
"""
import numpy as np
import pyproj
from typing import Tuple
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants
EARTH_RADIUS_M = 6371008.8  # WGS84 mean radius


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters

    Example:
        >>> distance = haversine_distance(39.7392, -104.9903, 39.8392, -104.9903)
        >>> print(f"{distance:.2f} meters")
        11119.49 meters
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return EARTH_RADIUS_M * c


def calculate_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Start point (degrees)
        lat2, lon2: End point (degrees)

    Returns:
        Bearing in degrees (0-360, 0=North, clockwise)

    Example:
        >>> bearing = calculate_bearing(39.7392, -104.9903, 39.8392, -104.9903)
        >>> print(f"{bearing:.2f}°")
        0.00°
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x))

    return (bearing + 360) % 360
```

```python
# tests/unit/test_geometry.py
"""Tests for geometry functions."""
import pytest
from ran_optimizer.core.geometry import haversine_distance, calculate_bearing


def test_haversine_distance_same_point():
    """Distance from point to itself should be 0."""
    distance = haversine_distance(39.7392, -104.9903, 39.7392, -104.9903)
    assert distance < 0.1  # Allow small floating point error


def test_haversine_distance_known():
    """Test against known distance."""
    # Denver to Boulder (approx 40km)
    denver = (39.7392, -104.9903)
    boulder = (40.0150, -105.2705)

    distance = haversine_distance(*denver, *boulder)

    # Should be around 40km (allow 5% error)
    assert 38000 < distance < 42000


def test_calculate_bearing_north():
    """Test bearing calculation for due north."""
    bearing = calculate_bearing(39.0, -105.0, 40.0, -105.0)
    assert 359 < bearing < 1 or bearing < 1  # Due north (0°)


def test_calculate_bearing_east():
    """Test bearing calculation for due east."""
    bearing = calculate_bearing(39.0, -105.0, 39.0, -104.0)
    assert 89 < bearing < 91  # Due east (90°)


@pytest.mark.parametrize("lat1,lon1,lat2,lon2,expected_bearing", [
    (0, 0, 1, 0, 0),      # North
    (0, 0, 0, 1, 90),     # East
    (0, 0, -1, 0, 180),   # South
    (0, 0, 0, -1, 270),   # West
])
def test_bearing_cardinal_directions(lat1, lon1, lat2, lon2, expected_bearing):
    """Test bearing for cardinal directions."""
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    assert abs(bearing - expected_bearing) < 5  # 5° tolerance
```

**Deliverables**:
- ✅ First core module extracted (geometry.py)
- ✅ Comprehensive unit tests
- ✅ Documentation with examples
- ✅ Baseline for future migrations

---

## Week 4: Documentation & Review

### Workstream 4.1: Developer Documentation (Days 1-3)

```markdown
# CONTRIBUTING.md

# Contributing to RAN Optimizer

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv or conda)

### Setup
```bash
# Clone repository
git clone <repo-url>
cd ran-optimizer

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Style
- Follow PEP 8
- Use Black for formatting: `black ran_optimizer/`
- Use flake8 for linting: `flake8 ran_optimizer/`
- Type hints required for all functions

### Testing
- Write tests for all new code
- Maintain 80%+ coverage
- Run tests: `pytest tests/`
- Check coverage: `pytest --cov=ran_optimizer tests/`

### Commit Messages
Follow conventional commits format:
```
<type>: <subject>

<body>

<footer>
```

Types: feat, fix, docs, refactor, test, chore

Example:
```
feat: Add RSRP prediction for synthetic grids

Implements IDW-based RSRP prediction for grids not covered
by drive test data. Uses k=8 nearest neighbors from same cell
with distance weighting.

Closes #123
```

## Pull Request Process

1. Create feature branch from `develop`
2. Make changes with tests
3. Ensure all tests pass
4. Update documentation
5. Push and create PR
6. Request review
7. Address feedback
8. Merge after approval

## Code Review Checklist

- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Type hints present
- [ ] Logging added for key operations
- [ ] Error handling appropriate
- [ ] Configuration changes documented
- [ ] Breaking changes noted
```

### Workstream 4.2: API Documentation (Days 4-5)

```python
# Generate API docs using Sphinx (optional but recommended)

# docs/conf.py (Sphinx configuration)
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'RAN Optimizer'
copyright = '2025, RADCOM Team'
author = 'RADCOM Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstring support
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
```

**Deliverables**:
- ✅ CONTRIBUTING.md
- ✅ Code review checklist
- ✅ Developer onboarding guide
- ✅ (Optional) Sphinx API documentation

---

## Phase 1 Completion Checklist

### Week 1: Version Control ✅
- [x] Git repository initialized
- [x] .gitignore configured
- [x] Branching strategy documented
- [x] Package structure created
- [x] setup.py and requirements.txt
- [x] Legacy code archived

### Week 2: Configuration ✅
- [x] Pydantic configuration schemas
- [x] YAML configs for operators
- [x] Configuration loader with validation
- [x] Structured logging (structlog)
- [x] Custom exception hierarchy

### Week 3: Data Foundation ✅
- [x] Data schemas (Pydantic)
- [x] Data loaders with validation
- [x] Error handling integrated
- [x] First core module extracted (geometry.py)
- [x] Unit tests with 80%+ coverage for new code

### Week 4: Documentation ✅
- [x] CONTRIBUTING.md
- [x] Developer documentation
- [x] Code review process
- [x] Example usage code

---

## Success Metrics

**Technical**:
- ✅ Installable package (pip install -e .)
- ✅ Configuration system working (load YAML → Pydantic model)
- ✅ Logging produces structured JSON output
- ✅ Data loaders validate input files
- ✅ 80%+ test coverage for new modules

**Process**:
- ✅ Weekly demos completed (4 demos)
- ✅ All code reviewed before merge
- ✅ Documentation updated for all changes
- ✅ Zero critical bugs in foundation code

**Team**:
- ✅ Team onboarded to new structure
- ✅ Development environment standardized
- ✅ Code review process established

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Team unfamiliar with Pydantic/structlog | Pair programming sessions, code reviews |
| Breaking existing workflows | Keep old code working, migrate gradually |
| Configuration complexity | Start simple, add features incrementally |
| Time pressure | Focus on must-haves, defer nice-to-haves |

---

## Next Steps (Phase 2 Preview)

After Phase 1, you'll have:
- ✅ Version-controlled, installable package
- ✅ Configuration management
- ✅ Logging and error handling
- ✅ Data validation foundation
- ✅ First core modules migrated

Phase 2 will focus on:
- Migrate remaining core modules (rf_models, interference)
- Add data lineage tracking
- Implement data quality reports
- Setup DVC for dataset versioning
- Expand test coverage to 50%+

---

## Questions Before Starting?

**Q: Can we skip any of these tasks?**
A: Configuration and logging are critical. Data schemas can be simplified initially but add them soon.

**Q: What if we have existing code to maintain?**
A: Keep it running in parallel. New code lives in `ran_optimizer/`, old code stays in place.

**Q: How much time for code reviews?**
A: Budget 20% of development time. Fast reviews = fast progress.

**Q: When can we start Phase 2?**
A: After Week 4 demo and stakeholder approval. Don't skip ahead without solid foundation.
