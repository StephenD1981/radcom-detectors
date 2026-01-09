# RAN Optimizer API Reference

| Document Information |                                      |
|---------------------|--------------------------------------|
| **Version**         | 2.0                                  |
| **Classification**  | Technical Reference                  |
| **Last Updated**    | November 2024                        |
| **Audience**        | Software Developers, System Integrators |

---

## Table of Contents

1. [Overview](#overview)
2. [Command Line Interface](#command-line-interface)
3. [Python API](#python-api)
4. [Detection Classes](#detection-classes)
   - [OvershooterDetector](#overshooterdetector)
   - [UndershooterDetector](#undershooterdetector)
   - [CoverageGapDetector](#coveragegapdetector)
   - [LowCoverageDetector](#lowcoveragedetector)
5. [Environment Classification](#environment-classification)
6. [Data Loading Functions](#data-loading-functions)
7. [Validation Framework](#validation-framework)
8. [Visualization API](#visualization-api)
9. [Utility Functions](#utility-functions)
10. [Exception Handling](#exception-handling)
11. [Complete Examples](#complete-examples)

---

## Overview

The RAN Optimizer provides two primary interfaces:

| Interface | Use Case | Entry Point |
|-----------|----------|-------------|
| **CLI** | Production pipelines, automation | `ran-optimize` command |
| **Python API** | Custom integrations, notebooks | `ran_optimizer.*` modules |

### Module Architecture

```
ran_optimizer/
├── runner.py           # High-level orchestration
├── data/
│   └── loaders.py      # Data loading functions
├── recommendations/
│   ├── overshooters.py # Overshooting detection
│   ├── undershooters.py# Undershooting detection
│   ├── coverage_gaps.py# Coverage gap detection
│   └── environment_aware.py  # Environment-specific detection
├── validation/
│   └── validators.py   # Result validation
├── visualization/
│   └── enhanced_map.py # Interactive dashboards
├── core/
│   └── geometry.py     # Geometric calculations
└── utils/
    ├── geohash.py      # Geohash operations
    ├── logging_config.py
    └── error_handling.py
```

---

## Command Line Interface

### Synopsis

```bash
ran-optimize [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input-dir PATH` | Directory containing input data files | `data/input` |
| `--output-dir PATH` | Directory for output files | `data/output` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config-dir PATH` | `config` | Configuration files directory |
| `--algorithms LIST` | all | Space-separated algorithm names |
| `--environment-aware` | enabled | Use environment-specific parameters |
| `--no-environment-aware` | - | Use uniform parameters for all cells |
| `--help` | - | Display help message |

### Algorithm Names

Valid values for `--algorithms`:
- `overshooting` — Detect cells transmitting beyond optimal range
- `undershooting` — Detect cells not reaching full coverage potential
- `no_coverage` — Identify areas with no cellular coverage
- `low_coverage` — Identify areas with weak signal strength

### Usage Examples

**Run all algorithms with defaults:**
```bash
ran-optimize --input-dir data/input --output-dir data/output
```

**Run specific algorithms:**
```bash
ran-optimize --input-dir data/input --output-dir data/output \
  --algorithms overshooting undershooting
```

**Use custom configuration:**
```bash
ran-optimize --input-dir data/input --output-dir data/output \
  --config-dir /etc/ran-optimizer/config
```

**Disable environment-aware detection:**
```bash
ran-optimize --input-dir data/input --output-dir data/output \
  --no-environment-aware
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid arguments |
| 2 | Data loading error |
| 3 | Processing error |
| 4 | Output write error |

---

## Python API

### High-Level Interface

#### `run_all()`

Execute all detection algorithms in a single call.

**Signature:**
```python
def run_all(
    input_dir: Path,
    output_dir: Path,
    algorithms: List[str] = None,
    environment_aware: bool = True,
    config_dir: Path = None,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_dir` | `Path` | Yes | Input data directory |
| `output_dir` | `Path` | Yes | Output directory |
| `algorithms` | `List[str]` | No | Algorithms to run (default: all) |
| `environment_aware` | `bool` | No | Enable environment-specific params (default: True) |
| `config_dir` | `Path` | No | Configuration directory (default: `config`) |

**Returns:**

```python
{
    'overshooting': pd.DataFrame,      # Overshooting recommendations
    'undershooting': pd.DataFrame,     # Undershooting recommendations
    'no_coverage': gpd.GeoDataFrame,   # No coverage gap polygons
    'low_coverage': {                  # Low coverage by frequency band
        700: gpd.GeoDataFrame,
        800: gpd.GeoDataFrame,
        1800: gpd.GeoDataFrame,
        2100: gpd.GeoDataFrame,
    },
    'environment': pd.DataFrame,       # Cell environment classifications
}
```

**Example:**
```python
from pathlib import Path
from ran_optimizer.runner import run_all

results = run_all(
    input_dir=Path("data/input"),
    output_dir=Path("data/output"),
    algorithms=["overshooting", "undershooting"],
    environment_aware=True,
)

print(f"Overshooting cells: {len(results['overshooting'])}")
print(f"Undershooting cells: {len(results['undershooting'])}")
```

---

## Detection Classes

### OvershooterDetector

Identifies cells transmitting beyond their optimal coverage area.

#### Class Definition

```python
class OvershooterDetector:
    def __init__(self, params: OvershooterParams):
        ...

    def detect(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ...
```

#### OvershooterParams

Configuration dataclass for overshooting detection.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `edge_traffic_percent` | `float` | 0.15 | 0.0–1.0 | Fraction of cell distance defining "edge zone" |
| `min_cell_distance` | `int` | 4000 | ≥0 | Minimum cell range to analyze (meters) |
| `percent_max_distance` | `float` | 0.7 | 0.0–1.0 | Distance threshold for expected coverage |
| `min_cell_count_in_grid` | `int` | 4 | ≥1 | Minimum competing cells to flag grid |
| `max_percentage_grid_events` | `float` | 0.25 | 0.0–1.0 | Maximum traffic share in overshooting grid |
| `interference_threshold_db` | `float` | 7.5 | ≥0 | Maximum RSRP gap from P90 for competing cell |
| `min_relative_reach` | `float` | 0.7 | 0.0–1.0 | Cell must reach ≥X% of max distance to grid |
| `rsrp_degradation_db` | `float` | 10.0 | ≥0 | Required RSRP drop from cell's maximum |
| `min_overshooting_grids` | `int` | 30 | ≥0 | Minimum grid count threshold (AND logic) |
| `percentage_overshooting_grids` | `float` | 0.10 | 0.0–1.0 | Percentage threshold (AND logic) |

**Constructor Options:**

```python
# Option 1: Direct instantiation
params = OvershooterParams(
    edge_traffic_percent=0.15,
    min_cell_distance=4000,
    min_overshooting_grids=30,
)

# Option 2: Load from configuration file
params = OvershooterParams.from_config("config/overshooting_params.json")
```

#### detect() Method

**Signature:**
```python
def detect(
    self,
    grid_df: pd.DataFrame,
    gis_df: pd.DataFrame,
) -> pd.DataFrame
```

**Input Requirements:**

`grid_df` must contain:
| Column | Type | Description |
|--------|------|-------------|
| `cilac` | `str` | Cell identifier |
| `grid` | `str` | Geohash grid identifier |
| `rsrp_mean` | `float` | Mean RSRP in dBm |
| `distance_m` | `float` | Distance from cell site (meters) |
| `event_count` | `int` | Number of measurement events |

`gis_df` must contain:
| Column | Type | Description |
|--------|------|-------------|
| `CellName` | `str` | Cell identifier |
| `Latitude` | `float` | Cell site latitude |
| `Longitude` | `float` | Cell site longitude |
| `Bearing` | `float` | Antenna azimuth (degrees) |

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `cell_name` | `str` | Cell identifier |
| `cilac` | `str` | CI-LAC identifier |
| `overshooting_grids` | `int` | Count of overshooting grids |
| `total_grids` | `int` | Total grids served by cell |
| `percentage_overshooting` | `float` | Proportion of grids overshooting |
| `tier_1_grids` | `int` | Critical interference grids (>2km beyond) |
| `tier_2_grids` | `int` | Moderate interference grids (1-2km beyond) |
| `tier_3_grids` | `int` | Minor interference grids (<1km beyond) |
| `max_overshoot_distance` | `float` | Maximum overshoot distance (meters) |
| `edge_traffic_events` | `int` | Traffic events in edge zone |
| `total_traffic_events` | `int` | Total traffic events |

**Example:**
```python
from ran_optimizer.recommendations.overshooters import (
    OvershooterDetector,
    OvershooterParams,
)
from ran_optimizer.data.loaders import load_grid_data, load_gis_data

# Load data
grid_df = load_grid_data("data/input/cell_coverage.csv")
gis_df = load_gis_data("data/input/gis.csv")

# Configure and run detection
params = OvershooterParams(
    min_cell_distance=3000,      # Analyze cells reaching 3km+
    min_overshooting_grids=25,   # Flag if 25+ grids overshooting
)

detector = OvershooterDetector(params)
results = detector.detect(grid_df, gis_df)

# Process results
print(f"Detected {len(results)} overshooting cells")
for _, cell in results.iterrows():
    print(f"  {cell['cell_name']}: {cell['overshooting_grids']} grids "
          f"({cell['percentage_overshooting']:.1%})")
```

---

### UndershooterDetector

Identifies cells not reaching their full coverage potential.

#### Class Definition

```python
class UndershooterDetector:
    def __init__(self, params: UndershooterParams):
        ...

    def detect(
        self,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ...
```

#### UndershooterParams

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_cell_max_distance` | `int` | 3000 | ≥0 | Below this = "short range" cell (meters) |
| `min_grid_count` | `int` | 100 | ≥1 | Minimum data points for analysis |
| `interference_pct_threshold` | `float` | 0.4 | 0.0–1.0 | Edge interference % to trigger flag |
| `traffic_pct_threshold` | `float` | 0.15 | 0.0–1.0 | Edge traffic % required |
| `min_rsrp_threshold` | `float` | -105 | - | Ignore grids below this RSRP (dBm) |

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `cell_name` | `str` | Cell identifier |
| `cilac` | `str` | CI-LAC identifier |
| `max_distance_m` | `float` | Current maximum reach (meters) |
| `edge_interference_pct` | `float` | Proportion of edge with interference |
| `edge_traffic_pct` | `float` | Proportion of traffic at edge |
| `recommended_uptilt` | `float` | Suggested tilt reduction (degrees) |

**Example:**
```python
from ran_optimizer.recommendations.undershooters import (
    UndershooterDetector,
    UndershooterParams,
)

params = UndershooterParams(
    min_cell_max_distance=2500,
    interference_pct_threshold=0.35,
)

detector = UndershooterDetector(params)
results = detector.detect(grid_df, gis_df)
```

---

### CoverageGapDetector

Identifies geographic areas with no cellular coverage.

#### Class Definition

```python
class CoverageGapDetector:
    def __init__(
        self,
        params: CoverageGapParams,
        boundary_shapefile: str = None,
    ):
        ...

    def detect(
        self,
        hulls_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        ...
```

#### CoverageGapParams

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_cluster_points` | `int` | 10 | Minimum geohashes to form cluster |
| `min_cluster_area_km2` | `float` | 0.1 | Minimum cluster area (km²) |
| `cluster_eps` | `float` | 0.003 | HDBSCAN clustering distance |

**Output Schema (GeoDataFrame):**

| Column | Type | Description |
|--------|------|-------------|
| `cluster_id` | `int` | Unique cluster identifier |
| `geometry` | `Polygon` | Gap polygon geometry |
| `area_km2` | `float` | Cluster area (km²) |
| `centroid_lat` | `float` | Cluster centroid latitude |
| `centroid_lon` | `float` | Cluster centroid longitude |
| `geohash_count` | `int` | Number of geohashes in cluster |

**Example:**
```python
from ran_optimizer.recommendations.coverage_gaps import (
    CoverageGapDetector,
    CoverageGapParams,
)
from ran_optimizer.data.loaders import load_cell_hulls

hulls_gdf = load_cell_hulls("data/input/cell_hulls.csv")

params = CoverageGapParams(
    min_cluster_points=15,
    min_cluster_area_km2=0.25,
)

detector = CoverageGapDetector(
    params,
    boundary_shapefile="data/boundaries/region.shp",
)

gaps = detector.detect(hulls_gdf)
print(f"Found {len(gaps)} coverage gaps")
print(f"Total gap area: {gaps['area_km2'].sum():.2f} km²")
```

---

### LowCoverageDetector

Identifies areas with weak signal strength (below threshold).

#### Class Definition

```python
class LowCoverageDetector:
    def __init__(self, params: LowCoverageParams):
        ...

    def detect(
        self,
        hulls_gdf: gpd.GeoDataFrame,
        grid_df: pd.DataFrame,
        gis_df: pd.DataFrame,
        bands: List[int] = None,
    ) -> Dict[int, gpd.GeoDataFrame]:
        ...
```

#### LowCoverageParams

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rsrp_threshold` | `float` | -115 | Signal below this = low coverage (dBm) |
| `min_cluster_points` | `int` | 10 | Minimum geohashes per cluster |
| `k_ring_size` | `int` | 2 | Neighbor ring size for density check |
| `min_density_ratio` | `float` | 0.3 | Required proportion of weak neighbors |

**Example:**
```python
from ran_optimizer.recommendations.coverage_gaps import (
    LowCoverageDetector,
    LowCoverageParams,
)

params = LowCoverageParams(
    rsrp_threshold=-110,  # Stricter threshold
    min_density_ratio=0.4,
)

detector = LowCoverageDetector(params)

results = detector.detect(
    hulls_gdf,
    grid_df,
    gis_df,
    bands=[700, 800, 1800, 2100],
)

for band, gdf in results.items():
    print(f"Band {band} MHz: {len(gdf)} low coverage clusters")
```

---

## Environment Classification

### load_or_create_cell_environments()

Classify cells by deployment environment (urban/suburban/rural).

**Signature:**
```python
def load_or_create_cell_environments(
    gis_df: pd.DataFrame,
    output_path: str = None,
) -> pd.DataFrame
```

**Classification Criteria:**

| Environment | Inter-Site Distance | Typical Deployment |
|-------------|--------------------|--------------------|
| `urban` | < 1,000 m | City centers, dense commercial |
| `suburban` | 1,000–3,000 m | Residential, light commercial |
| `rural` | > 3,000 m | Countryside, highways |

**Output Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | `str` | Cell identifier |
| `environment` | `str` | Classification: `urban`, `suburban`, `rural` |
| `nearest_site_distance` | `float` | Distance to nearest neighboring site (m) |

**Example:**
```python
from ran_optimizer.core import load_or_create_cell_environments

env_df = load_or_create_cell_environments(
    gis_df,
    output_path="output/cell_environments.csv",
)

print(env_df['environment'].value_counts())
# urban       127
# suburban     84
# rural        43
```

---

## Data Loading Functions

### load_grid_data()

Load and validate grid measurement data.

```python
def load_grid_data(
    path: str,
    required_columns: List[str] = None,
) -> pd.DataFrame
```

**Default Required Columns:**
- `cilac`, `grid`, `rsrp_mean`, `distance_m`

**Example:**
```python
from ran_optimizer.data.loaders import load_grid_data

grid_df = load_grid_data(
    "data/input/cell_coverage.csv",
    required_columns=['cilac', 'grid', 'rsrp_mean', 'distance_m', 'event_count'],
)
```

### load_gis_data()

Load and validate cell site geographic data.

```python
def load_gis_data(
    path: str,
    required_columns: List[str] = None,
) -> pd.DataFrame
```

**Default Required Columns:**
- `CellName`, `Latitude`, `Longitude`, `Bearing`

### load_cell_hulls()

Load cell coverage hull geometries.

```python
def load_cell_hulls(path: str) -> gpd.GeoDataFrame
```

**Returns:** GeoDataFrame with `geometry` column containing coverage polygons.

---

## Validation Framework

### OvershootingValidator

Validate overshooting recommendations before implementation.

```python
class OvershootingValidator:
    def validate(self, df: pd.DataFrame) -> ValidationResult
```

#### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool          # Overall validation status
    error_count: int        # Number of blocking errors
    warning_count: int      # Number of warnings
    issues: List[Issue]     # Detailed issue list
```

#### Issue

```python
@dataclass
class Issue:
    severity: str           # 'ERROR' or 'WARNING'
    message: str            # Human-readable description
    cell_id: str            # Affected cell identifier
    field: str              # Related data field
```

#### Validation Rules

| Rule | Threshold | Severity | Description |
|------|-----------|----------|-------------|
| Max downtilt change | >2° | WARNING | Large tilt change may over-correct |
| Max total tilt | >15° | ERROR | Exceeds safe antenna tilt limit |
| Coverage reduction | >50% | WARNING | Significant coverage loss expected |
| Minimum tilt | <0° | ERROR | Negative tilt not physically possible |

**Example:**
```python
from ran_optimizer.validation import OvershootingValidator

validator = OvershootingValidator()
result = validator.validate(overshooting_df)

if not result.is_valid:
    for issue in result.issues:
        if issue.severity == 'ERROR':
            print(f"BLOCKED: {issue.cell_id} - {issue.message}")
```

---

## Visualization API

### create_enhanced_map()

Generate interactive HTML dashboard with all detection results.

**Signature:**
```python
def create_enhanced_map(
    overshooting_df: pd.DataFrame = None,
    undershooting_df: pd.DataFrame = None,
    gis_df: pd.DataFrame = None,
    no_coverage_gdf: gpd.GeoDataFrame = None,
    low_coverage_gdfs: Dict[int, gpd.GeoDataFrame] = None,
    overshooting_grid_df: pd.DataFrame = None,
    undershooting_grid_df: pd.DataFrame = None,
    cell_hulls_gdf: gpd.GeoDataFrame = None,
    output_file: Path = None,
    title: str = "RAN Optimizer Dashboard",
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `overshooting_df` | `DataFrame` | Overshooting cell recommendations |
| `undershooting_df` | `DataFrame` | Undershooting cell recommendations |
| `gis_df` | `DataFrame` | Cell site locations |
| `no_coverage_gdf` | `GeoDataFrame` | No coverage gap polygons |
| `low_coverage_gdfs` | `Dict[int, GeoDataFrame]` | Low coverage by band |
| `overshooting_grid_df` | `DataFrame` | Grid-level overshooting detail |
| `undershooting_grid_df` | `DataFrame` | Grid-level undershooting detail |
| `cell_hulls_gdf` | `GeoDataFrame` | Cell coverage polygons |
| `output_file` | `Path` | Output HTML file path |
| `title` | `str` | Dashboard title |

**Example:**
```python
from pathlib import Path
from ran_optimizer.visualization.enhanced_map import create_enhanced_map

create_enhanced_map(
    overshooting_df=overshooting_results,
    undershooting_df=undershooting_results,
    gis_df=gis_df,
    no_coverage_gdf=gap_clusters,
    low_coverage_gdfs=low_coverage_by_band,
    cell_hulls_gdf=hulls_gdf,
    output_file=Path("output/maps/dashboard.html"),
    title="Cork Region Analysis - Q4 2024",
)
```

---

## Utility Functions

### Geometric Calculations

```python
from ran_optimizer.core.geometry import (
    haversine_distance,
    calculate_bearing,
    bearing_difference,
)
```

#### haversine_distance()

Calculate great-circle distance between two coordinates.

```python
def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float  # Returns meters
```

#### calculate_bearing()

Calculate compass bearing from point 1 to point 2.

```python
def calculate_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float  # Returns degrees (0-360)
```

#### bearing_difference()

Calculate angular difference handling 360° wrap-around.

```python
def bearing_difference(
    bearing1: float,
    bearing2: float,
) -> float  # Returns degrees (0-180)
```

### Geohash Operations

```python
from ran_optimizer.utils.geohash import (
    decode_geohash,
    encode_geohash,
    get_geohash_neighbors,
)
```

#### decode_geohash()

```python
def decode_geohash(geohash: str) -> Tuple[float, float]
# Returns (latitude, longitude)
```

#### encode_geohash()

```python
def encode_geohash(
    lat: float,
    lon: float,
    precision: int = 7,
) -> str
```

#### get_geohash_neighbors()

```python
def get_geohash_neighbors(
    geohash: str,
    k: int = 1,  # Ring size
) -> Set[str]
```

### Logging

```python
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Processing started", cell_count=100)
logger.warning("Missing data", cell_id="ABC123", field="band")
logger.error("Processing failed", error="Out of memory")
```

---

## Exception Handling

### Exception Hierarchy

```
RANOptimizerError (base)
├── DataLoadError        # File not found, format error
├── ValidationError      # Schema validation failure
├── ConfigurationError   # Invalid configuration
└── ProcessingError      # Algorithm execution failure
```

### Error Handling Pattern

```python
from ran_optimizer.data.loaders import load_grid_data
from ran_optimizer.utils.error_handling import (
    DataLoadError,
    ValidationError,
    ConfigurationError,
)

try:
    grid_df = load_grid_data("data/input/cell_coverage.csv")

except DataLoadError as e:
    # File not found, permission denied, corrupt file
    logger.error(f"Data load failed: {e}")
    sys.exit(2)

except ValidationError as e:
    # Missing required columns, invalid data types
    logger.error(f"Data validation failed: {e}")
    sys.exit(2)

except ConfigurationError as e:
    # Invalid parameter values
    logger.error(f"Configuration error: {e}")
    sys.exit(1)
```

---

## Complete Examples

### Example 1: Basic Detection Pipeline

```python
"""
Basic overshooting and undershooting detection.
"""
from pathlib import Path
from ran_optimizer.data.loaders import load_grid_data, load_gis_data
from ran_optimizer.recommendations.overshooters import (
    OvershooterDetector,
    OvershooterParams,
)
from ran_optimizer.recommendations.undershooters import (
    UndershooterDetector,
    UndershooterParams,
)

# Load data
grid_df = load_grid_data("data/input/cell_coverage.csv")
gis_df = load_gis_data("data/input/gis.csv")

# Detect overshooting
over_detector = OvershooterDetector(OvershooterParams())
overshooting = over_detector.detect(grid_df, gis_df)

# Detect undershooting
under_detector = UndershooterDetector(UndershooterParams())
undershooting = under_detector.detect(grid_df, gis_df)

# Output results
print(f"Overshooting: {len(overshooting)} cells")
print(f"Undershooting: {len(undershooting)} cells")

# Save to CSV
overshooting.to_csv("output/overshooting.csv", index=False)
undershooting.to_csv("output/undershooting.csv", index=False)
```

### Example 2: Environment-Aware Detection

```python
"""
Detection with environment-specific parameter tuning.
"""
from pathlib import Path
from ran_optimizer.runner import run_all
from ran_optimizer.core import load_or_create_cell_environments
from ran_optimizer.data.loaders import load_gis_data

# Load GIS data and classify environments
gis_df = load_gis_data("data/input/gis.csv")
env_df = load_or_create_cell_environments(gis_df)

# Run all algorithms with environment awareness
results = run_all(
    input_dir=Path("data/input"),
    output_dir=Path("data/output"),
    environment_aware=True,
)

# Merge environment info with results
overshooting_with_env = results['overshooting'].merge(
    env_df[['cell_id', 'environment']],
    left_on='cell_name',
    right_on='cell_id',
)

# Analyze by environment
for env in ['urban', 'suburban', 'rural']:
    count = len(overshooting_with_env[
        overshooting_with_env['environment'] == env
    ])
    print(f"{env.capitalize()}: {count} overshooting cells")
```

### Example 3: Full Production Pipeline

```python
"""
Complete production pipeline with validation and visualization.
"""
from pathlib import Path
from ran_optimizer.data.loaders import (
    load_grid_data,
    load_gis_data,
    load_cell_hulls,
)
from ran_optimizer.core import load_or_create_cell_environments
from ran_optimizer.recommendations.overshooters import (
    OvershooterDetector,
    OvershooterParams,
)
from ran_optimizer.recommendations.undershooters import (
    UndershooterDetector,
    UndershooterParams,
)
from ran_optimizer.recommendations.coverage_gaps import (
    CoverageGapDetector,
    CoverageGapParams,
)
from ran_optimizer.validation import OvershootingValidator
from ran_optimizer.visualization.enhanced_map import create_enhanced_map
from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)

# Configuration
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load all data
logger.info("Loading data...")
grid_df = load_grid_data(INPUT_DIR / "cell_coverage.csv")
gis_df = load_gis_data(INPUT_DIR / "gis.csv")
hulls_gdf = load_cell_hulls(INPUT_DIR / "cell_hulls.csv")

# 2. Classify environments
logger.info("Classifying cell environments...")
env_df = load_or_create_cell_environments(
    gis_df,
    output_path=OUTPUT_DIR / "cell_environments.csv",
)

# 3. Detect overshooting
logger.info("Running overshooting detection...")
over_params = OvershooterParams(
    min_cell_distance=4000,
    min_overshooting_grids=30,
)
over_detector = OvershooterDetector(over_params)
overshooting = over_detector.detect(grid_df, gis_df)
logger.info(f"Found {len(overshooting)} overshooting cells")

# 4. Validate overshooting recommendations
logger.info("Validating recommendations...")
validator = OvershootingValidator()
validation = validator.validate(overshooting)

if validation.error_count > 0:
    logger.error(f"Validation failed with {validation.error_count} errors")
    for issue in validation.issues:
        if issue.severity == 'ERROR':
            logger.error(f"  {issue.cell_id}: {issue.message}")
else:
    logger.info("All recommendations validated successfully")

# 5. Detect undershooting
logger.info("Running undershooting detection...")
under_detector = UndershooterDetector(UndershooterParams())
undershooting = under_detector.detect(grid_df, gis_df)
logger.info(f"Found {len(undershooting)} undershooting cells")

# 6. Detect coverage gaps
logger.info("Running coverage gap detection...")
gap_detector = CoverageGapDetector(CoverageGapParams())
gaps = gap_detector.detect(hulls_gdf)
logger.info(f"Found {len(gaps)} coverage gaps")

# 7. Save results
logger.info("Saving results...")
overshooting.to_csv(OUTPUT_DIR / "overshooting.csv", index=False)
undershooting.to_csv(OUTPUT_DIR / "undershooting.csv", index=False)
gaps.to_file(OUTPUT_DIR / "coverage_gaps.geojson", driver="GeoJSON")

# 8. Create visualization
logger.info("Generating dashboard...")
create_enhanced_map(
    overshooting_df=overshooting,
    undershooting_df=undershooting,
    gis_df=gis_df,
    no_coverage_gdf=gaps,
    cell_hulls_gdf=hulls_gdf,
    output_file=OUTPUT_DIR / "maps" / "dashboard.html",
    title="RAN Optimization Analysis",
)

logger.info("Pipeline complete!")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"Dashboard: {OUTPUT_DIR / 'maps' / 'dashboard.html'}")
```

---

## Related Documentation

- [[ALGORITHMS]] — Algorithm technical specifications
- [[CONFIGURATION]] — Parameter configuration guide
- [[DATA_FORMATS]] — Input/output data specifications
