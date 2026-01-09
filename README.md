# RAN Optimizer

A Python package for Radio Access Network (RAN) optimization and cell coverage analysis. Detects network issues including overshooting cells, undershooting cells, coverage gaps, and low coverage areas.

## Features

- **Overshooting Detection**: Identifies cells transmitting beyond their intended coverage area
- **Undershooting Detection**: Finds cells with insufficient coverage reach
- **No Coverage Gap Detection**: Locates areas with no network coverage
- **Low Coverage Detection**: Identifies areas with weak signal strength (per-band)
- **Environment-Aware Parameters**: Automatically adjusts detection thresholds for urban, suburban, and rural environments
- **Band-Aware Analysis**: Considers frequency band when calculating interference

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ran-optimizer

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Using the Runner Script

Run all detection algorithms:

```bash
python -m ran_optimizer.runner \
    --input-dir data/vf-ie/input-data \
    --output-dir data/vf-ie/output-data
```

Run specific algorithms:

```bash
python -m ran_optimizer.runner \
    --algorithms overshooting undershooting \
    --input-dir data/vf-ie/input-data \
    --output-dir data/vf-ie/output-data
```

Use standard (non-environment-aware) detection:

```bash
python -m ran_optimizer.runner \
    --no-environment-aware \
    --input-dir data/vf-ie/input-data \
    --output-dir data/vf-ie/output-data
```

### Using Python API

```python
from ran_optimizer.recommendations.overshooters import OvershooterDetector, OvershooterParams
from ran_optimizer.recommendations.undershooters import UndershooterDetector
import pandas as pd

# Load data
grid_df = pd.read_csv("data/vf-ie/input-data/cell_coverage.csv")
gis_df = pd.read_csv("data/vf-ie/input-data/cork-gis.csv")

# Detect overshooting cells
detector = OvershooterDetector()
overshooters = detector.detect(grid_df, gis_df)
print(f"Found {len(overshooters)} overshooting cells")
```

## Input Data Requirements

### Grid Data (cell_coverage.csv)

Cell measurement data with the following columns:

| Column | Description |
|--------|-------------|
| `grid` or `geohash7` | Geohash7 grid identifier |
| `cilac` or `cell_id` | Cell identifier |
| `avg_rsrp` or `rsrp` | Reference Signal Received Power (dBm) |
| `event_count` or `total_traffic` | Number of measurements |
| `distance_to_cell` or `distance_m` | Distance to serving cell (meters) |
| `Band` or `band` | Frequency band (700, 800, 1800, 2100) |

### GIS Data (cork-gis.csv)

Cell site information with the following columns:

| Column | Description |
|--------|-------------|
| `CILAC` or `cell_id` | Cell identifier |
| `Latitude`, `Longitude` | Cell location |
| `Bearing` or `azimuth_deg` | Antenna azimuth (degrees) |
| `Height` or `height_m` | Antenna height (meters) |
| `TiltM` or `mechanical_tilt` | Mechanical tilt (degrees) |
| `TiltE` or `electrical_tilt` | Electrical tilt (degrees) |
| `Band` | Frequency band |

### Cell Hulls (cell_hulls.csv)

Coverage hull polygons for gap detection:

| Column | Description |
|--------|-------------|
| `cell_name` or `cell_id` | Cell identifier |
| `geometry` | WKT polygon geometry |

## Output Files

The runner produces the following outputs:

| File | Description |
|------|-------------|
| `overshooting_cells_environment_aware.csv` | Overshooting cells with downtilt recommendations |
| `undershooting_cells_environment_aware.csv` | Undershooting cells with uptilt recommendations |
| `no_coverage_clusters.geojson` | Coverage gap polygons |
| `low_coverage_band_{band}.geojson` | Low coverage areas per frequency band |
| `cell_environment.csv` | Cell environment classification (urban/suburban/rural) |
| `maps/enhanced_dashboard.html` | Interactive visualization dashboard |

## Visualization

The runner automatically generates an interactive HTML dashboard combining all detection results:

### Enhanced Dashboard Features

- **Unified Map View**: All 4 algorithm outputs on a single interactive map
- **Filtering**: Filter by issue type, severity, environment, and frequency band
- **Summary Statistics**: Overview panel showing counts and averages
- **Detailed Popups**: Click any cell or area for detailed information and recommendations
- **Export**: Download statistics as JSON for further analysis
- **Layer Control**: Toggle individual layers on/off

### Generating Visualizations

The dashboard is automatically generated when running the full pipeline:

```bash
python -m ran_optimizer.runner --input-dir data/vf-ie/input-data --output-dir data/vf-ie/output-data
# Dashboard saved to: data/vf-ie/output-data/maps/enhanced_dashboard.html
```

You can also generate visualizations separately:

```python
from ran_optimizer.visualization import generate_enhanced_map_from_files

generate_enhanced_map_from_files(
    output_dir="data/vf-ie/output-data",
    gis_file="data/vf-ie/input-data/cork-gis.csv",
    output_file="data/vf-ie/output-data/maps/dashboard.html",
)
```

## Configuration

Algorithm parameters can be customized via JSON config files in the `config/` directory:

- `overshooting_params.json` - Overshooting detection thresholds
- `undershooting_params.json` - Undershooting detection thresholds
- `coverage_gaps.json` - Coverage gap detection parameters

Example config structure:

```json
{
  "base_parameters": {
    "min_cell_distance": 4000,
    "edge_traffic_percent": 0.15
  },
  "environment_overrides": {
    "urban": {
      "min_cell_distance": 2000,
      "edge_traffic_percent": 0.10
    },
    "rural": {
      "min_cell_distance": 6000,
      "edge_traffic_percent": 0.20
    }
  }
}
```

## Project Structure

```
ran-optimizer/
├── ran_optimizer/
│   ├── core/              # Core utilities (environment classification)
│   ├── data/              # Data loaders and schemas
│   ├── recommendations/   # Detection algorithms
│   │   ├── overshooters.py
│   │   ├── undershooters.py
│   │   ├── coverage_gaps.py
│   │   └── environment_aware.py
│   ├── visualization/     # Map visualization
│   │   ├── enhanced_map.py     # Unified dashboard
│   │   └── map_overshooters.py # Detailed cell maps
│   ├── utils/             # Logging, config, geohash utilities
│   ├── validation/        # Result validation
│   └── runner.py          # Unified CLI runner
├── config/                # Algorithm configuration files
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── data/
│   └── vf-ie/             # Vodafone Ireland sample data
│       ├── input-data/
│       └── output-data/
│           └── maps/      # Generated visualizations
└── docs/                  # Additional documentation
```

## Testing

Run unit tests:

```bash
pytest tests/unit/ -v
```

Run all tests:

```bash
pytest tests/ -v
```

## Algorithm Overview

### Overshooting Detection

Identifies cells where signal reaches beyond intended coverage:
1. Find cells with measurements beyond minimum distance threshold
2. Identify "edge bins" (farthest measurements with low traffic share)
3. Calculate relative reach and overshooting grid percentage
4. Flag cells exceeding thresholds
5. Recommend downtilt adjustments

### Undershooting Detection

Identifies cells with insufficient coverage:
1. Find cells with short maximum serving distance
2. Calculate band-aware interference (same-band competition only)
3. Filter by traffic and interference thresholds
4. Calculate uptilt impact on coverage
5. Recommend uptilt adjustments

### Coverage Gap Detection

Identifies areas without network coverage:
1. Cluster cell hulls by proximity
2. Create convex hull for each cluster
3. Find gap polygons (uncovered areas)
4. Apply k-ring density filtering
5. Cluster gaps with HDBSCAN

### Low Coverage Detection

Identifies weak signal areas per frequency band:
1. Find single-server regions (only 1 cell provides coverage)
2. Filter by RSRP threshold
3. Apply k-ring density filtering
4. Cluster with HDBSCAN and create polygons

## License

Proprietary - RADCOM

## Contributing

1. Create a feature branch from `develop`
2. Make changes and add tests
3. Run `pytest tests/` to verify
4. Submit a pull request
