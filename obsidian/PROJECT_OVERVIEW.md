# RADCOM RAN Optimization Recommendations Project

## Executive Summary

This project is a comprehensive **Radio Access Network (RAN) optimization system** that generates automated recommendations for cellular network improvements. It analyzes grid-based geolocation data combined with network performance metrics to identify and recommend specific antenna tilt adjustments, detect crossed feeders, identify interference issues, and optimize coverage.

**Project Status**: Research/Prototype Phase
**Target Operators**: Vodafone Ireland (VF-IE) and DISH Network (Denver)
**Primary Technology**: Python-based geospatial analysis with machine learning elements

---

## Project Architecture

### Directory Structure

```
5-radcom-recommendations/
├── code/                          # Legacy grid enrichment scripts (VF-IE & DISH)
├── code-opt-data-sources/         # Production-ready data source generation
├── data/
│   ├── input-data/                # Raw network data (GIS, grid measurements)
│   └── output-data/               # Generated recommendations & datasets
├── docs/                          # PowerPoint presentations & HTML exports
└── explore/                       # Jupyter notebooks for algorithm development
    └── recommendations/           # 13 notebooks implementing features
```

### Core Components

#### 1. Data Sources (`code-opt-data-sources/`)
**Purpose**: Generate tilt-adjusted coverage predictions

**Key Files**:
- `create-data-sources.py` - Main orchestration script
- `config.py` - Centralized configuration for operators
- `utils/grid_cell_functions.py` (1550 lines) - Core geospatial processing
- `utils/interference_functions.py` (262 lines) - Interference detection algorithms

**Capabilities**:
- Creates convex hulls representing cell coverage
- Generates synthetic grid points for tilt scenarios (±1°, ±2°)
- Predicts RSRP changes using 3GPP vertical attenuation models
- Handles both actual measured data and projected coverage areas

#### 2. Grid Enrichment (`code/`)
**Purpose**: Process raw geolocation data into analysis-ready formats

**Scripts**:
- `create-bin-cell-enrichment-vf-ie.py` - Vodafone Ireland processing
- `create-bin-cell-enrichment-dish.py` - DISH network processing
- `create-bin-cell-enrichment-vf-ie-pm.py` - Performance management variant

**Process**:
1. Load grid measurements (RSRP/RSRQ/SINR per geohash)
2. Calculate cell-to-grid distances and bearing differences
3. Aggregate event counts and radio quality metrics
4. Join with GIS data (antenna parameters, location, tilt)
5. Export enriched datasets for recommendation algorithms

#### 3. Recommendation Algorithms (`explore/recommendations/`)

**Implementation**: 13 Jupyter notebooks (prototypes)

| Feature | Notebook | Purpose |
|---------|----------|---------|
| **Overshooters** | tilt-optimisation-overshooters.ipynb | Cells serving too far from site |
| **Undershooters** | tilt-optimisation-undershooters.ipynb | Cells with insufficient coverage |
| **Interference (Grid)** | tilt-optimisation-interference.ipynb | Multi-cell interference in grids |
| **Interference (v2)** | tilt-optimisation-interference-v2.ipynb | Enhanced interference detection |
| **Low Coverage** | tilt-optimisation-low-coverage.ipynb | Coverage gap identification |
| **Crossed Feeders** | crossed-feeders.ipynb | Antenna feed cable swap detection |
| **PCI Optimization** | pci_opt.ipynb | Physical Cell ID conflicts |
| **Impact Enrichment** | impact-enrichment.ipynb | Network event analysis |
| **Data Validation** | check_code_output_files.ipynb | Output quality assurance |

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Geospatial**: GeoPandas, Shapely, Geohash, PyProj
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Folium (interactive maps)

### Key Libraries & Purposes
| Library | Usage |
|---------|-------|
| `geohash` | Grid-based location encoding (precision-7 ≈ 153m × 153m) |
| `shapely` | Polygon operations (convex hulls, expansions) |
| `pyproj` | Geodesic calculations (WGS84 distances) |
| `scipy.spatial.cKDTree` | Fast nearest-neighbor lookups for RSRP prediction |
| `sklearn.cluster.AgglomerativeClustering` | RSRP-based interference grouping |

---

## Data Flow

```
┌─────────────────────┐
│  Raw Network Data   │
│  - GIS (antenna)    │
│  - Grid (drive test)│
│  - Relations        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Grid Enrichment     │ (code/*.py)
│ - Distance calc     │
│ - Bearing alignment │
│ - Metric aggregation│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Source Gen     │ (code-opt-data-sources/)
│ - Convex hulls      │
│ - Tilt projections  │
│ - RSRP prediction   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Recommendations     │ (explore/recommendations/)
│ - Overshooters      │
│ - Interference      │
│ - Crossed feeders   │
│ - Coverage gaps     │
└─────────────────────┘
```

---

## Input Data Requirements

### 1. GIS Data (Cell Attributes)
**Format**: CSV
**Required Columns**:
- `Name` - Cell identifier
- `CILAC` - Combined cell/LAC ID
- `Latitude`, `Longitude` - Site coordinates
- `Bearing` - Azimuth (0-360°)
- `TiltE`, `TiltM` - Electrical/mechanical tilt
- `Height` - Antenna height (meters)
- `HBW` - Horizontal beamwidth
- `Band`, `FreqMHz` - Frequency information

### 2. Grid Data (Measurements)
**Format**: CSV
**Typical Structure**:
- `grid` - Geohash-7 identifier
- `global_cell_id` - Cell serving the grid
- `avg_rsrp`, `avg_rsrq`, `avg_sinr` - Radio quality metrics
- `eventCount` - Number of measurements
- `geometry` - WKT point geometry

### 3. Optional: Cell Relations
- Neighbor cell definitions
- Handover relationships

---

## Key Algorithms

### 1. Tilt Impact Estimation
**Function**: `estimate_distance_after_tilt()` (grid_cell_functions.py:813)

**Physics Model**: 3GPP-style vertical antenna pattern
```python
A_v(θ) = min(12 * ((θ - α) / HPBW_v)², SLA_v)
```
Where:
- `θ` = elevation angle to grid point
- `α` = antenna downtilt (electrical + mechanical)
- `HPBW_v` = vertical half-power beamwidth (default 6.5°)
- `SLA_v` = side-lobe attenuation cap (default 30 dB)

**Path Loss Model**: Log-distance
```python
RSRP(d) = RSRP₀ - α * log₁₀(d/d₀)
```
Path loss exponent `α` fitted per-cell from observed data (fallback: 35 dB/decade).

### 2. RSRP Prediction for Missing Grids
**Function**: `predict_grid_rsrp_wgs84_same_cell_only()` (grid_cell_functions.py:664)

**Method**: Inverse Distance Weighting (IDW) with cell-specific fallbacks
1. Find K=8 nearest known bins from same cell
2. Weight by `1 / distance²`
3. Fallback to per-cell path loss model if < 3 neighbors
4. Final fallback to global model or median RSRP

### 3. Interference Clustering
**Function**: `find_interference_cells()` (interference_functions.py:88)

**Multi-Step Filter**:
1. Identify grids with multiple strong cells (RSRP within 5 dB)
2. Use `AgglomerativeClustering` with 5 dB complete linkage
3. Apply geohash k-ring neighbor analysis (k=3, 49-cell neighborhood)
4. Require 33% of neighbors also show interference
5. Exclude grids with dominant cell (>30% events, 10+ dB stronger)

### 4. Overshooting Detection
**Criteria** (tilt-optimisation-overshooters.ipynb):
- Cell serves grids beyond 70% of max distance
- Grid has 3+ competing cells
- Cell provides <25% of grid's events
- Predicted RSRP drop < 20% if cell removed
- Minimum 50 qualifying grids per cell

### 5. Crossed Feeder Detection
**Function**: `calcScore()` (crossed-feeders.ipynb)

**Scoring Formula**:
```python
score = (impact%) × (distance/max_distance) × (angular_deviation/180°)
```
Where `angular_deviation` measures how far outside the cell's beamwidth pattern an impact occurred.

Top 5% scored cells flagged for investigation.

---

## Output Datasets

### Generated by `create-data-sources.py`

| File | Rows (Denver) | Purpose |
|------|---------------|---------|
| `cell_coverage_complete.csv` | ~881K | Current coverage (all grids) |
| `cell_coverage_1_degree_dt.csv` | Subset | 1° downtilt projection |
| `cell_coverage_2_degree_dt.csv` | Subset | 2° downtilt projection |
| `cell_coverage_1_degree_ut.csv` | Extended | 1° uptilt projection |
| `cell_coverage_2_degree_ut.csv` | Extended | 2° uptilt projection |
| `cell_hulls.csv` | ~3K cells | Convex hull geometries |
| `cell_distance_metrics.csv` | ~3K cells | Max TA per tilt scenario |

### Recommendation Outputs

| Feature | File | Content |
|---------|------|---------|
| Overshooters | `overshooting_final_candidates.csv` | 26 cells (Denver) |
| Interference | `interference-cell-list.csv` | ~184 cells (Denver) |
| Crossed Feeders | `Crossed_Feeders_Results.csv` | 160 suspect cells |

---

## Configuration Management

**File**: `code-opt-data-sources/config.py`

**Key Parameters**:
```python
# Data paths (auto-configured per operator)
INPUT_PATH_DISH  = DATA_ROOT / "input-data" / "dish" / "grid" / "denver"
GIS_PATH_DISH    = DATA_ROOT / "input-data" / "dish" / "gis"
OUTPUT_PATH_DISH = DATA_ROOT / "output-data" / "dish" / "denver" / ...

# Interference module thresholds
min_filtered_cells_per_grid = 3      # Grid complexity
min_cell_event_count = 25            # Sample size
perc_grid_events = 0.05              # 5% contribution
dominant_perc_grid_events = 0.3      # Dominance threshold
max_rsrp_diff = 5                    # dB clustering width
grid_ring = 3                        # Geohash neighborhood
perc_interference = 0.33             # 33% spatial clustering
```

---

## Known Limitations

### 1. **No Version Control**
- Project not in Git repository
- Multiple file versions with unclear lineage
- Risk of configuration drift

### 2. **Prototype Code Quality**
- Notebooks contain production logic (not scripts)
- Heavy use of `try/except` with generic error handling
- Inconsistent naming conventions (`snake_case` vs `camelCase`)
- Hardcoded magic numbers scattered throughout

### 3. **Data Dependencies**
- Requires operator-specific input formats
- No schema validation or data quality checks
- Assumes clean GIS data (no missing coordinates, valid tilts)

### 4. **Performance Issues**
- Grid enrichment: ~15-30 minutes for 1M rows
- Data source generation: ~45 minutes (including hull expansion)
- No parallelization or incremental processing

### 5. **Testing**
- No unit tests
- No integration tests
- Manual validation through notebooks only

### 6. **Documentation**
- PowerPoint-based documentation (non-searchable)
- No API documentation
- Inline comments sparse in critical sections

---

## Operator-Specific Configurations

### Vodafone Ireland (VF-IE)
- **Market**: Cork region
- **Data Source**: MEA grid measurements
- **Grid File**: `grid-cell-data-150m.csv`
- **GIS File**: `cork-gis.csv`
- **Technologies**: LTE (Ericsson)

### DISH Network (Denver)
- **Market**: Denver metro
- **Data Source**: Drive test bins
- **Grid File**: `bins_enrichment_dn.csv`
- **GIS File**: `gis.csv`
- **Technologies**: 5G NR (n70, n71, n66 bands)
- **Unique Features**: Multi-band analysis, NR-specific metrics

---

## Next Steps Required

See [PRODUCTION_PLAN.md](./PRODUCTION_PLAN.md) for detailed roadmap.

**Critical Path**:
1. Convert notebooks to production modules
2. Implement comprehensive testing
3. Add data validation layer
4. Create CI/CD pipeline
5. Develop monitoring/alerting
6. Build user interface for recommendations

---

## Contact & Ownership

**Project Type**: Network optimization research
**Deployment Status**: Offline analysis tool
**Primary Users**: RF engineering teams

**Key Stakeholders**:
- RF optimization engineers (recommendation consumers)
- Data engineering (pipeline maintenance)
- Product management (feature prioritization)
