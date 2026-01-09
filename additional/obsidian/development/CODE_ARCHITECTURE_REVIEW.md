# Code Architecture & Quality Review

## Overview

This document provides a comprehensive technical review of the codebase architecture, design patterns, code quality, and technical debt.

---

## Architecture Analysis

### Current State: Hybrid Prototype/Production

```
┌─────────────────────────────────────────────────────────┐
│                   PROJECT STRUCTURE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  PRODUCTION-READY           PROTOTYPE                    │
│  ┌───────────────┐         ┌──────────────┐            │
│  │ code-opt-     │         │  explore/    │            │
│  │ data-sources/ │         │  recommend-  │            │
│  │               │         │  ations/     │            │
│  │ ✓ Modular     │         │              │            │
│  │ ✓ Functions   │         │ × Notebooks  │            │
│  │ ✓ Config mgmt │         │ × Inline viz │            │
│  │ ✓ Error hand. │         │ × Hard-coded │            │
│  └───────┬───────┘         └──────┬───────┘            │
│          │                        │                     │
│          └────────┬───────────────┘                     │
│                   ▼                                     │
│           LEGACY SCRIPTS                                │
│           ┌──────────────┐                              │
│           │  code/*.py   │                              │
│           │              │                              │
│           │ × Duplicated │                              │
│           │ × 3 variants │                              │
│           │ × No utils   │                              │
│           └──────────────┘                              │
└─────────────────────────────────────────────────────────┘
```

**Assessment**: The project has **two distinct architectural approaches** that need consolidation.

---

## Module Breakdown

### 1. `code-opt-data-sources/` (Production Grade)

**Strengths**:
- ✅ Clean separation of concerns
- ✅ Centralized configuration (`config.py`)
- ✅ Reusable utility modules
- ✅ Operator-agnostic design
- ✅ Vectorized operations for performance

**File Analysis**:

#### `create-data-sources.py` (464 lines)
**Purpose**: Orchestration script for dataset generation

**Design Pattern**: Pipeline with staged data transforms

**Code Structure**:
```python
# Good practices:
- Operator selection at runtime
- Try/except blocks with descriptive messages
- Function-based transforms
- Timing instrumentation

# Areas for improvement:
- Main loop is sequential (not parallelizable)
- Intermediate dataframes not cleaned up (memory)
- No progress indicators
- Commented-out code (lines 402-421, 429-439)
```

**Critical Section** (Lines 137-142):
```python
hull_hashes = geohash7_inside_hulls_fast(
    hulls,
    precision=7,
    geometry_mode="cell",
    simplify_tolerance_m=10
)
```
This is the **computational bottleneck** (~10-15 minutes). Handles 3K cells × 50K geohashes.

#### `utils/grid_cell_functions.py` (1550 lines)

**Assessment**: ⚠️ **Monolithic utility module** - should be split

**Proposed Refactoring**:
```
grid_cell_functions.py (1550 lines)
    ↓
┌─────────────────────┬──────────────────┬──────────────────┐
│ geometry_utils.py   │ rsrp_models.py   │ tilt_physics.py  │
│ - GeoDataFrame ops  │ - IDW prediction │ - 3GPP antenna   │
│ - Convex hulls      │ - Path loss fit  │ - Distance calc  │
│ - Geohash coverage  │ - Clustering     │ - RSRP projection│
│ (~400 lines)        │ (~500 lines)     │ (~400 lines)     │
└─────────────────────┴──────────────────┴──────────────────┘
```

**Technical Highlights**:

1. **Geohash Grid Coverage** (Lines 338-372)
   - Custom algorithm using aligned cell centers
   - 10x faster than naive point-in-polygon
   - Uses prepared geometries for intersection tests

2. **RSRP Prediction Models** (Lines 664-802)
   - Tiered fallback strategy (same-cell IDW → cell model → global model → median)
   - Vectorized haversine distances
   - cKDTree for O(log n) neighbor lookups
   - Geodesic (WGS84) calculations via PyProj

3. **Tilt Physics** (Lines 809-866)
   - Implements 3GPP vertical antenna pattern
   - Log-distance path loss model
   - Handles both downtilt and uptilt scenarios
   - Edge case handling (0° tilt for uptilt)

**Code Quality Issues**:

| Issue | Lines | Severity | Fix |
|-------|-------|----------|-----|
| Unused variables | 1540-1544 | Low | Remove cleanup code |
| Magic numbers | Throughout | Medium | Extract to constants |
| Long functions | 664-802 (139 lines) | Medium | Break into sub-functions |
| Suppressed warnings | 18-20 | Low | Document rationale |
| No docstring examples | Most functions | Medium | Add usage examples |

#### `utils/interference_functions.py` (262 lines)

**Purpose**: Multi-cell interference detection

**Algorithm Complexity**: O(n × k) where n = grids, k = ring size (7²)

**Key Functions**:

1. **`kring()`** (Lines 13-23) - Geohash ring expansion
   - Uses LRU cache for performance
   - BFS-style neighbor expansion
   - Returns frozenset for hashability

2. **`cluster_with_complete_linkage()`** (Lines 51-86)
   - Scikit-learn AgglomerativeClustering wrapper
   - 5 dB distance threshold (configurable)
   - Handles edge cases (NaN, single value, empty)

3. **`find_interference_cells()`** (Lines 88-258) - Main pipeline
   - **6-stage filter cascade**:
     1. Band-wise processing
     2. RSRP differential clustering
     3. Dominant cell removal
     4. Geospatial clustering (k-ring)
     5. Minimum cluster size
     6. Weight calculation (RSRP-based)

**Performance Bottleneck** (Line 207):
```python
records = [(gh, count_present_in_kring(gh, k, interference_set))
           for gh in interference_set]
```
For 30K grids × 49 neighbors = 1.47M lookups. **Takes ~5-8 minutes**.

**Optimization Opportunity**:
- Pre-compute k-ring for all geohashes
- Store in SQLite/DuckDB for reuse
- Reduce from O(n²) to O(n)

### 2. `code/` (Legacy Grid Enrichment)

**Files**:
- `create-bin-cell-enrichment-vf-ie.py` (366 lines)
- `create-bin-cell-enrichment-dish.py` (Similar structure)
- `create-bin-cell-enrichment-vf-ie-pm.py` (PM variant)

**Issues**:

1. **Code Duplication**: ~80% overlap between files
   - Same functions: `dbm_to_mw()`, `calculate_sinr()`, `findDistance()`
   - Should be in shared utilities module

2. **Hardcoded Paths**:
   ```python
   input_path = "./../data/input-data/vf-ie/grid/"
   enrichment_path = "./../data/input-data/vf-ie/enrichment/"
   gis_path = "./../data/input-data/vf-ie/gis/"
   output_path = "./../data/output-data/vf-ie/grid/"
   ```

3. **No Error Recovery**:
   ```python
   except:
       print("\nIssue loading grid data, exiting program..")
       sys.exit(0)
   ```
   - Loses all progress on error
   - No partial output
   - Non-specific error messages

4. **Inefficient Pandas Operations**:
   ```python
   grid_agg_df['distance_to_cell'], grid_agg_df['cell_angle_to_grid'] = \
       zip(*grid_agg_df.apply(lambda x: findDistance(...), axis=1))
   ```
   - Row-by-row apply (slow)
   - Should vectorize with NumPy

**Recommendation**: **Deprecate** these scripts once `code-opt-data-sources/` is production-ready.

### 3. `explore/recommendations/` (Notebooks)

**Current Usage**: Primary location for recommendation logic

**Architecture Problem**:
- Production algorithms embedded in exploratory notebooks
- No CI/CD integration
- Version control unfriendly (JSON format)
- Can't import/reuse between notebooks

**Migration Path**:
```
Notebooks (Current)                  Python Modules (Target)
────────────────────                 ──────────────────────
tilt-optimisation-                   ran_optimizer/
  overshooters.ipynb     ──────→       recommendations/
                                         overshooters.py
  undershooters.ipynb    ──────→         undershooters.py
  interference.ipynb     ──────→         interference.py
  crossed-feeders.ipynb  ──────→         crossed_feeders.py

(Keep notebooks for)                 (Move logic to)
- Visualization                      - Testable functions
- Ad-hoc analysis                    - Parameterized algorithms
- Prototyping                        - Reusable components
```

**Example Refactoring**:

**Before** (tilt-optimisation-overshooters.ipynb):
```python
# Cell 14 (inline parameters)
edge_traffic_percent = 0.1
min_cell_distance = 5000
percent_max_distance = 0.7
min_cell_count_in_grid = 3
# ... more parameters ...

# Cell 17 (algorithm)
grid_geo_data_tails = grid_geo_data.copy()
grid_geo_data_tails = grid_geo_data_tails.sort_values(...)
grid_geo_data_tails['cum_events'] = grid_geo_data_tails.groupby(...)
# ... 50+ lines of logic ...
```

**After** (ran_optimizer/recommendations/overshooters.py):
```python
from dataclasses import dataclass

@dataclass
class OvershooterConfig:
    edge_traffic_percent: float = 0.1
    min_cell_distance: float = 5000.0
    percent_max_distance: float = 0.7
    min_cell_count_in_grid: int = 3
    # ...

class OvershooterDetector:
    def __init__(self, config: OvershooterConfig):
        self.config = config

    def find_candidates(
        self,
        grid_data: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Identify overshooting cells.

        Args:
            grid_data: Enriched grid-cell measurements

        Returns:
            DataFrame with columns:
                - cell_name
                - grid_count
                - overshooting_grids
                - percentage_overshooting
        """
        # Testable, documented logic here
        ...
```

---

## Design Patterns Analysis

### ✅ Good Patterns

1. **Configuration Management**
   - `config.py` with operator-specific paths
   - Runtime selection via user input
   - Centralized parameter tuning

2. **Functional Decomposition**
   - Small, single-purpose functions in utilities
   - Named intermediate steps (readable pipeline)

3. **Vectorization**
   - NumPy arrays for bulk calculations
   - Avoid row-by-row `apply()` where possible

4. **Caching**
   - `@lru_cache` on `kring()` function
   - Prevents redundant geohash neighbor lookups

### ⚠️ Anti-Patterns

1. **God Objects**
   - `grid_cell_functions.py` - 1550 lines, 30+ functions
   - Single module handles geometry, prediction, physics, I/O

2. **Magic Numbers**
   ```python
   # grid_cell_functions.py:426
   K_SAME_CELL = 8
   K_GLOBAL = 24
   BORE_SIGMA_DEG = 35.0

   # What do these represent?
   # How were they derived?
   # When should they change?
   ```

3. **Swallowed Exceptions**
   ```python
   try:
       # complex operation
   except:  # ← Too broad
       print("Operation failed")  # ← No context
       sys.exit(0)  # ← Loses all work
   ```

4. **Side Effects in Utilities**
   ```python
   def clean_clamp_metrics(df):
       # ...
       df[cols].hist(figsize=(15, 10))  # ← Plotting in data function!
       return df
   ```

5. **Mutable Default Arguments**
   ```python
   # (Not present, but watch for in future code)
   def process(data, config={}):  # ← Dangerous
       config['processed'] = True
   ```

---

## Code Quality Metrics

### Complexity Analysis

| File | Lines | Functions | Avg. Complexity | Max Complexity |
|------|-------|-----------|----------------|----------------|
| grid_cell_functions.py | 1550 | 28 | ~55 lines/fn | 139 lines |
| interference_functions.py | 262 | 6 | ~44 lines/fn | 170 lines |
| create-data-sources.py | 464 | 1 (main) | 464 lines | 464 lines |
| create-bin-...-vf-ie.py | 366 | 13 | ~28 lines/fn | - |

**Target**: <50 lines/function, <200 lines/file

### Type Annotations

**Current State**: 5% coverage (mostly function signatures, no type hints)

**Example**:
```python
# Current
def calculate_distances(df):
    # ...

# Should be
def calculate_distances(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add 'distance_to_cell' and 'cell_max_distance_to_cell' columns.

    Args:
        df: GeoDataFrame with 'geometry', 'Latitude', 'Longitude'

    Returns:
        Input DataFrame with added distance columns

    Raises:
        KeyError: If required columns missing
        ValueError: If geometries invalid
    """
    # ...
```

**Benefits**:
- IDE autocomplete
- Static analysis (mypy)
- Self-documenting interfaces
- Catch bugs before runtime

### Documentation Coverage

| Category | Current | Target |
|----------|---------|--------|
| Module docstrings | 10% | 100% |
| Function docstrings | 40% | 100% |
| Inline comments | 15% | 30% |
| Usage examples | 0% | 50% |
| Architecture docs | PowerPoint | Markdown |

### Error Handling

**Patterns Observed**:

1. **Generic Catch-All** (Most common)
   ```python
   except:
       print("Error occurred")
       sys.exit(0)
   ```

2. **Exception Printed** (Better)
   ```python
   except Exception as e:
       print(f"Function failed: {e}")
   ```

3. **Specific Exceptions** (Rare)
   ```python
   except KeyError as e:
       raise ValueError(f"Missing column: {e}")
   ```

**Recommendation**:
- Use specific exception types
- Log errors (don't just print)
- Provide context for debugging
- Allow graceful degradation where possible

---

## Technical Debt Summary

### High Priority

1. **Notebook → Module Migration** (Effort: 4 weeks)
   - Convert 13 notebooks to Python packages
   - Preserve visualization in separate notebook layer
   - Add unit tests for all algorithms

2. **Modularize grid_cell_functions.py** (Effort: 1 week)
   - Split into 4 focused modules
   - Extract constants to configuration
   - Add comprehensive docstrings

3. **Remove Code Duplication** (Effort: 3 days)
   - Create shared utilities from `code/` scripts
   - Consolidate SINR/distance calculations
   - Deprecate redundant files

4. **Add Type Annotations** (Effort: 2 weeks)
   - Annotate all function signatures
   - Add mypy to CI pipeline
   - Document expected DataFrame schemas

### Medium Priority

5. **Improve Error Handling** (Effort: 1 week)
   - Replace `sys.exit()` with exceptions
   - Add logging framework
   - Create custom exception hierarchy

6. **Performance Optimization** (Effort: 2 weeks)
   - Parallelize hull generation
   - Cache k-ring calculations
   - Profile and optimize hot paths

7. **Add Data Validation** (Effort: 1 week)
   - Pydantic schemas for input data
   - GIS data quality checks
   - Range validation for metrics

### Low Priority

8. **Code Style Consistency** (Effort: 2 days)
   - Apply Black formatter
   - Fix naming conventions
   - Remove commented code

9. **Documentation** (Effort: 1 week)
   - API reference (Sphinx)
   - Algorithm explanations
   - Deployment guides

---

## Recommendations

### Immediate Actions

1. **Create Package Structure**
   ```
   ran_optimizer/
   ├── __init__.py
   ├── core/
   │   ├── geometry.py      (from grid_cell_functions)
   │   ├── rf_models.py     (RSRP, path loss, tilt)
   │   └── interference.py  (interference_functions)
   ├── recommendations/
   │   ├── overshooters.py
   │   ├── undershooters.py
   │   ├── crossed_feeders.py
   │   └── interference.py
   ├── data/
   │   ├── loaders.py
   │   ├── validators.py
   │   └── schemas.py
   └── utils/
       ├── config.py
       └── logging.py
   ```

2. **Add Testing Infrastructure**
   - `pytest` for unit tests
   - `pytest-cov` for coverage reports
   - Sample data fixtures for reproducible tests

3. **Setup CI/CD**
   - GitHub Actions or Jenkins
   - Run tests on every commit
   - Code quality checks (mypy, Black, flake8)

### Long-Term Vision

**Production-Ready Architecture**:

```
┌─────────────────────────────────────────────┐
│          RAN Optimization Platform          │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │  Data Layer  │      │  API Layer      │ │
│  │  - Loaders   │◄─────┤  - REST/GraphQL │ │
│  │  - Validators│      │  - Auth         │ │
│  │  - Caching   │      └────────┬────────┘ │
│  └──────┬───────┘               │          │
│         │                       │          │
│  ┌──────▼──────────────────────▼────────┐ │
│  │     Recommendation Engine             │ │
│  │  ┌───────────┐   ┌─────────────────┐ │ │
│  │  │ Scheduler │   │ Feature Modules │ │ │
│  │  │ - Airflow │   │ - Overshooters  │ │ │
│  │  │ - Triggers│   │ - Interference  │ │ │
│  │  └───────────┘   │ - Crossed Feed  │ │ │
│  │                  └─────────────────┘ │ │
│  └──────────────────────────────────────┘ │
│                                           │
│  ┌──────────────────────────────────────┐ │
│  │         Monitoring & Alerts          │ │
│  │  - Prometheus metrics                │ │
│  │  - Error tracking (Sentry)           │ │
│  │  - Performance dashboards            │ │
│  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## Conclusion

**Current State**: Prototype with production-quality components mixed with research code

**Path Forward**:
1. Preserve proven algorithms
2. Refactor for maintainability
3. Add quality controls (tests, types, docs)
4. Build production infrastructure around core logic

**Timeline Estimate**: 3-4 months for full production readiness
