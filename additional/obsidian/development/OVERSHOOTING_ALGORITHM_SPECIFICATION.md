# Overshooting Cell Detection Algorithm - Exact Specification

## Algorithm Overview

A cell is classified as "overshooting" if it meets **ALL** of the following criteria through a multi-step filtering process.

---

## Step 1: Identify Edge Traffic Bins

**Purpose**: Determine which grid bins represent the "far edge" of each cell's coverage area.

**Method**:
- For each cell, calculate the **15th percentile** of distance (edge_traffic_percent = 0.15)
- All bins with `distance >= 15th percentile distance` are considered "edge bins"
- This captures the furthest **85%** of grid bins for each cell

**Parameter**:
```python
edge_traffic_percent = 0.15  # 15% threshold
```

**Result**: Creates an "edge_bins" dataset containing only far-field measurements.

---

## Step 2: Distance Filter

**Purpose**: Only consider cells serving traffic beyond a minimum distance threshold.

**Criteria**:
```python
max_distance_m >= 4000  # Cell must serve at least 4km away
```

**Parameter**:
```python
min_cell_distance = 4000  # 4km minimum
```

**Why**: Short-range cells (e.g., urban microcells) shouldn't be flagged for overshooting.

---

## Step 3: Grid Bin Competition Filter

**Purpose**: Identify bins where the cell has excessive far-field presence despite competition.

For each edge bin, check if:

### Criterion A: Multi-Cell Competition
```python
cells_in_grid >= 3  # At least 3 cells serve this bin
```

### Criterion B: Cell Not Dominant
```python
cell_traffic_pct <= 0.25  # Cell has ≤25% of traffic in this bin
```

**Parameters**:
```python
min_cell_count_in_grid = 3       # Min cells in grid
max_percentage_grid_events = 0.25 # Max 25% cell share
```

**Logic**: If a bin has multiple cells competing AND the target cell isn't dominant, it suggests the cell is reaching too far into areas better served by others.

**Result**: Only edge bins meeting BOTH criteria are flagged as "overshooting bins".

---

## Step 4: RSRP Degradation Check

**Parameter**:
```python
rsrp_offset = 0.8  # 80% RSRP allowed (20% degradation)
```

**Current Implementation**: Simplified - parameter exists but full RSRP validation not yet implemented in filtering logic.

**Intended Logic**: Edge bins should have degraded signal (RSRP < 80% of cell's max RSRP).

---

## Step 5: Final Classification Thresholds

A cell is classified as OVERSHOOTING if it meets **BOTH**:

### Threshold A: Absolute Count
```python
overshooting_grids >= 30  # At least 30 bins flagged
```

### Threshold B: Percentage
```python
percentage_overshooting >= 0.05  # At least 5% of cell's total bins
```

**Parameters**:
```python
min_overshooting_grids = 30           # Minimum 30 bins
percentage_overshooting_grids = 0.05  # Minimum 5%
```

**Calculation**:
```python
percentage_overshooting = overshooting_grids / total_grids
```

---

## Complete Parameter Set

```python
@dataclass
class OvershooterParams:
    # Step 1: Edge traffic threshold
    edge_traffic_percent: float = 0.15

    # Step 2: Distance filters
    min_cell_distance: float = 4000
    percent_max_distance: float = 0.7  # Not currently used

    # Step 3: Grid bin criteria
    min_cell_count_in_grid: int = 3
    max_percentage_grid_events: float = 0.25

    # Step 4: RSRP degradation
    rsrp_offset: float = 0.8  # Not fully implemented

    # Step 5: Final thresholds
    min_overshooting_grids: int = 30
    percentage_overshooting_grids: float = 0.05
```

---

## Algorithm Flow Diagram

```
All Grid Bins (1.95M measurements)
    ↓
[Step 1: Edge Bins Filter]
    → Take bins beyond 15th percentile distance per cell
    → Result: 1.66M edge bins (85%)
    ↓
[Step 2: Distance Filter on Cells]
    → Keep only cells with max_distance >= 4km
    → Result: 1,327 cells (from 1,660)
    ↓
[Step 3: Competition Filter on Bins]
    → Keep edge bins where:
      • ≥3 cells serve the bin AND
      • Cell has ≤25% of bin traffic
    → Result: 939,671 overshooting bins
    ↓
[Step 4: Count per Cell]
    → Group overshooting bins by cell_id
    → Calculate overshooting_grids and percentage per cell
    ↓
[Step 5: Final Thresholds]
    → Keep cells where:
      • overshooting_grids >= 30 AND
      • percentage_overshooting >= 5%
    → Result: 1,302 overshooting cells
```

---

## Current Results Summary

Using default parameters on VF Ireland Cork dataset:

| Metric | Value |
|--------|-------|
| Total cells | 1,660 |
| Overshooting cells | 1,302 (78.4%) |
| Cells passing distance filter | 1,327 |
| Edge bins identified | 1,657,646 (85%) |
| Overshooting bins (after competition) | 939,671 |

---

## Key Observations for Review

### ⚠️ **CRITICAL ISSUE: Frequency Band Not Considered**

**Problem**: The competition filter currently counts ALL cells in a grid bin, regardless of frequency band.

**Current Logic** (Line 318-319):
```python
grid_cell_counts = edge_bins.groupby('geohash7').agg({
    'cell_id': 'nunique',  # Counts all cells regardless of band
    'total_traffic': 'sum',
}).reset_index()
```

**Why This is Wrong**:
- A 1800 MHz cell doesn't compete with 2100 MHz or 700 MHz cells
- They operate on different frequencies - no interference or competition
- Algorithm is **over-counting competition**
- This likely explains the **78.4% overshooting rate** (too high!)

**Correct Logic Should Be**:
```python
# Group by BOTH geohash7 AND frequency band
grid_cell_counts = edge_bins.groupby(['geohash7', 'Band']).agg({
    'cell_id': 'nunique',  # Count cells per band
    'total_traffic': 'sum',
}).reset_index()
```

**Available Data**:
- `Band` column: 700, 800, 1800, 2100 (4 bands in dataset)
- `FreqMHz` column: Available but has NaN values
- Band distribution: 700 (21 cells), 800 (31 cells), 1800 (24 cells), 2100 (12 cells)

**Impact**:
- **HIGH PRIORITY FIX REQUIRED**
- Current results likely have many false positives
- Should re-run analysis after implementing band-aware competition

---

### 1. **Edge Traffic Percent (15%)**
- Currently identifies the **furthest 85%** of bins as "edge"
- This is a very inclusive definition of "edge"
- **Question**: Should "edge" be the furthest 10% or 20% instead?

### 2. **Min Cell Distance (4km)**
- All cells serving beyond 4km are candidates
- In the dataset, 1,327 cells (80%) exceed this
- **Question**: Should this be 5km or 6km for macro cells?

### 3. **Competition Filter (≥3 cells, ≤25% share)**
- Bin must have 3+ cells competing **within same band**
- Cell must not be dominant (<25% traffic share)
- This is fairly lenient
- **Question**: Should require 4+ cells or <20% share?

### 4. **Final Thresholds (30 bins, 5%)**
- 30 bins minimum is relatively low
- 5% percentage is also low
- **Question**: Should this be 50 bins and 10% to focus on severe cases?

### 5. **RSRP Degradation Not Enforced**
- Parameter exists but not actively filtering
- Could add requirement: edge RSRP must be X dB below best RSRP
- **Question**: Should we enforce RSRP degradation check?

---

## Potential Adjustments to Consider

### Conservative (Fewer Overshooters)
```python
edge_traffic_percent = 0.10           # Top 90% = "edge"
min_cell_distance = 6000              # 6km minimum
min_cell_count_in_grid = 4            # 4+ cells competing
max_percentage_grid_events = 0.20     # Max 20% share
min_overshooting_grids = 50           # 50 bins minimum
percentage_overshooting_grids = 0.10  # 10% minimum
```

### Aggressive (More Overshooters)
```python
edge_traffic_percent = 0.20           # Top 80% = "edge"
min_cell_distance = 3000              # 3km minimum
min_cell_count_in_grid = 2            # 2+ cells competing
max_percentage_grid_events = 0.30     # Max 30% share
min_overshooting_grids = 20           # 20 bins minimum
percentage_overshooting_grids = 0.03  # 3% minimum
```

---

## Recommendations for Review

1. **Review edge_traffic_percent**: 15% captures 85% of bins as "edge" - is this too broad?

2. **Validate competition filter**: The 3-cell, 25%-share criteria may be too lenient

3. **Consider stricter final thresholds**: 30 bins and 5% flags a lot of cells - maybe 50 bins and 10%?

4. **Implement RSRP degradation**: Add actual RSRP quality check to filter

5. **Domain expert input**: Network engineers should validate if 78.4% overshooting rate seems realistic

---

## Source Code Location

- Algorithm: `ran_optimizer/recommendations/overshooters.py`
- Parameters: Lines 19-40
- Edge bins: Lines 217-245
- Distance filter: Lines 303-311
- Competition filter: Lines 316-336
- Final thresholds: Lines 363-376
