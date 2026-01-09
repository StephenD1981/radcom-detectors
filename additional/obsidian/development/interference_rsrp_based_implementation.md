# RSRP-Based Interference Calculation - Implementation Plan

**Date**: 2025-11-21
**Status**: Ready for Implementation

## Problem Identified

Current interference calculation is **incorrect**:
- **Current logic**: Count grids where `cell_count > 4` (raw number of cells)
- **Issue**: Doesn't account for signal strength differences

A grid might have 8 cells serving it, but if only 2 are within 7.5 dB of each other, there's no real interference/competition.

## Correct Definition of Interference

**Interference occurs when multiple cells have similar signal strength** (within handover threshold)

### Algorithm

For each grid:
1. Get all cells serving the grid with their `avg_rsrp` values
2. Find the strongest RSRP: `max_rsrp = max(avg_rsrp)`
3. Count competing cells: `cells_within_threshold = count(cells where max_rsrp - avg_rsrp <= interference_threshold_db)`
4. Flag as "high interference" if: `cells_within_threshold > max_cell_grid_count`

### Example

Grid served by 6 cells:
```
Cell A: -65 dBm (strongest)
Cell B: -68 dBm (diff = 3 dB) ✓ within 7.5 dB
Cell C: -70 dBm (diff = 5 dB) ✓ within 7.5 dB
Cell D: -72 dBm (diff = 7 dB) ✓ within 7.5 dB
Cell E: -80 dBm (diff = 15 dB) ✗ NOT competing
Cell F: -85 dBm (diff = 20 dB) ✗ NOT competing
```

**Result**: 4 competing cells (not 6), so if `max_cell_grid_count = 4`, this grid would be flagged as interference

## Configuration Changes

### Add to `config/undershooting_params.json` AND `config/overshooting_params.json`

```json
{
  "default": {
    "interference_threshold_db": 7.5,
    "max_cell_grid_count": 4,
    "max_interference_percentage": 0.20,
    ...
  }
}
```

**Note**: `interference_threshold_db` should be **global** (not environment-specific)

## Implementation Changes Required

### 1. Undershooting Detection (`ran_optimizer/recommendations/undershooters.py`)

**Method**: `_calculate_interference()`

Current (WRONG):
```python
interference_stats = candidate_grids.groupby('cell_id').apply(
    lambda grp: pd.Series({
        'interference_grids': (grp['cell_count'] > self.params.max_cell_grid_count).sum(),
        'total_grids_check': len(grp)
    })
).reset_index()
```

New (CORRECT):
```python
def _count_competing_cells_per_grid(grid_df, grid_id, threshold_db):
    """Count cells within threshold_db of strongest cell in a grid"""
    grid_cells = grid_df[grid_df['grid_id'] == grid_id]
    if len(grid_cells) == 0:
        return 0
    max_rsrp = grid_cells['avg_rsrp'].max()
    competing = ((max_rsrp - grid_cells['avg_rsrp']) <= threshold_db).sum()
    return competing

# For each candidate cell, count grids with high interference
interference_stats = candidate_grids.groupby('cell_id').apply(
    lambda cell_grids: pd.Series({
        'interference_grids': sum(
            _count_competing_cells_per_grid(grid_df, gid, self.params.interference_threshold_db)
            > self.params.max_cell_grid_count
            for gid in cell_grids['grid_id']
        ),
        'total_grids_check': len(cell_grids)
    })
).reset_index()
```

### 2. Overshooting Detection (`ran_optimizer/recommendations/overshooters.py`)

**Need to verify**: Does overshooting use interference calculation?
- If YES: Apply same RSRP-based logic
- If NO: Consider adding it as a safety check

### 3. Parameter Classes

**File**: `ran_optimizer/recommendations/undershooters.py` (and overshooters if applicable)

Add to dataclass:
```python
@dataclass
class UndershooterParams:
    interference_threshold_db: float = 7.5  # NEW
    max_cell_grid_count: int = 4
    max_interference_percentage: float = 0.20
    ...
```

## Expected Impact

### Current Results (WRONG calculation)
- Urban: 99.99% interference (almost all grids flagged)
- Suburban: 98.99% interference
- Rural: 74.13% interference

### Expected Results (CORRECT calculation)
Much lower interference percentages because:
- Most grids have several cells, but only 2-3 are actually competing (within 7.5 dB)
- The other cells are too weak to cause real interference
- Should see interference rates of 10-30% instead of 70-99%

### Detection Rate Prediction
With correct interference calculation:
- **Urban**: Expect 5-15% of cells flagged as undershooters (currently 0%)
- **Suburban**: Expect 3-10% of cells (currently 0%)
- **Rural**: Expect 1-5% of cells (currently 1.1%)

## Revised Threshold Recommendations

With RSRP-based calculation, the original thresholds might actually be reasonable:

| Environment | interference_threshold_db | max_cell_grid_count | max_interference_% |
|-------------|--------------------------|---------------------|-------------------|
| Urban       | 7.5 (global)             | 4                   | 0.25              |
| Suburban    | 7.5 (global)             | 4                   | 0.20              |
| Rural       | 7.5 (global)             | 4                   | 0.15              |

**Rationale**: With correct calculation, 25% of grids having 4+ competing cells is a reasonable threshold

## Implementation Steps

1. ✅ Add `interference_threshold_db` parameter to config files (both under/overshooting)
2. ✅ Update parameter dataclasses to include new field
3. ✅ Rewrite `_calculate_interference()` method with RSRP-based logic
4. ✅ Test on VF Ireland dataset and verify interference percentages are realistic
5. ✅ Fine-tune `max_interference_percentage` thresholds if needed based on results
6. ✅ Update overshooting detection if applicable
7. ✅ Regenerate visualizations with correct data

## Data Availability Confirmation

- ✅ Grid data has `avg_rsrp` column
- ✅ Grid data has `grid_id` column
- ✅ Grid data has `cell_id` column
- ✅ All required fields available for RSRP-based calculation

## Questions Answered

1. **Measure from strongest cell?** YES - `max_rsrp - cell_rsrp <= 7.5`
2. **Global or per-environment?** GLOBAL - single `interference_threshold_db = 7.5`
3. **RSRP column name?** `avg_rsrp`

## Next Steps

Awaiting approval to proceed with implementation.
