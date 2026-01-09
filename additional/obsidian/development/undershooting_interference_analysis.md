# Undershooting Detection - Interference Filter Analysis

**Date**: 2025-11-21
**Status**: Issue Identified - Awaiting Decision

## Executive Summary

The undershooting detection is currently producing unrealistic results due to overly strict interference filtering. The interference calculation is working correctly, but reveals that our network has high cell density (which is normal and intentional), causing nearly all candidates to be filtered out in urban/suburban areas.

## Current Test Results

### Environment-Aware Detection Results
- **Total undershooters found**: 9 cells (0.54% of network)
- **Urban**: 0 cells (0.0% of 310 cells)
- **Suburban**: 0 cells (0.0% of 563 cells)
- **Rural**: 9 cells (1.1% of 787 cells)

### Interference Metrics Observed
- **Urban**: 99.99% average interference (154 candidates → 0 after filter)
- **Suburban**: 98.99% average interference (242 candidates → 0 after filter)
- **Rural**: 74.13% average interference (198 candidates → 13 after filter, 9 final)

## Root Cause Analysis

### The Issue
The interference filter is working as designed, but the thresholds are calibrated incorrectly:

1. **Current definition of "interference"**: Grids where `cell_count > max_cell_grid_count`
2. **Current threshold**: `max_cell_grid_count = 4` cells per grid
3. **Reality**: Most grids in urban/suburban areas are served by 4+ cells

### Why This Happens
Modern cellular networks are **intentionally designed** with overlapping coverage:
- **Redundancy**: Multiple cells ensure no coverage gaps
- **Capacity**: Load balancing across multiple cells
- **Handover zones**: Smooth transitions between cells
- **Macro diversity**: Better signal quality through cell selection

**Having 4-6 cells per grid is NORMAL, not "interference"**

## Current Configuration

### Active Thresholds (config/undershooting_params.json)

| Environment | max_cell_distance | max_cell_grid_count | max_interference_% | Result |
|-------------|------------------|---------------------|-------------------|---------|
| Urban       | 3,500m          | 4 (default)         | 25%               | 0 cells |
| Suburban    | 7,000m          | 4 (default)         | 20%               | 0 cells |
| Rural       | 12,000m         | 4 (default)         | 15%               | 9 cells |

### What The Filter Does

For each candidate cell:
1. Count how many of its grids have `cell_count > 4`
2. Calculate `interference_percentage = interfering_grids / total_grids`
3. **Reject** if `interference_percentage > threshold`

**Urban example**: Cell has 100 grids, 99 have >4 cells → 99% interference → REJECTED (>25% threshold)

## The Fundamental Problem

**Current Logic**: "Don't uptilt if cell is in a high-density area"

**Flaw**: High density is the norm in urban/suburban networks, not an exception

**What we actually want**: "Don't uptilt if it would create problematic interference in specific locations"

## Solution Options

### Option 1: Significantly Relax Interference Thresholds (Recommended)
Allow realistic network density while still catching extreme cases:

| Environment | Current Threshold | Proposed Threshold | Rationale |
|-------------|------------------|-------------------|-----------|
| Urban       | 25%              | 95%               | Accept high density; only block if 95%+ of grids have 6+ cells |
| Suburban    | 20%              | 90%               | Moderate density expected; block if 90%+ have 5+ cells |
| Rural       | 15%              | 80%               | Lower density; maintain stricter oversight |

**Also increase `max_cell_grid_count`**:
- Urban: 4 → 6 cells (only flag grids with 7+ cells as "interference")
- Suburban: 4 → 5 cells (only flag grids with 6+ cells)
- Rural: 4 → 4 cells (keep current - rural should have less overlap)

**Pros**:
- Maintains the interference filter for future tuning
- Still catches extreme outliers
- Allows detection in normal-density areas

**Cons**:
- Filter becomes mostly inactive
- May allow uptilt in some inappropriate areas

### Option 2: Remove Interference Filter Entirely
Delete the interference percentage check from the detection logic.

**Pros**:
- Simplest solution
- Other filters (max_cell_distance, min_coverage_increase, min_distance_gain) provide protection
- Acknowledges that "interference" is hard to define

**Cons**:
- Loses potential safeguard against problematic uptilts
- Would require code changes to remove the filter
- Less flexible for future tuning

### Option 3: Redesign Interference Logic
Change from "percentage of grids with high cell count" to something more sophisticated:
- Measure actual RSRP/SINR degradation from uptilt
- Calculate increase in handover events
- Model capacity impact on neighboring cells

**Pros**:
- More accurate representation of real interference
- Could provide better recommendations

**Cons**:
- Requires significant development effort
- Needs additional data (RSRP measurements, network performance metrics)
- May be overengineering for current needs

## Recommendation

**Proceed with Option 1**: Relax interference thresholds to realistic levels.

### Proposed Changes to `config/undershooting_params.json`

```json
"urban": {
  "max_cell_distance": 3500,
  "max_cell_grid_count": 6,
  "max_interference_percentage": 0.95,
  "min_coverage_increase_1deg": 0.05,
  "min_coverage_increase_2deg": 0.10,
  "min_distance_gain_1deg_m": 25,
  "min_distance_gain_2deg_m": 50
},

"suburban": {
  "max_cell_distance": 7000,
  "max_cell_grid_count": 5,
  "max_interference_percentage": 0.90,
  "min_coverage_increase_1deg": 0.04,
  "min_coverage_increase_2deg": 0.08,
  "min_distance_gain_1deg_m": 50,
  "min_distance_gain_2deg_m": 100
},

"rural": {
  "max_cell_distance": 12000,
  "min_cell_event_count": 150,
  "max_cell_grid_count": 4,
  "max_interference_percentage": 0.80,
  "min_coverage_increase_1deg": 0.03,
  "min_coverage_increase_2deg": 0.06,
  "min_distance_gain_1deg_m": 100,
  "min_distance_gain_2deg_m": 200
}
```

### Expected Impact

Based on the test data:
- **Urban**: Should now detect some candidates (currently 154 candidates before interference filter)
- **Suburban**: Should now detect some candidates (currently 242 candidates before interference filter)
- **Rural**: Should remain similar (9 cells, possibly a few more)

**Estimated detection rate**: 3-8% of cells per environment (reasonable for undershooting)

## Alternative Consideration

If we want to be more conservative and see if the other filters are sufficient, we could also:
- Keep `max_cell_grid_count = 4` (current value)
- Just increase `max_interference_percentage` to 0.99 for all environments
- This effectively disables the filter while keeping it in code for future use

This would let us see how many undershooters are found with just the distance, coverage, and traffic filters active.

## Questions for Review

1. **Do these relaxed thresholds make sense for VF Ireland network?**
2. **Should we be more conservative (0.99 for all) or environment-specific (0.95/0.90/0.80)?**
3. **Is the `max_cell_grid_count` adjustment appropriate (6/5/4 cells)?**
4. **Should we proceed with Option 1, or consider Option 2 (remove filter)?**
5. **Do you want to see the results with relaxed thresholds before finalizing?**

## Next Steps (Pending Approval)

1. Update `config/undershooting_params.json` with chosen thresholds
2. Re-run detection test to verify realistic results
3. Review candidate list for sanity check
4. Regenerate unified visualization with corrected undershooting data
5. Document final parameter rationale

---

**Files Modified So Far**:
- `ran_optimizer/recommendations/undershooters.py` - Implemented interference calculation
- `config/undershooting_params.json` - Updated distance thresholds, added distance gain parameters
- Detection results: `data/output-data/vf-ie/recommendations/undershooting_cells_environment_aware.csv`
