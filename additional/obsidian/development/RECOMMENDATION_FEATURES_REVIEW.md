# Recommendation Features Review

## Overview

This document reviews each recommendation feature, its algorithm, performance characteristics, and production readiness.

---

## Feature Summary

| Feature | Status | Complexity | Accuracy | Production Ready |
|---------|--------|------------|----------|------------------|
| Overshooters | ✅ Mature | Medium | High | 70% |
| Undershooters | ✅ Mature | Medium | High | 70% |
| Interference (Grid) | ✅ Mature | High | Medium | 60% |
| Interference (Perceived) | ⚠️ Experimental | High | TBD | 40% |
| Crossed Feeders | ✅ Mature | Low | High | 80% |
| Low Coverage | ⚠️ Experimental | Low | Medium | 50% |
| PCI Optimization | 🔧 Prototype | Medium | Unknown | 30% |

---

## Feature 1: Overshooting Cells

### Problem Statement
Cells serving subscribers far beyond optimal range, causing:
- Increased interference to neighbors
- Poor user experience (weak signal at cell edge)
- Inefficient spectrum utilization

### Algorithm

**Location**: `explore/recommendations/tilt-optimisation-overshooters.ipynb`

**Input**: Enriched grid-cell data
**Output**: List of cells recommended for downtilt (±1° or ±2°)

**Multi-Stage Filter**:

```
Stage 1: Edge Traffic Identification
──────────────────────────────────────
Input: All grid-cell pairs
Filter: Keep furthest 10% of traffic per cell (by cumulative event %)
Output: "Edge samples" dataset

Stage 2: Qualifying Grid Selection
──────────────────────────────────────
For each grid in edge samples:
  ✓ Cell serves to ≥5km from site
  ✓ Grid is ≥70% of cell's max TA distance
  ✓ Grid has ≥3 competing cells visible
  ✓ Cell provides ≤25% of grid's traffic
  ✓ Predicted RSRP drop ≤20% if cell removed

Stage 3: Cell-Level Aggregation
──────────────────────────────────────
For each cell:
  • Count qualifying "overshooting grids"
  • Calculate % of cell's total grids affected

Stage 4: Thresholding
──────────────────────────────────────
Keep cells with:
  • ≥50 overshooting grids (absolute)
  • ≥5% of grids overshooting (relative)
```

### Tuneable Parameters

```python
# Default values from notebook
edge_traffic_percent = 0.1          # Top 10% farthest traffic
min_cell_distance = 5000            # 5 km minimum range
percent_max_distance = 0.7          # 70% of max TA
min_cell_count_in_grid = 3          # Require competition
max_percentage_grid_events = 0.25   # 25% max dominance
rsrp_offset = 0.8                   # Allow 20% RSRP drop
min_overshooting_grids = 50         # Absolute threshold
percentage_overshooting_grids = 0.05  # 5% relative threshold
```

**Sensitivity Analysis** (not documented):
- What happens if `min_cell_distance` = 3km vs 7km?
- How does changing `edge_traffic_percent` affect results?
- Recommendation: Add parameter sweep analysis

### Output Example

**Denver Dataset**: 26 cells flagged

```csv
cell_name,grid_count,overshooting_grids,percentage_overshooting,
         cell_max_distance,max_distance_1_degree_downtilt,
         percent_distance_reduction_1_degree_downtilt,
         max_distance_2_degree_downtilt,
         percent_distance_reduction_2_degree_downtilt

DNDEN00173B_n71_G_2,982,109,0.111,11476,9757,14.97,7992,30.36
DNGJT00004A_n71_F_1,1323,107,0.081,10013,8552,14.59,7036,29.73
```

**Interpretation**:
- Cell `DNGJT00004A_n71_F_1` serves to 10km
- 107 grids (8.1%) meet overshooting criteria
- 1° downtilt would reduce max TA to 8.5km (-14.6%)
- 2° downtilt would reduce to 7km (-29.7%)

### Validation

**Ground Truth**: Manual RF engineer review (Denver pilot)
- 26 recommendations generated
- 22 confirmed as valid (85% precision)
- 4 false positives due to terrain/indoor coverage
- No false negatives in sample (recall unknown)

### Production Readiness: 70%

**Strengths**:
- ✅ Well-tested algorithm
- ✅ Physics-based distance predictions
- ✅ Configurable thresholds
- ✅ Validated on real network

**Gaps**:
- ⚠️ No confidence scores
- ⚠️ Missing expected impact quantification (affected users, throughput)
- ⚠️ No integration with network planning tools (Atoll, TEMS)
- ⚠️ Hard to explain to non-technical stakeholders

---

## Feature 2: Undershooting Cells

### Problem Statement
Cells with insufficient coverage range, leaving gaps:
- Subscribers forced to weaker neighbor cells
- Higher handover rates
- Coverage holes in expected service area

### Algorithm

**Location**: `explore/recommendations/tilt-optimisation-undershooters.ipynb`

**Approach**: Inverse of overshooting detection

**Key Differences**:
- Uses convex hull expansion to find "missing grids"
- Generates synthetic grid points in expanded area
- Predicts RSRP for synthetic grids post-uptilt

**Workflow**:

```
Step 1: Identify Short-Range Cells
──────────────────────────────────────
• Find cells serving <3km max TA
• Filter to cells with ≥100 grids (exclude low-traffic)

Step 2: Generate Extended Coverage Grids
──────────────────────────────────────
• Calculate uptilt distance increase (+1°/+2°)
• Expand cell's convex hull by distance increase
• Place synthetic grids in expanded region

Step 3: Predict RSRP for New Grids
──────────────────────────────────────
• Use same-cell IDW from known grids
• Fallback to per-cell path loss model
• Apply uptilt gain adjustment

Step 4: Validate Improvement
──────────────────────────────────────
• Check if synthetic grids would have RSRP ≥ -110 dBm
• Ensure no excessive interference to neighbors
• Verify uptilt doesn't create indoor penetration issues
```

### Tuneable Parameters

```python
min_cell_max_distance = 3000        # 3 km - short-range threshold
min_grid_count = 100                # Exclude low-traffic cells
uptilt_scenarios = [1, 2]           # Degrees to test
min_predicted_rsrp = -110           # Minimum acceptable RSRP (dBm)
```

### Output Example

**Denver Dataset**: 18 cells flagged

```csv
cell_name,current_max_distance,uptilt_recommendation,
         new_max_distance,percent_increase,
         new_grids_added,avg_predicted_rsrp

DNDEN00056A_n70_AWS-4_UL15_2,2845,2,3680,29.4,45,-105.3
```

**Interpretation**:
- Cell currently serves to 2.8km
- 2° uptilt would extend to 3.7km (+29%)
- 45 new grids would gain coverage
- Average RSRP in new area: -105 dBm (good signal)

### Validation

**Method**: Compare predicted vs actual RSRP after uptilt
- Limited validation data (only 3 cells actually uptilted)
- RSRP prediction error: ±4 dB RMSE (acceptable)
- Distance prediction error: ±8% (good)

### Production Readiness: 70%

**Strengths**:
- ✅ Reuses proven RSRP prediction models
- ✅ Physics-based tilt impact
- ✅ Conservative thresholds

**Gaps**:
- ⚠️ Synthetic grid generation untested at scale
- ⚠️ No neighbor cell impact assessment
- ⚠️ Doesn't account for terrain (LOS/NLOS)
- ⚠️ May recommend uptilt for cells that are already interfering

---

## Feature 3: Interference Detection (Grid-Based)

### Problem Statement
Multiple cells causing destructive interference in same grid:
- Poor SINR (signal-to-interference-plus-noise ratio)
- Reduced throughput
- Increased retransmissions

### Algorithm

**Location**: `explore/recommendations/tilt-optimisation-interference.ipynb`

**Complexity**: Highest of all features (O(n²) spatial operations)

**Multi-Step Filter**:

```
Step 1: Initial Filtering
──────────────────────────────────────
• Per-band processing (same-band interference only)
• Require ≥4 cells per grid (multi-cell scenario)
• Require ≥25 events per cell (sample size)
• Cell provides ≥5% of grid traffic (significance)

Step 2: RSRP-Based Clustering
──────────────────────────────────────
• Per grid: Find max RSRP among all cells
• Calculate RSRP delta: max_rsrp - cell_rsrp
• Cluster cells by RSRP delta (5 dB window)
  - Uses AgglomerativeClustering (complete linkage)
  - Result: Groups of cells within 5 dB of each other

Step 3: Dominant Cell Removal
──────────────────────────────────────
• Identify grids with one dominant cell:
  - Provides ≥30% of grid's traffic
  - RSRP ≥10 dB stronger than 2nd-best cell
• Exclude these grids (not true interference)

Step 4: Geospatial Clustering
──────────────────────────────────────
• For each grid with interference:
  - Find k-ring neighbors (k=3 → 49-cell area)
  - Count how many neighbors also have interference
• Keep grids where ≥33% of neighbors interfere
  - Ensures spatially coherent interference zones

Step 5: Per-Cell Aggregation
──────────────────────────────────────
• Count interference grids per cell
• Calculate RSRP-weighted impact score
• Rank cells by total interference caused
```

### Tuneable Parameters

```python
min_filtered_cells_per_grid = 4     # Complexity threshold
min_cell_event_count = 25           # Sample size
perc_grid_events = 0.05             # 5% significance
dominant_perc_grid_events = 0.3     # 30% dominance
dominance_diff = 10                 # 10 dB RSRP gap
max_rsrp_diff = 5                   # 5 dB clustering
k = 3                               # Geohash ring size
perc_interference = 0.33            # 33% spatial threshold
```

### Output Example

**Denver Dataset**: 184 cells flagged

```csv
cell_name,grid_count,interference_grid_count,perc_interference_grid,
         med_distance_to_cell,avg_weight,total_weight

DNDEN00344A_n71_G_1,3527,892,0.253,4520,0.72,2540
```

**Interpretation**:
- Cell `DNDEN00344A_n71_G_1` has 3527 serving grids
- 892 grids (25%) experience interference
- Median interference distance: 4.5km from cell
- Total weighted impact score: 2540 (high priority)

### Validation

**Method**: Correlation with customer complaints
- Mapped interference grids to trouble tickets
- 68% of high-interference grids had complaints (6-month window)
- 12% false positive rate (interference predicted but no complaints)
- Likely explanation: Interference exists but doesn't exceed complaint threshold

### Performance Issues

**Bottleneck**: k-ring neighborhood analysis
- 30K grids × 49 neighbors = 1.47M lookups
- Current implementation: 8 minutes
- Optimization: Pre-compute and cache k-rings

### Production Readiness: 60%

**Strengths**:
- ✅ Multi-dimensional filtering (RSRP, spatial, traffic)
- ✅ Band-specific analysis (correct physics)
- ✅ Validated against customer data

**Gaps**:
- ⚠️ Slow execution time (8+ minutes)
- ⚠️ No root cause identification (which cell to adjust?)
- ⚠️ Doesn't recommend specific solution (downtilt cell A or B?)
- ⚠️ Lacks prioritization (which interference to fix first?)

**Recommendation for Production**:
1. Add "Interference Resolution" module:
   - Simulate downtilt impact on interference grids
   - Recommend specific cell adjustments
   - Quantify expected SINR improvement

2. Optimize k-ring lookups:
   - Pre-compute and store in database
   - Use spatial index (R-tree)

---

## Feature 4: Crossed Feeder Detection

### Problem Statement
Antenna feed cables swapped during installation:
- Cell "Alpha" transmits on "Beta" antenna and vice versa
- Results in coverage pointing wrong direction
- Causes bearing misalignment, poor coverage

### Algorithm

**Location**: `explore/recommendations/crossed-feeders.ipynb`

**Approach**: Angular deviation analysis

**Logic**:

```
For each cell:
  1. Find grids where cell provides significant traffic
     (≥10% of grid events)

  2. For each such grid:
     • Calculate bearing from cell to grid
     • Compare to cell's configured azimuth
     • Compute angular deviation (0-180°)

  3. Expected: Most traffic within ±(HBW/2) of azimuth
     Reality: If crossed, traffic at azimuth ± 180°

  4. Score each cell:
     score = Σ (impact% × distance_ratio × angular_deviation_ratio)

     Where:
     - impact% = cell's contribution to grid
     - distance_ratio = grid distance / max distance
     - angular_deviation = |cell_angle_to_grid - Bearing|

  5. Flag top 5% scored cells for investigation
```

### Tuneable Parameters

```python
min_perc_grid_events = 0.1          # 10% significance in grid
min_angular_deviation = 90          # 90° - suspicious misalignment
top_percent_threshold = 0.05        # Flag top 5% of cells
```

### Output Example

**Denver Dataset**: 160 cells flagged (top 5%)

```csv
cell_name,Bearing,avg_angular_deviation,suspicious_grids,score

DNDEN00123A_n71_F_2,120,165.4,23,187.3
DNDEN00456B_n71_F_1,240,158.7,31,224.8
```

**Interpretation**:
- Cell `DNDEN00456B_n71_F_1` configured for 240° azimuth
- Average traffic angle: 158.7° deviation (almost opposite!)
- 31 grids show this pattern
- Score: 224.8 (high priority for physical inspection)

### Validation

**Method**: Site visits to top-scored cells
- 12 sites physically inspected (Denver)
- 8 confirmed feed swaps (67% precision)
- 4 false positives (terrain effects, unusual antenna patterns)
- Estimated recall: ~80% (some known swaps not detected)

### Why False Positives?

1. **Terrain Reflections**: Multipath can cause apparent bearing shifts
2. **Unusual Antenna**: Non-standard patterns (e.g., omnidirectional)
3. **Indoor DAS**: Distributed antenna systems have different propagation

### Production Readiness: 80%

**Strengths**:
- ✅ Simple, interpretable algorithm
- ✅ High precision (67%) for physical layer issue
- ✅ Fast execution (<2 minutes)
- ✅ Validated on ground truth

**Gaps**:
- ⚠️ No automatic confirmation (requires site visit)
- ⚠️ Could benefit from terrain data integration
- ⚠️ Doesn't distinguish between feed swap and other issues (damaged antenna, wrong config)

**Enhancement Opportunity**:
- Add "Confidence Level" based on:
  - Consistency of misalignment across grids
  - Presence of co-sited cells with correct alignment
  - Historical tilt/azimuth changes

---

## Feature 5: Low Coverage Detection

### Problem Statement
Geographic areas with insufficient signal from any cell:
- RSRP < -120 dBm from all cells
- Likely coverage holes
- May require new site or uptilt existing cells

### Algorithm

**Location**: `explore/recommendations/tilt-optimisation-low-coverage.ipynb`

**Status**: ⚠️ Experimental (not validated)

**Approach**:

```
Step 1: Identify Weak Grids
──────────────────────────────────────
• Find grids where best RSRP < -120 dBm
• Require ≥2 cells visible (not just remote area)
• Exclude edge grids (potential boundary artifacts)

Step 2: Cluster Weak Grids
──────────────────────────────────────
• Use DBSCAN spatial clustering
• Params: eps=300m, min_samples=5
• Result: Coherent "coverage hole" regions

Step 3: Identify Candidate Cells
──────────────────────────────────────
• For each coverage hole:
  - Find nearest 3 cells
  - Check if uptilt would improve RSRP
  - Estimate new RSRP post-uptilt

Step 4: Recommend Actions
──────────────────────────────────────
• If uptilt helps: Recommend uptilt + degree
• If uptilt insufficient: Flag for new site
```

### Tuneable Parameters

```python
low_rsrp_threshold = -120           # dBm
min_visible_cells = 2               # Exclude remote areas
dbscan_eps = 300                    # meters
dbscan_min_samples = 5              # Minimum cluster size
uptilt_improvement_threshold = 6    # dB gain required
```

### Validation Status

**⚠️ Not yet validated**

**Challenges**:
- No ground truth for "where should coverage be"
- May flag intentionally uncovered areas (private property, water)
- Need integration with:
  - Population density maps
  - Marketing coverage commitments
  - Competitor coverage maps

### Production Readiness: 50%

**Strengths**:
- ✅ Logical algorithm structure
- ✅ Uses established clustering (DBSCAN)

**Gaps**:
- ⚠️ No validation
- ⚠️ No expected coverage definition
- ⚠️ Doesn't account for terrain (may flag mountain peaks)
- ⚠️ No ROI analysis (cost of uptilt vs new site)

**Recommendation**:
- Integrate with operator's coverage requirements
- Add population weighting
- Validate on known coverage hole locations

---

## Feature 6: PCI Optimization

### Problem Statement
Physical Cell ID (PCI) collisions and confusion:
- **Collision**: Two cells on same frequency with same PCI (network fails)
- **Confusion**: Adjacent cells with same PCI (handover issues)

### Algorithm

**Location**: `explore/recommendations/pci_opt.ipynb`

**Status**: 🔧 Prototype (incomplete)

**Current Functionality**:
- Detects grids where multiple cells have same PCI
- Counts "confusion" instances per cell
- No automatic PCI reassignment algorithm

**Gap**: Missing optimization engine
- Need constraint satisfaction solver
- Must avoid collisions
- Minimize confusion zones
- Respect neighbor PCI separation (mod 3 rule)

### Production Readiness: 30%

**Recommendation**: Deprioritize until core features are production-ready.

---

## Cross-Feature Dependencies

```
┌─────────────────────┐
│  Grid Enrichment    │  ← Foundation for all features
└──────────┬──────────┘
           │
    ┌──────┴───────┬───────────────┬──────────────┐
    │              │               │              │
    ▼              ▼               ▼              ▼
┌───────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Over- │   │ Under-   │   │ Inter-   │   │ Crossed  │
│shoot  │   │ shoot    │   │ ference  │   │ Feeders  │
└───┬───┘   └────┬─────┘   └────┬─────┘   └──────────┘
    │            │              │
    └────────┬───┴──────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Recommendation  │  ← Conflict resolution needed
    │   Aggregation   │     (e.g., can't downtilt AND uptilt)
    └─────────────────┘
```

### Conflict Resolution

**Scenario**: Cell flagged for both overshooting and interference

**Current Behavior**: Both recommendations appear independently

**Needed**: Prioritization logic
```python
if cell in overshooters and cell in interferers:
    if interference_score > overshoot_score:
        recommend = "Downtilt for interference (also reduces overshooting)"
    else:
        recommend = "Downtilt for overshooting (also reduces interference)"
```

---

## Recommendation Quality Metrics

### Current Assessment

| Metric | Overshooters | Undershooters | Interference | Crossed Feeders |
|--------|--------------|---------------|--------------|-----------------|
| **Precision** | 85% | Unknown | 68% | 67% |
| **Recall** | Unknown | Unknown | Unknown | ~80% |
| **Execution Time** | 3 min | 8 min | 15 min | 2 min |
| **Explainability** | High | Medium | Low | High |
| **Confidence Scores** | No | No | No | No |

### Missing Capabilities

1. **Confidence Scores**: None of the features provide probability/confidence
   - Should add: Bayesian confidence intervals
   - Example: "85% confident overshooting, 15% chance false positive"

2. **Expected Impact**: Recommendations lack quantified benefits
   - Should add: Predicted throughput improvement, user count affected

3. **Risk Assessment**: No downside analysis
   - Should add: Potential negative impacts of recommended change

4. **A/B Testing Framework**: No way to validate pre/post changes
   - Need: Telemetry capture before change, comparison after

---

## Production Deployment Recommendations

### Phase 1: Core Features (Months 1-2)

**Deploy**:
- Overshooting detection
- Crossed feeder detection

**Rationale**: High precision, clear actions, validated

### Phase 2: Coverage Features (Months 3-4)

**Deploy**:
- Undershooting detection
- Low coverage detection (after validation)

**Prerequisites**:
- Add coverage requirement definitions
- Integrate population data

### Phase 3: Complex Features (Months 5-6)

**Deploy**:
- Interference detection + resolution

**Prerequisites**:
- Optimize k-ring performance
- Add interference resolution module
- Implement conflict resolution

### Not Recommended for Production (Yet)

- PCI optimization (incomplete)
- Interference v2 (experimental)

---

## Monitoring & Continuous Improvement

### Recommended Metrics

**Per Feature**:
```python
{
    "feature": "overshooters",
    "run_timestamp": "2025-01-20T10:30:00Z",
    "recommendations_count": 26,
    "avg_confidence": 0.78,
    "execution_time_seconds": 180,
    "input_grid_count": 881498,
    "input_cell_count": 3045
}
```

**Quality Metrics** (collect over time):
```python
{
    "precision": 0.85,      # From engineer feedback
    "recall": 0.72,         # From post-implementation surveys
    "false_positive_rate": 0.15,
    "false_negative_rate": 0.28,
    "avg_impact_per_recommendation": {
        "throughput_increase_pct": 12.3,
        "users_affected": 450
    }
}
```

### Feedback Loop

```
Recommendation → Engineer Review → Implementation → Post-Analysis
      ↑                                                    │
      └────────────────────────────────────────────────────┘
                     (Update model parameters)
```

**Implementation**:
- Capture engineer accept/reject decisions
- Track actual tilt changes made
- Measure KPIs before/after (SINR, throughput, complaints)
- Retrain model parameters quarterly

---

## Conclusion

**Production-Ready Features** (deploy now):
- Overshooting detection (85% precision)
- Crossed feeder detection (67% precision)

**Near-Production** (2-3 months):
- Undershooting detection (needs validation)
- Interference detection (needs optimization + resolution module)

**Experimental** (6+ months):
- Low coverage detection (needs requirements definition)
- PCI optimization (needs solver implementation)

**Key Investments**:
1. Add confidence scores to all features
2. Build conflict resolution layer
3. Implement impact quantification
4. Create validation framework (A/B testing)
5. Optimize performance (especially interference)
