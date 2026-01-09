# RAN Optimizer Algorithm Specification

**Document Version:** 2.0
**Last Updated:** 2024
**Classification:** Technical Reference

---

## Document Purpose

This specification defines the detection algorithms used by the RAN Optimizer to identify and remediate cell coverage anomalies. It provides:

- Mathematical foundations for each detection method
- Step-by-step algorithmic procedures
- Threshold definitions and their rationale
- Output schema specifications
- Operational guidelines for interpretation

**Audience:** RF Engineers, Network Optimization Teams, System Integrators

---

## Table of Contents

1. [Overshooting Detection](#1-overshooting-detection)
2. [Undershooting Detection](#2-undershooting-detection)
3. [No Coverage Detection](#3-no-coverage-detection)
4. [Low Coverage Detection](#4-low-coverage-detection)
5. [Environment-Aware Classification](#5-environment-aware-classification)
6. [Validation Framework](#6-validation-framework)
7. [Quick Reference](#7-quick-reference)

---

## 1. Overshooting Detection

### 1.1 Problem Definition

**Overshooting** occurs when a cell's radio signal propagates beyond its intended service boundary, causing:

| Impact | Description | KPI Effect |
|--------|-------------|------------|
| **Pilot Pollution** | Multiple cells compete for UE attachment in the same geographic area | Increased handover failures, reduced throughput |
| **Resource Waste** | Cell serves distant UEs inefficiently | Reduced spectral efficiency, increased interference |
| **Neighbor Conflicts** | Distant cell appears in neighbor lists unnecessarily | Delayed handovers, ping-pong effects |

### 1.2 Algorithm Overview

The overshooting detection pipeline applies six sequential filters to identify cells transmitting beyond their optimal coverage boundary.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OVERSHOOTING DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Grid-level measurements (RSRP, traffic, cell associations)       │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   STEP 1     │    │   STEP 2     │    │   STEP 3     │               │
│  │  Identify    │───►│  Min Range   │───►│   RSRP       │               │
│  │  Edge Bins   │    │   Filter     │    │ Competition  │               │
│  │  (top 15%)   │    │  (≥4000m)    │    │  Analysis    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   STEP 6     │    │   STEP 5     │    │   STEP 4     │               │
│  │  Severity    │◄───│  Threshold   │◄───│  Signal      │               │
│  │  Scoring     │    │  Validation  │    │  Degradation │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                                                                │
│         ▼                                                                │
│  OUTPUT: Flagged cells + severity scores + tilt recommendations          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Detailed Algorithm Steps

#### Step 1: Edge Bin Identification

For each cell *c*, identify the geographic extent of coverage and isolate edge regions.

**Procedure:**
1. Collect all grids *G<sub>c</sub>* where cell *c* has measurement data
2. Calculate distance *d<sub>g</sub>* from cell site to each grid centroid
3. Sort grids by distance (descending)
4. Select top *p*% as edge bins, where *p* = `edge_traffic_percent` (default: 15%)

**Mathematical Definition:**
```
Edge_Bins(c) = { g ∈ G_c : d_g ≥ D_p(c) }

where D_p(c) = p-th percentile of distances in G_c
```

**Rationale:** Edge bins represent the cell's coverage frontier where overshooting manifests as signal presence in areas better served by other cells.

---

#### Step 2: Minimum Range Pre-Filter

Exclude cells with insufficient range from overshooting analysis.

**Condition:**
```
max(d_g for g ∈ G_c) ≥ min_cell_distance (default: 4000m)
```

**Rationale:** Small cells, indoor distributed antenna systems (DAS), and picocells operate at short range by design. Flagging these as "overshooting" would generate false positives.

---

#### Step 3: RSRP-Based Competition Analysis

For each edge bin, determine if the cell faces significant competition from stronger cells.

**Procedure:**

1. **Calculate Reference Signal Level**
   ```
   P90_rsrp(g) = 90th percentile of RSRP values from all cells in grid g
   ```

2. **Identify Competing Cells**
   ```
   Competing_Cells(g) = { c : RSRP_c(g) ≥ P90_rsrp(g) - interference_threshold_db }

   where interference_threshold_db = 7.5 dB (default)
   ```

3. **Apply Competition Criteria**

   Grid *g* is marked as a **competition bin** if:
   ```
   |Competing_Cells(g)| ≥ min_cell_count_in_grid  AND
   Traffic_Share(c, g) ≤ max_percentage_grid_events

   where:
     min_cell_count_in_grid = 4 (default)
     max_percentage_grid_events = 0.25 (default)
   ```

**Interpretation:** A cell is "competing" if its signal strength is within 7.5 dB of the strongest signals in that area. The 7.5 dB threshold corresponds to approximately one order of magnitude in received power ratio.

---

#### Step 3b: Relative Distance Filter

Ensure the cell under analysis is actually reaching far—not just present in an area where *other* cells are overshooting.

**Condition:**
```
Relative_Reach(c, g) = d_c(g) / max(d_x(g) for all cells x in g)

Grid passes filter if: Relative_Reach(c, g) ≥ min_relative_reach (default: 0.7)
```

**Rationale:** If cell *c* reaches 3 km to a grid, but another cell reaches 10 km to that same grid, cell *c* is not the overshooting culprit—the 10 km cell is.

---

#### Step 4: RSRP Degradation Filter

Verify that the cell's signal is genuinely degraded at the edge (consistent with propagation loss over distance).

**Condition:**
```
RSRP_c(g) ≤ max(RSRP_c) - rsrp_degradation_db

where rsrp_degradation_db = 10.0 dB (default)
```

**Rationale:** A cell with uniformly strong RSRP everywhere is not overshooting—it has excellent coverage. Overshooting is characterized by weak-but-present signals at distant locations.

---

#### Step 5: Final Threshold Validation

A cell is classified as **OVERSHOOTING** if both conditions are satisfied:

| Condition | Parameter | Default |
|-----------|-----------|---------|
| Absolute count | `overshooting_grids ≥ min_overshooting_grids` | 30 |
| Relative proportion | `percentage_overshooting ≥ percentage_overshooting_grids` | 10% |

**Logic:** `AND` (both conditions must be true)

**Rationale:** Requiring both thresholds prevents:
- Large cells from being flagged due to a few edge anomalies (relative threshold)
- Small cells from being flagged when raw counts are misleadingly low (absolute threshold)

---

#### Step 6: Severity Scoring

Compute a composite severity score *S* ∈ [0, 1] using weighted factors:

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Overshooting grid count | 30% | Normalized count vs. maximum observed |
| Percentage overshooting | 25% | Direct percentage value |
| Maximum distance reached | 20% | Normalized distance vs. expected |
| RSRP degradation magnitude | 15% | dB drop from peak to edge |
| Traffic impact | 10% | Traffic volume in affected grids |

**Severity Classification:**

| Category | Score Range | Recommended Action |
|----------|-------------|-------------------|
| **CRITICAL** | 0.80 - 1.00 | Immediate intervention required |
| **HIGH** | 0.60 - 0.79 | Schedule within 7 days |
| **MEDIUM** | 0.40 - 0.59 | Schedule within 30 days |
| **LOW** | 0.20 - 0.39 | Address during planned maintenance |
| **MINIMAL** | 0.00 - 0.19 | Monitor; no action required |

---

### 1.4 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cell_id` | string | Unique cell identifier (CILAC format) | `262011234567890` |
| `cell_name` | string | Human-readable cell name | `DUB_NORTH_L18_S1` |
| `overshooting_grids` | integer | Count of grids flagged as overshooting | `45` |
| `percentage_overshooting` | float | Proportion of cell's grids that are overshooting | `0.12` |
| `total_grids` | integer | Total grids served by this cell | `375` |
| `max_distance_m` | float | Maximum observed cell range (meters) | `8500.0` |
| `avg_edge_rsrp` | float | Mean RSRP in edge bins (dBm) | `-105.2` |
| `severity_score` | float | Composite severity [0-1] | `0.75` |
| `severity_category` | string | Classification label | `HIGH` |
| `recommended_tilt_change` | integer | Suggested electrical downtilt adjustment (degrees) | `2` |

---

### 1.5 Visual Reference

```
NORMAL OPERATION                          OVERSHOOTING CONDITION
─────────────────                         ─────────────────────

    Cell A                                    Cell A
      │                                         │
      ▼                                         ▼
  ┌───────┐                               ┌───────┐
  │       │  ◄── Intended coverage        │       │  ◄── Intended coverage
  │   A   │                               │   A   │
  │       │                               │       │
  └───────┘                               └───────┘
                                                │
      │                                         │ Signal extends
      │ Clear boundary                          │ beyond boundary
      ▼                                         ▼
  ┌───────┐                               ┌───────┬───────┐
  │       │                               │       │░░░░░░░│
  │   B   │  ◄── Cell B serves this       │   B   │░OVER░░│ ◄── Overlap zone
  │       │                               │       │░SHOOT░│    (4+ cells,
  └───────┘                               └───────┴───────┘     <25% traffic)
```

---

## 2. Undershooting Detection

### 2.1 Problem Definition

**Undershooting** occurs when a cell's coverage footprint is smaller than optimal, resulting in:

| Impact | Description | KPI Effect |
|--------|-------------|------------|
| **Coverage Gaps** | Geographic areas with degraded service | Increased drops, user complaints |
| **Capacity Underutilization** | Cell serves fewer users than infrastructure supports | Reduced return on investment |
| **Neighbor Overload** | Adjacent cells compensate, becoming congested | Unbalanced traffic distribution |

### 2.2 Algorithm Overview

Undershooting detection identifies cells whose coverage radius falls below expected values for their deployment environment.

### 2.3 Detection Criteria

A cell is flagged as **UNDERSHOOTING** when ALL conditions are met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Range deficit** | Below environment minimum | Cell not reaching expected distance |
| **Edge interference** | > 40% | Other cells dominate at coverage edge |
| **Edge traffic present** | > 15% | User demand exists in underserved area |

**Environment-Specific Range Expectations:**

| Environment | Expected Range | Undershooting Threshold |
|-------------|----------------|------------------------|
| Rural | 8 - 15 km | < 4 km |
| Suburban | 3 - 8 km | < 3 km |
| Urban | 1 - 3 km | < 2.5 km |

### 2.4 Band-Aware Analysis

Radio frequency characteristics determine natural propagation limits:

| Band | Frequency | Typical Range | Propagation Notes |
|------|-----------|---------------|-------------------|
| **Low Band** | 700-900 MHz | 5 - 15 km | Excellent building penetration; long range |
| **Mid Band** | 1800 MHz | 2 - 5 km | Balanced capacity and coverage |
| **High Band** | 2100-2600 MHz | 1 - 3 km | High capacity; limited range |

**Key Insight:** A 2100 MHz cell reaching 2 km operates normally. A 700 MHz cell reaching only 2 km is likely undershooting.

### 2.5 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cell_id` | string | Unique cell identifier | `262011234567891` |
| `max_distance_m` | float | Current maximum range (meters) | `2500.0` |
| `edge_interference_pct` | float | Interference ratio at edge [0-1] | `0.55` |
| `edge_traffic_pct` | float | Traffic proportion at edge [0-1] | `0.18` |
| `severity_score` | float | Composite severity [0-1] | `0.65` |
| `recommended_uptilt` | float | Suggested uptilt adjustment (degrees) | `1.5` |
| `predicted_distance_gain_m` | float | Expected range increase (meters) | `400.0` |

---

## 3. No Coverage Detection

### 3.1 Problem Definition

**No coverage areas** are geographic zones entirely outside the service footprint of all cells.

### 3.2 Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│    ┌─────────┐              ┌─────────┐                         │
│    │ Cell A  │              │ Cell B  │                         │
│    │coverage │              │coverage │                         │
│    └─────────┘              └─────────┘                         │
│                ████████████                                      │
│                ██ NO COV ██  ◄── Gap: No cell reaches this area │
│                ████████████                                      │
│    ┌─────────┐              ┌─────────┐                         │
│    │ Cell C  │              │ Cell D  │                         │
│    │coverage │              │coverage │                         │
│    └─────────┘              └─────────┘                         │
│                                                                  │
│    ─────────────────────────────────────                        │
│              ANALYSIS BOUNDARY                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Detection Procedure

1. **Union Coverage Polygons:** Combine convex hulls of all cell footprints
2. **Define Analysis Boundary:** Use provided shapefile or buffered coverage extent
3. **Compute Difference:** `Gap = Boundary - Union(Coverage)`
4. **Filter Artifacts:** Remove gaps below minimum area threshold
5. **Cluster:** Group adjacent gaps into discrete regions

### 3.4 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cluster_id` | integer | Unique gap identifier | `1` |
| `n_points` | integer | Geohash count in gap | `25` |
| `area_km2` | float | Gap area (km²) | `0.45` |
| `centroid_lat` | float | Center latitude (WGS84) | `51.8765` |
| `centroid_lon` | float | Center longitude (WGS84) | `-8.4321` |
| `geometry` | WKT | Polygon boundary | `POLYGON((...))` |

---

## 4. Low Coverage Detection

### 4.1 Problem Definition

**Low coverage areas** receive signal, but below the threshold for reliable service.

**Service Threshold:** RSRP < -115 dBm

### 4.2 Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   RSRP Heat Map               Detection Pipeline             │
│                                                              │
│   ░░░░░░░░░░░░░░░░░           Step 1: RSRP threshold        │
│   ░░░░░▒▒▒▒▒░░░░░░░           (< -115 dBm)                  │
│   ░░░▒▒█████▒▒░░░░░                    │                    │
│   ░░░▒███████▒░░░░░                    ▼                    │
│   ░░░░▒▒███▒▒░░░░░░           Step 2: K-ring density filter │
│   ░░░░░░▒▒▒░░░░░░░░           (≥30% neighbors also weak)    │
│   ░░░░░░░░░░░░░░░░░                    │                    │
│                                        ▼                    │
│   Legend:                     Step 3: HDBSCAN clustering    │
│   ░ = Good    (-80 to -100)           │                    │
│   ▒ = Fair    (-100 to -115)          ▼                    │
│   █ = LOW COV (< -115)        Step 4: Polygon generation    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 K-Ring Density Filter

The algorithm validates weak signal measurements using spatial consistency:

```
    G7 ─ G4 ─ G5
    │    │    │
    G6 ─ X  ─ G1      X = Grid under test
    │    │    │       G1-G8 = Neighbors (k=2 ring)
    G8 ─ G3 ─ G2
```

**Rule:** Grid X is retained only if ≥30% of neighbors G1-G8 also exhibit low coverage.

**Rationale:** Eliminates isolated anomalies (indoor measurements, temporary interference) while preserving genuine coverage deficits.

### 4.4 Per-Band Output

Separate GeoJSON files per frequency band:
- `low_coverage_band_700.geojson`
- `low_coverage_band_800.geojson`
- `low_coverage_band_1800.geojson`
- `low_coverage_band_2100.geojson`

**Rationale:** Coverage characteristics vary by band. An area with low 800 MHz coverage may have adequate 1800 MHz service.

### 4.5 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cluster_id` | integer | Unique cluster identifier | `3` |
| `band` | integer | Frequency band (MHz) | `800` |
| `n_points` | integer | Geohash count in cluster | `15` |
| `area_km2` | float | Cluster area (km²) | `0.28` |
| `avg_rsrp` | float | Mean RSRP in cluster (dBm) | `-117.5` |
| `min_rsrp` | float | Minimum RSRP in cluster (dBm) | `-122.3` |
| `geometry` | WKT | Polygon boundary | `POLYGON((...))` |

---

## 5. Environment-Aware Classification

### 5.1 Environment Detection

Cells are automatically classified by **Inter-Site Distance (ISD)**:

| Environment | ISD Range | Cell Characteristics |
|-------------|-----------|---------------------|
| **Urban** | < 1 km | High density, extensive overlap, small footprints |
| **Suburban** | 1 - 3 km | Moderate density, partial overlap |
| **Rural** | > 3 km | Low density, minimal overlap, large footprints |

### 5.2 Parameter Scaling

Detection thresholds adapt to environment:

| Parameter | Urban | Suburban | Rural | Purpose |
|-----------|-------|----------|-------|---------|
| `min_cell_distance` | 3000m | 5000m | 8000m | Range floor for overshooting analysis |
| `min_overshooting_grids` | 30 | 50 | 80 | Absolute grid threshold |
| `edge_traffic_percent` | 0.15 | 0.10 | 0.08 | Edge definition (rural cells have proportionally larger edges) |

---

## 6. Validation Framework

All recommendations pass through safety validators before output.

### 6.1 Overshooting Validator

| Rule | Limit | Exceeded Behavior |
|------|-------|-------------------|
| Maximum downtilt change | 2° | Recommendation capped |
| Maximum total electrical tilt | 15° | Warning flag raised |
| Estimated coverage reduction | > 50% | Warning: potential gap creation |

### 6.2 Undershooting Validator

| Rule | Limit | Exceeded Behavior |
|------|-------|-------------------|
| Maximum uptilt change | 2° | Recommendation capped |
| Minimum total electrical tilt | 0° | Cannot go negative |
| Estimated interference increase | > 60% | Warning: pilot pollution risk |
| Predicted coverage increase | > 200% | Warning: unrealistic estimate |

---

## 7. Quick Reference

### 7.1 Decision Tree

```
FOR EACH CELL:
│
├─► OVERSHOOTING ANALYSIS
│   │
│   ├── Range ≥ 4000m?
│   │   └── NO → Skip (small cell)
│   │
│   ├── Identify edge bins (top 15% by distance)
│   │
│   ├── For each edge bin:
│   │   ├── Count cells within 7.5 dB of P90 RSRP
│   │   ├── Check: ≥4 competing cells?
│   │   ├── Check: ≤25% traffic share?
│   │   ├── Check: Relative reach ≥70%?
│   │   └── Check: RSRP degraded ≥10 dB?
│   │
│   ├── Count passing grids
│   │
│   └── FLAG if: ≥30 grids AND ≥10% of total
│       └── Recommend: Downtilt 1-2°
│
├─► UNDERSHOOTING ANALYSIS
│   │
│   ├── Range below environment threshold?
│   ├── Edge interference > 40%?
│   ├── Edge traffic > 15%?
│   │
│   └── FLAG if: ALL conditions met
│       └── Recommend: Uptilt (max 2°)
│
├─► NO COVERAGE ANALYSIS
│   │
│   ├── Union all coverage polygons
│   ├── Subtract from analysis boundary
│   └── Cluster and output gaps
│
└─► LOW COVERAGE ANALYSIS
    │
    ├── Find grids with RSRP < -115 dBm
    ├── Apply k-ring density filter (≥30% neighbors)
    ├── Cluster with HDBSCAN
    └── Output per-band GeoJSON
```

### 7.2 Default Parameters Summary

| Parameter | Default | Algorithm |
|-----------|---------|-----------|
| `edge_traffic_percent` | 0.15 | Overshooting |
| `min_cell_distance` | 4000m | Overshooting |
| `min_cell_count_in_grid` | 4 | Overshooting |
| `max_percentage_grid_events` | 0.25 | Overshooting |
| `interference_threshold_db` | 7.5 dB | Overshooting |
| `min_relative_reach` | 0.7 | Overshooting |
| `rsrp_degradation_db` | 10.0 dB | Overshooting |
| `min_overshooting_grids` | 30 | Overshooting |
| `percentage_overshooting_grids` | 0.10 | Overshooting |
| `rsrp_threshold` | -115 dBm | Low Coverage |
| `neighbor_density_threshold` | 0.30 | Low Coverage |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [[CONFIGURATION]] | Parameter tuning guide |
| [[DATA_FORMATS]] | Input/output file specifications |
| [[API_REFERENCE]] | Python API documentation |

---

*Document maintained by the RAN Optimization Team. For questions or corrections, contact the technical documentation group.*
