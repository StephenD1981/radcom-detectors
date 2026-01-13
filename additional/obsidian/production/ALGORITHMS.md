# RAN Optimizer Algorithm Specification

**Document Version:** 5.0
**Last Updated:** 2026-01-13
**Classification:** Technical Reference

---

## Document Purpose

This specification defines the detection algorithms used by the RAN Optimizer to identify and remediate network issues across multiple dimensions: coverage, interference, PCI planning, carrier aggregation, and physical layer anomalies. It provides:

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
5. [Interference Detection](#5-interference-detection)
6. [PCI Planning](#6-pci-planning)
7. [PCI Collision Detection](#7-pci-collision-detection)
8. [Carrier Aggregation Imbalance](#8-carrier-aggregation-imbalance)
9. [Crossed Feeder Detection](#9-crossed-feeder-detection)
10. [Environment-Aware Classification](#10-environment-aware-classification)
11. [Validation Framework](#11-validation-framework)
12. [Quick Reference](#12-quick-reference)

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

Grid passes filter if: Relative_Reach(c, g) ≥ min_relative_reach (default: 0.70)
```

**Rationale:** If cell *c* reaches 3 km to a grid, but another cell reaches 10 km to that same grid, cell *c* is not the overshooting culprit—the 10 km cell is.

---

#### Step 3c: Azimuth Filtering

Ensure the grid is in front of the antenna (within main beam direction).

**Condition:**
```
Angular_Deviation(c, g) = |bearing(cell_c → grid_g) - azimuth_c|

Grid passes filter if: Angular_Deviation(c, g) ≤ max_azimuth_deviation_deg (default: 90°)
```

**Rationale:** Grids more than 90° off the antenna bearing are behind or to the side. These should not be flagged as overshooting since they're not in the intended coverage direction.

---

#### Step 4: RSRP Degradation Filter

Verify that the cell's signal is genuinely degraded at the edge (consistent with propagation loss over distance).

**Condition:**
```
RSRP_c(g) ≤ Reference_RSRP(c) - rsrp_degradation_db

where:
  Reference_RSRP(c) = Quantile-based reference RSRP (default: 85th percentile, rsrp_reference_quantile = 0.85)
  rsrp_degradation_db = 10.0 dB (default)
```

**Rationale:**
- A cell with uniformly strong RSRP everywhere is not overshooting—it has excellent coverage
- Overshooting is characterized by weak-but-present signals at distant locations
- P85 quantile reference (not P100/max) avoids outlier-driven false negatives from close-in strong measurements
- Quantile-based approach is more robust to measurement anomalies than using maximum RSRP

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

| Component | Weight | Calculation | Source Parameter |
|-----------|--------|-------------|------------------|
| Overshooting grid count | 30% | Normalized count vs. 95th percentile | `severity_weight_bins` |
| Percentage overshooting | 25% | Direct percentage value | `severity_weight_percentage` |
| Maximum distance reached | 20% | Normalized: (dist - 4000) / (35000 - 4000) | `severity_weight_distance` |
| RSRP degradation magnitude | 15% | Normalized: (RSRP - (-120)) / ((-70) - (-120)) | `severity_weight_rsrp` |
| Traffic impact | 10% | Traffic volume in overshooting grids | `severity_weight_traffic` |

**Normalization Ranges (from config):**
- Distance: 4,000m (min) to 35,000m (max)
- RSRP: -120 dBm (worst) to -70 dBm (best)
- Grid counts: 95th percentile normalization to avoid outlier distortion

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
| **Edge interference** | > 20% | Other cells dominate at coverage edge (max_interference_percentage = 0.20) |
| **Edge traffic present** | > 15% | User demand exists in underserved area |

**Note:** Edge interference threshold is configurable and environment-aware (default: 20% for suburban, 25% for urban, 15% for rural).

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

### 2.5 Severity Scoring

Compute a composite severity score *S* ∈ [0, 1] using weighted factors:

| Component | Weight | Calculation | Source Parameter |
|-----------|--------|-------------|------------------|
| Coverage increase potential | 30% | Normalized coverage gain from tilt adjustment | `severity_weight_coverage` |
| New grids reachable | 25% | Number of new grids that would gain coverage | `severity_weight_new_grids` |
| Low interference at edge | 20% | Low interference indicates good uptilt candidate | `severity_weight_low_interference` |
| Distance gain potential | 15% | Predicted range extension (meters) | `severity_weight_distance` |
| Traffic at edge | 10% | Traffic volume in underserved edge areas | `severity_weight_traffic` |

**Coverage Increase Thresholds:**

| Uptilt Change | Min Coverage Increase | Min Distance Gain | Min New Grids |
|---------------|----------------------|-------------------|---------------|
| **1° uptilt** | 5% (`min_coverage_increase_1deg: 0.05`) | 50m (`min_distance_gain_1deg_m`) | 5 grids (`min_new_grids_1deg`) |
| **2° uptilt** | 10% (`min_coverage_increase_2deg: 0.10`) | 100m (`min_distance_gain_2deg_m`) | 10 grids (`min_new_grids_2deg`) |

**Physical Constraints:**
- Minimum tilt reached: Flag `MIN_TILT_REACHED` when cell already at 0° electrical tilt (cannot uptilt further)
- Maximum uptilt change: 2° (recommendation capped per validator)

### 2.6 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cell_id` | string | Unique cell identifier | `262011234567891` |
| `max_distance_m` | float | Current maximum range (meters) | `2500.0` |
| `edge_interference_pct` | float | Interference ratio at edge [0-1] | `0.55` |
| `edge_traffic_pct` | float | Traffic proportion at edge [0-1] | `0.18` |
| `severity_score` | float | Composite severity [0-1] | `0.65` |
| `recommended_uptilt` | float | Suggested uptilt adjustment (degrees) | `1.5` |
| `predicted_distance_gain_m` | float | Expected range increase (meters) | `400.0` |
| `predicted_coverage_increase` | float | Predicted coverage gain percentage | `0.08` |
| `predicted_new_grids` | integer | Number of new grids that would gain coverage | `7` |
| `constraint_flags` | list | Physical constraints (e.g., MIN_TILT_REACHED) | `[]` |

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
│    ─────────────────────────────────────────                    │
│              ANALYSIS BOUNDARY                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Detection Procedure

**Method:** Cell Hull Clustering Approach

1. **Cluster Cell Hulls:** Use DBSCAN (eps=5km) to group nearby cell coverage areas
2. **Create Cluster Hull:** For each cell cluster, compute convex hull encompassing all cells
3. **Find Gap Polygons:** Compute difference between cluster hull and union of cell coverage
4. **Extract Geohashes:** Get all geohash7 grids within gap polygons
5. **K-Ring Density Filter:** Keep only geohashes where ≥40 of 49 neighbors (k=3) are also gaps
6. **Cluster Gaps:** Use HDBSCAN (min_cluster_size=10) to group spatially dense gaps
7. **Create Polygons:** Generate alpha shape polygons for each gap cluster
8. **Boundary Clipping:** Optionally clip to provided boundary shapefile

**Key Parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `cell_cluster_eps_km` | 5.0 | Maximum distance for cell clustering |
| `k_ring_steps` | 3 | Neighborhood size (3 rings = 49 neighbors) |
| `min_missing_neighbors` | 40 | Out of 49 neighbors, how many must be gaps |
| `hdbscan_min_cluster_size` | 10 | Minimum geohashes to form a cluster |
| `alpha_shape_alpha` | None | Alpha for concave hull (None = auto) |

**Rationale for K-Ring Filtering:** Eliminates isolated anomalies (indoor measurements, temporary issues) while preserving genuine coverage deficits. Requires spatial consistency across a 3-ring neighborhood.

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

**Technology Standards:**
- **LTE:** RSRP < -115 dBm (3GPP TS 36.214)
- **NR:** SS-RSRP < -110 dBm (3GPP TS 38.215)

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

### 4.3 Band-Specific Thresholds

Different frequency bands have different service thresholds:

| Band | Frequency | RSRP Threshold | Rationale |
|------|-----------|----------------|-----------|
| **L700** | 700 MHz | -110 to -120 dBm | Coverage layer; stricter requirement |
| **L800** | 800 MHz | -110 to -120 dBm | Coverage layer; stricter requirement |
| **L1800** | 1800 MHz | -115 to -125 dBm | Capacity layer; more lenient |
| **L2100** | 2100 MHz | -115 to -125 dBm | Capacity layer; more lenient |
| **L2600** | 2600 MHz | -115 to -125 dBm | Capacity layer; more lenient |

**Environment-Aware Thresholds:**

| Environment | Min RSRP (dBm) | Min Sample Count | Rationale |
|-------------|----------------|------------------|-----------|
| **Urban** | -110 | 10 | Dense network; stricter requirements |
| **Suburban** | -115 | 5 | Balanced network; moderate requirements |
| **Rural** | -120 | 2 | Sparse network; lenient requirements |

### 4.4 K-Ring Density Filter

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

### 4.5 Per-Band Output

Separate GeoJSON files per frequency band:
- `low_coverage_band_700.geojson`
- `low_coverage_band_800.geojson`
- `low_coverage_band_1800.geojson`
- `low_coverage_band_2100.geojson`

**Rationale:** Coverage characteristics vary by band. An area with low 800 MHz coverage may have adequate 1800 MHz service.

### 4.6 Data Quality Filters

Low coverage detection applies the following data quality filters:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| **RSRP range** | -140.0 to -30.0 dBm | Remove unrealistic measurements (outliers, errors) |
| **Minimum area** | 0.5 km² (`min_area_km2`) | Filter micro-clusters (isolated anomalies) |
| **Sample count** | Environment-specific | Urban: 10, Suburban: 5, Rural: 2 minimum samples |

**Rationale:**
- RSRP outside -140 to -30 dBm indicates measurement errors or data corruption
- Micro-clusters below 0.5 km² are typically indoor measurements or temporary issues, not genuine coverage deficits
- Rural areas require lower sample counts due to sparse user density

### 4.7 Output Schema

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

## 5. Interference Detection

### 5.1 Problem Definition

**High interference** occurs when multiple cells provide similar signal strength in the same area, causing:

| Impact | Description | KPI Effect |
|--------|-------------|------------|
| **Poor SINR** | Signal-to-Interference-plus-Noise Ratio degradation | Reduced throughput, increased retransmissions |
| **Handover Failures** | UE confusion during cell reselection | Dropped calls, service interruption |
| **Resource Contention** | Multiple cells competing for same UE | Inefficient spectrum usage |

### 5.2 Algorithm Overview

**Method:** Geohash-Based Spatial Analysis

The interference detector identifies areas where multiple cells have similar RSRP values, indicating pilot pollution and potential SINR degradation.

### 5.3 Detection Procedure

**Pipeline:**

1. **Grid Filtering:** For each grid, identify cells within max_rsrp_diff (5dB) of strongest cell
2. **RSRP Clustering:** Group similar-strength cells using fixed-anchor or complete-linkage clustering
3. **Dominance Detection:** Exclude grids where one cell clearly dominates (>5dB stronger, >30% traffic)
4. **Cell Count Filter:** Keep grids with ≥4 competing cells and no cell exceeding 25% traffic share
5. **K-Ring Spatial Clustering:** Filter isolated grids; keep only areas with ≥33% of k=3 neighbors also showing interference
6. **HDBSCAN Clustering:** Group spatially dense interference areas (min_cluster_size=5)
7. **Polygon Generation:** Create alpha shape boundaries for each interference cluster
8. **SINR Validation:** Filter clusters with avg_sinr > 0 dB (optional, if SINR data available)

### 5.4 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_filtered_cells_per_grid` | 4 | Minimum competing cells to flag interference |
| `max_rsrp_diff` | 5.0 dB | Maximum RSRP difference from strongest cell |
| `dominance_diff` | 5.0 dB | RSRP gap indicating dominance (exclude) |
| `max_percentage_grid_events` | 0.25 | Max traffic share for any cell in interference area |
| `k_ring_steps` | 3 | Spatial neighborhood size (49 neighbors) |
| `perc_interference` | 0.33 | Min fraction of neighbors with interference |
| `sinr_threshold_db` | 0.0 | Filter clusters with SINR above this |

### 5.5 Band-Specific Thresholds

Different bands have different interference characteristics due to propagation properties:

| Band | max_rsrp_diff | dominance_diff | Rationale |
|------|---------------|----------------|-----------|
| **L700** | 6.0 dB | 6.0 dB | Sub-1GHz: better propagation, wider natural overlap, more lenient |
| **L800** | 6.0 dB | 6.0 dB | Sub-1GHz: better propagation, wider natural overlap, more lenient |
| **L900** | 5.5 dB | 5.5 dB | Sub-1GHz/low mid-band: intermediate threshold |
| **L1800** | 5.0 dB | 5.0 dB | Mid-band: higher path loss, tighter cells, standard threshold |
| **L2100** | 5.0 dB | 5.0 dB | Mid-band: higher path loss, tighter cells, standard threshold |
| **L2600** | 4.5 dB | 4.5 dB | High mid-band: even higher path loss |
| **L3500** | 4.0 dB | 4.0 dB | C-band (5G NR): very high path loss, very tight cells, strict |
| **L3700** | 4.0 dB | 4.0 dB | C-band (5G NR): very high path loss, very tight cells, strict |
| **N78** | 4.0 dB | 4.0 dB | 5G NR C-band: 3500 MHz equivalent |
| **N257** | 3.5 dB | 3.5 dB | mmWave (28 GHz): extremely tight cells, very strict |
| **N258** | 3.5 dB | 3.5 dB | mmWave (26 GHz): extremely tight cells, very strict |

**Propagation Physics:**
- **Sub-1GHz (700-900 MHz):** Lower path loss → cells naturally overlap more → require wider RSRP difference tolerance (6.0 dB)
- **Mid-band (1800-2600 MHz):** Moderate path loss → moderate overlap → standard tolerance (4.5-5.0 dB)
- **C-band (3500-3700 MHz):** High path loss → tight cells → strict tolerance (4.0 dB)
- **mmWave (>24 GHz):** Very high path loss → very tight cells → very strict tolerance (3.5 dB)

### 5.6 Severity Scoring

Composite severity score *S* ∈ [0, 1] based on:

| Component | Weight | Description |
|-----------|--------|-------------|
| Number of grids | 35% | More affected grids = more severe |
| Number of cells | 25% | More cells involved = more complex |
| Affected area | 20% | Larger area = greater impact |
| Average RSRP | 20% | Worse (lower) RSRP = more severe |

**Normalization:** 95th percentile used as maximum to avoid outlier distortion.

### 5.7 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cluster_id` | string | Unique cluster identifier | `L1800_3` |
| `band` | string | Frequency band | `L1800` |
| `n_grids` | integer | Number of interference grids | `45` |
| `n_cells` | integer | Number of cells involved | `6` |
| `cells` | list | Cell names in cluster | `['CK001L1', 'CK002L1', ...]` |
| `centroid_lat` | float | Cluster center latitude | `51.8765` |
| `centroid_lon` | float | Cluster center longitude | `-8.4321` |
| `area_km2` | float | Cluster area (km²) | `2.35` |
| `avg_rsrp` | float | Average RSRP (dBm) | `-95.2` |
| `avg_sinr` | float | Average SINR (dB), if available | `-2.5` |
| `severity_score` | float | Composite severity [0-1] | `0.78` |
| `severity_category` | string | Classification | `HIGH` |
| `geometry` | WKT | Alpha shape polygon | `POLYGON((...))` |

---

## 6. PCI Planning

### 6.1 Problem Definition

**PCI (Physical Cell Identity)** issues cause mobile devices to confuse cells during measurement and handover:

| Issue Type | Definition | Impact |
|------------|------------|--------|
| **PCI Confusion** | Serving cell has 2+ neighbors with same PCI on same band | Measurement ambiguity during handover |
| **PCI Collision** | Relevant cells share same PCI with overlapping coverage | Cell identity confusion during initial access |
| **Mod 3 Conflict** | PCI mod 3 match causes PSS interference (3GPP TS 36.211 §6.11) | Degraded cell search performance |
| **Mod 30 Conflict** | PCI mod 30 match causes RS interference (3GPP TS 36.211 §6.10) | RSRP measurement errors |

### 6.2 Algorithm Overview

**Method:** Handover Relation-Based Analysis

Uses directed handover relationships to identify PCI issues based on actual network topology and traffic patterns.

### 6.3 Detection Procedure

**Data Model:**
- **Directed graph** of handover relations: cell_name → to_cell_name with weight (HO volume)
- **Traffic weighting:** out_ho (S→N), in_ho (N→S), act_ho = out_ho + in_ho
- **Share calculation:** Proportion of serving cell's total HO traffic to each neighbor

**Pipeline:**

1. **Build Network Graph:**
   - Create directed edges from handover relations
   - Calculate bidirectional traffic (out_ho, in_ho, act_ho)
   - Compute traffic shares per serving cell
   - Build collision relevance pairs (1-hop neighbors + optional 2-hop)

2. **PCI Confusion Detection:**
   - For each serving cell, group neighbors by (PCI, band) tuple
   - Flag groups with ≥2 neighbors on same PCI+band
   - Calculate severity = sum of all except strongest (act_ho)
   - Rank by severity for prioritization

3. **PCI Collision Detection:**
   - For each relevant cell pair, check if PCI values conflict
   - Filter by distance (default: 30 km radius)
   - Detect exact collision (same PCI), mod 3, and mod 30 conflicts
   - Calculate severity using traffic-weighted pair relevance

4. **Blacklist Suggestions:**
   - Identify weak/dead relations in confusion groups
   - Conservative approach: auto-blacklist only if dead both directions
   - Suggest low-activity relations for manual review
   - Ensure minimum active neighbors remain after blacklisting

### 6.4 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `couple_cosectors` | False | Force co-sector cells to share same PCI |
| `min_active_neighbors_after_blacklist` | 2 | Safety threshold for blacklist suggestions |
| `max_collision_radius_m` | 30000.0 | Maximum distance for collision detection (meters) |
| `two_hop_factor` | 0.25 | Severity multiplier for 2-hop collisions |
| `include_mod3_inter_site` | False | Include inter-site mod3 conflicts (default: intra-site only) |
| `confusion_alpha` | 1.0 | Weight for confusion severity in optimization |
| `collision_beta` | 1.0 | Weight for collision severity in optimization |
| `pci_change_cost` | 5.0 | Cost penalty for changing a cell's PCI |
| `low_activity_max_act` | 5.0 | Max total HO activity for low-activity flag |
| `low_activity_max_share` | 0.001 | Max share threshold for low-activity flag |

**3GPP PCI Ranges:**
- **LTE:** 0-503 (504 PCIs, TS 36.211)
- **NR:** 0-1007 (1008 PCIs, TS 38.211)

### 6.5 Mod Conflict Severity Factors

PCI planning applies severity multipliers for mod 3 and mod 30 conflicts:

| Conflict Type | Base Severity Factor | Purpose |
|---------------|---------------------|---------|
| **MOD3** | 0.5 (`mod3_severity_factor`) | PSS interference (3GPP TS 36.211 §6.11) - moderate impact on cell search |
| **MOD30** | 0.3 (`mod30_severity_factor`) | RS interference (3GPP TS 36.211 §6.10) - lower impact on measurements |
| **EXACT** | 1.0 (no reduction) | Full collision - critical cell identity confusion |

### 6.6 Intra-Site vs Inter-Site Severity

Severity adjustments based on site relationship (applied AFTER base factors):

| Conflict Type | Inter-Site | Intra-Site Adjustment | Rationale |
|---------------|------------|----------------------|-----------|
| **EXACT** | 1.0 (baseline) | -0.10 penalty (`intra_site_penalty_exact`) | Intra-site manageable internally but still critical |
| **MOD3** | 1.0 (baseline) | +0.25 bonus (`intra_site_bonus_mod3`) | Co-located cells with PSS conflict = severe UE confusion |
| **MOD30** | 1.0 (baseline) | +0.15 bonus (`intra_site_bonus_mod30`) | Co-located cells with RS conflict = measurement errors |

**Note:** Mod 3/30 intra-site are MORE severe because UEs see both cells with strong signals simultaneously.

**Example Calculation:**
- MOD3 inter-site: base severity × 0.5 (mod3 factor) × 1.0 (inter-site) = 0.5× severity
- MOD3 intra-site: base severity × 0.5 (mod3 factor) × 1.25 (1.0 + 0.25 bonus) = 0.625× severity

### 6.6 Output Schema

**Confusion Output:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `serving` | string | Serving cell name | `CK001L1` |
| `band` | string | Frequency band | `L1800` |
| `confusion_pci` | integer | Conflicting PCI value | `42` |
| `group_size` | integer | Number of neighbors with this PCI | `3` |
| `neighbors` | string | Comma-separated neighbor list | `CK002L1,CK003L1,CK004L1` |
| `severity_score` | float | Normalized severity [0-1] | `0.85` |
| `severity_category` | string | Classification | `CRITICAL` |

**Collision Output:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cell_a` | string | First cell name | `CK001L1` |
| `cell_b` | string | Second cell name | `CK002L1` |
| `pci_a` | integer | First cell PCI | `42` |
| `pci_b` | integer | Second cell PCI | `42` or `45` (mod3) |
| `band` | string | Frequency band | `L1800` |
| `conflict_type` | string | `exact`, `mod3`, or `mod30` | `exact` |
| `hop_type` | string | `1-hop` or `2-hop` | `1-hop` |
| `site_a` | string | First cell site ID | `CK001` |
| `site_b` | string | Second cell site ID | `CK002` |
| `intra_site` | boolean | Same site flag | `False` |
| `pair_weight` | float | Traffic-weighted relevance | `8.5` |
| `severity_score` | float | Normalized severity [0-1] | `0.92` |
| `severity_category` | string | Classification | `CRITICAL` |

---

## 7. PCI Collision Detection

### 7.1 Problem Definition

**PCI collisions** occur when cells with identical (or conflicting) PCI values have overlapping coverage areas, causing:

| Issue Type | 3GPP Reference | Impact |
|------------|----------------|--------|
| **Exact Collision** | Same PCI, same band | Cell identity confusion during initial access |
| **Mod 3 Conflict** | TS 36.211 §6.11 | PSS (Primary Sync Signal) interference |
| **Mod 30 Conflict** | TS 36.211 §6.10 | RS (Reference Signal) interference |

**Note:** This module detects PCI **collisions** (coverage overlap). For PCI **confusion** (neighbor relation ambiguity), see [PCI Planning](#6-pci-planning).

### 7.2 Algorithm Overview

**Method:** Per-Band Convex Hull Overlap Analysis

Uses cell coverage hulls to detect spatial overlap between cells with conflicting PCI values.

### 7.3 Detection Procedure

**Pipeline:**

1. **Input Validation:**
   - Check required columns: cell_name, geometry, band, pci
   - Filter null/invalid geometries (only Polygon/MultiPolygon accepted)
   - Validate PCI ranges per band (LTE: 0-503, NR: 0-1007)
   - Remove duplicate cell names

2. **Per-Band Processing:**
   - Process each frequency band separately
   - Pre-project geometries to UTM for accurate area calculations (if >100 cells)
   - Group cells by PCI (exact), PCI mod 3, PCI mod 30

3. **Overlap Detection:**
   - Use STRtree spatial index for efficient pair finding
   - For each cell pair in PCI group, check intersection
   - Filter same-site pairs (optional, configurable)
   - Calculate overlap area and percentage (relative to smaller cell)

4. **Conflict Classification:**
   - **Exact:** Same PCI → CRITICAL severity
   - **Mod 3:** Same PCI mod 3 (PSS conflict) → HIGH/MEDIUM severity
   - **Mod 30:** Same PCI mod 30 (RS conflict) → LOW/MINIMAL severity

5. **Severity Scoring:**
   - Components: overlap_pct (50%), distance (30%), conflict_type (20%)
   - Closer cells + higher overlap + exact conflict = highest severity

### 7.4 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `overlap_threshold` | 0.10 | Minimum 10% overlap to flag |
| `min_overlap_area_km2` | 0.0005 | Minimum 500m² overlap |
| `filter_same_site` | True | Exclude same-site cell pairs |
| `check_mod3_conflicts` | True | Detect PSS interference |
| `check_mod30_conflicts` | False | Detect RS interference (optional) |
| `mod3_overlap_threshold` | 0.30 | Higher threshold for mod conflicts |
| `pre_project_geometries` | True | Pre-project for performance (>100 cells) |

### 7.5 Severity Calculation

**Formula:**
```
severity_score = 0.50 × overlap_score + 0.30 × distance_score + 0.20 × conflict_factor

where:
  overlap_score = min(overlap_pct / 100, 1.0)
  distance_score = max(0, 1.0 - distance_km / 10.0)  # 0km→1.0, 10km→0.0
  conflict_factor = 1.0 (exact), 0.6 (mod3), 0.3 (mod30)
```

### 7.6 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `detector` | string | Detection type | `PCI_COLLISION` |
| `conflict_type` | string | `exact`, `mod3`, or `mod30` | `exact` |
| `band` | string | Frequency band | `L1800` |
| `pci1` | integer | First cell PCI | `42` |
| `pci2` | integer | Second cell PCI | `42` or `45` (mod3) |
| `cell1_name` | string | First cell name | `CK001L1` |
| `cell2_name` | string | Second cell name | `CK002L1` |
| `overlap_percentage` | float | Overlap relative to smaller cell | `45.2` |
| `overlap_area_km2` | float | Overlap area (km²) | `1.23` |
| `distance_km` | float | Distance between centroids | `3.5` |
| `severity_score` | float | Composite severity [0-1] | `0.88` |
| `severity_category` | string | Classification | `CRITICAL` |
| `recommendation` | string | Suggested action | `Change PCI for either cell...` |

---

## 8. Carrier Aggregation Imbalance

### 8.1 Problem Definition

**CA imbalance** occurs when the coverage band (anchor) and capacity band have mismatched footprints:

| Issue | Description | Impact |
|-------|-------------|--------|
| **Insufficient CA Coverage** | Capacity band doesn't cover full coverage band footprint | Users on coverage band can't aggregate with capacity band |
| **Wasted Capacity** | Capacity band serves areas outside coverage band | Spectrum underutilized where CA not possible |

**LTE CA Background:** Coverage band (e.g., L800) provides anchor; capacity band (e.g., L1800) provides additional throughput. For CA to work, UE must be within BOTH footprints simultaneously.

### 8.2 Algorithm Overview

**Method:** Per-Cell/Per-Site Hull Intersection Analysis

Compares coverage and capacity band hull areas to detect imbalances where capacity doesn't adequately overlap with coverage.

### 8.3 Detection Procedure

**Configuration:** Network-specific CA pairs defined in config file:

```json
{
  "ca_pairs": [
    {
      "name": "L800-L1800",
      "coverage_band": "L800",
      "capacity_band": "L1800",
      "coverage_threshold": 0.70
    }
  ],
  "cell_name_pattern": "(\\w+)[A-Z]+(\\d)"
}
```

**Pipeline:**

1. **Input Validation:**
   - Check required columns: cell_name, geometry, band
   - Filter null/empty/invalid geometries
   - Validate sample counts (if available) for hull confidence
   - Check temporal alignment (if timestamps available)

2. **Cell Name Parsing:**
   - Extract site_id and sector using regex pattern
   - Create site_sector identifier (e.g., "CK001_1")
   - Validate parse success rate (must be >50% or fail)

3. **Per CA Pair Analysis:**
   - Filter to coverage and capacity bands
   - Group by site_sector
   - For each site_sector with BOTH bands:
     - Calculate intersection area (CA-capable zone)
     - Calculate coverage_ratio = intersection / coverage_area
     - Flag if coverage_ratio < threshold (default: 70%)

4. **Environment-Aware Thresholds (Optional):**
   - Urban: 85% threshold (dense network, expect tight alignment)
   - Suburban: 70% threshold (default)
   - Rural: 60% threshold (sparse network, more lenient)

5. **Severity Scoring:**
   - Severity = 1.0 - coverage_ratio (inverted: lower coverage = higher severity)
   - Lower capacity overlap = higher severity

### 8.4 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ca_pairs` | *required* | List of coverage→capacity band pairs to check |
| `cell_name_pattern` | *required* | Regex with 2 capture groups (site_id, sector) |
| `coverage_threshold` | 0.70 | Minimum coverage ratio (70%) |
| `use_environment_thresholds` | False | Enable environment-aware thresholds |
| `min_sample_count` | 100 | Minimum UE measurements for hull confidence |
| `max_data_age_days` | 30 | Maximum age of hull data |
| `max_temporal_gap_days` | 7 | Maximum gap between coverage and capacity hulls |

**Data Source:** Coverage hulls should be generated from **actual UE measurement points** (triangulated trace data), not geometric approximations. This ensures accuracy based on real-world coverage.

### 8.5 Severity Calculation

**Formula:**
```
severity_score = 1.0 - coverage_ratio

where:
  coverage_ratio = intersection_area / coverage_area
```

**Thresholds:**
- **CRITICAL** (0.80-1.00): <20% capacity coverage
- **HIGH** (0.60-0.79): 20-40% capacity coverage
- **MEDIUM** (0.40-0.59): 40-60% capacity coverage
- **LOW** (0.20-0.39): 60-80% capacity coverage

### 8.6 Output Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `detector` | string | Detection type | `CA_IMBALANCE` |
| `ca_pair` | string | CA pair name | `L800-L1800` |
| `coverage_band` | string | Coverage (anchor) band | `L800` |
| `capacity_band` | string | Capacity band | `L1800` |
| `site_sector` | string | Site and sector identifier | `CK001_1` |
| `coverage_cell_name` | string | Coverage band cell | `CK001L8_1` |
| `capacity_cell_name` | string | Capacity band cell | `CK001L18_1` |
| `coverage_area_km2` | float | Coverage band area (km²) | `8.5` |
| `capacity_area_km2` | float | Capacity band area (km²) | `4.2` |
| `intersection_area_km2` | float | Overlapping area (km²) | `3.5` |
| `coverage_ratio` | float | Intersection / coverage | `0.41` |
| `coverage_percentage` | float | Percentage form | `41.0` |
| `severity_score` | float | 1.0 - coverage_ratio | `0.59` |
| `severity_category` | string | Classification | `MEDIUM` |
| `recommendation` | string | Suggested action | `Adjust L1800 tilt/azimuth...` |

---

## 9. Crossed Feeder Detection

### 9.1 Problem Definition

**Crossed feeders** (sector/feeder swaps) occur when antenna cables are physically connected to the wrong radios:

| Pattern | Description | Confidence |
|---------|-------------|------------|
| **Reciprocal Swap** | Cell A's traffic goes to Cell B's azimuth, Cell B's traffic goes to Cell A's azimuth | HIGH |
| **Multiple Anomalies** | Multiple cells at same site+band with out-of-beam traffic, no clean swap | MEDIUM (POSSIBLE_SWAP) |
| **Single Anomaly** | One cell with out-of-beam traffic, likely azimuth issue | LOW (SINGLE_ANOMALY) |
| **Repan Candidate** | Only cell on band with out-of-beam traffic (can't be swap) | LOW (REPAN) |

**Physical Cause:** During installation or maintenance, feeder cables connecting antennas to radios are accidentally swapped between sectors.

### 9.2 Algorithm Overview

**Method:** Swap Pattern Analysis via Traffic Direction

Analyzes the direction of handover traffic relative to cell azimuth to detect anomalies indicating physical connection errors.

### 9.3 Detection Procedure

**Pipeline:**

1. **Relation-Level Geometry:**
   - Filter to same-technology relations (LTE→LTE, NR→NR)
   - Join GIS data for serving and neighbor cell locations
   - Calculate bearing from serving to neighbor cell
   - Determine if relation is in-beam using expanded beamwidth (1.5× with cap at 60°)
   - Apply band-specific distance filters (L700: 32km, L1800: 25km, etc.)
   - Keep inter-site relations only (feeders connect to external neighbors)

2. **Per-Cell Metrics:**
   - Calculate total, in-beam, and out-of-beam traffic weights
   - Count total and out-of-beam relations
   - Calculate out-of-beam ratio
   - Determine dominant traffic direction (weighted circular mean of out-of-beam angles)
   - Extract top suspicious relations

3. **Site-Level Swap Detection:**
   - Group by site + band
   - Identify swap candidates (loose thresholds: 50% OOB ratio, 3 relations)
   - For each candidate pair, check reciprocal swap pattern:
     - Cell A's traffic direction matches Cell B's azimuth (within 30°)
     - Cell B's traffic direction matches Cell A's azimuth (within 30°)
   - Flag HIGH confidence if reciprocal swap detected

4. **Classification:**
   - **HIGH_POTENTIAL_SWAP:** Part of detected reciprocal swap pair
   - **POSSIBLE_SWAP:** Multiple cells anomalous but no clean swap (≥2)
   - **SINGLE_ANOMALY:** Single cell anomaly (likely azimuth issue)
   - **REPAN:** Single cell on band (can't be feeder swap, review antenna pan)
   - **NONE:** No out-of-beam anomaly

5. **Anomaly Thresholds (Strict for MEDIUM/LOW):**
   - Must meet ALL: out-of-beam ratio ≥50%, weight ≥5.0, total relations ≥5, OOB relations ≥3

### 9.4 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_radius_m` | 30000 | Maximum relation distance |
| `band_max_radius_m` | L700: 32000m, L800: 30000m, L1800: 25000m, L2100: 20000m, L2600: 15000m | Per-band distance limits |
| `min_distance_m` | 500 | Minimum relation distance |
| `hbw_cap_deg` | 60.0 | Cap horizontal beamwidth at 60° |
| `min_out_of_beam_ratio` | 0.50 | Minimum 50% out-of-beam traffic (CRITICAL FIX) |
| `min_out_of_beam_weight` | 5.0 | Minimum total out-of-beam weight (CRITICAL FIX) |
| `min_total_relations` | 5 | Need 5+ relations for confidence |
| `min_out_of_beam_relations` | 3 | Need 3+ OOB relations to flag |
| `use_strength_col` | `cell_perc_weight` | Column for relation strength |
| `distance_weighting` | True | Apply distance-based weighting |
| `angle_weighting` | True | Apply angle-based weighting |
| `percentile` | 0.95 | Percentile for flagging (top 5%) |

**Detection Criteria for Classification:**
- **HIGH_POTENTIAL_SWAP:** Reciprocal swap pattern detected (cells point to each other's azimuths within 30°)
- **POSSIBLE_SWAP:** 2+ cells anomalous at same site+band but no clean reciprocal pattern
- **SINGLE_ANOMALY:** 1 cell anomalous, likely azimuth configuration issue
- **REPAN:** Only cell on band (can't be feeder swap, review antenna pan)

**Rationale:**
- **50% OOB ratio threshold (updated from 60%):** Balances sensitivity with specificity. Physical feeder swaps typically show 50%+ out-of-beam traffic as antenna serves wrong direction.
- **Weight threshold of 5.0 (updated from 10.0):** Lower threshold catches more genuine swaps without excessive false positives. Reciprocal swap pattern provides additional confidence.
- Reciprocal swap pattern provides high confidence even with lower per-cell metrics.

### 9.5 Severity Calculation

**Base Scores by Confidence:**
- HIGH_POTENTIAL_SWAP: 0.90 base + up to 0.10 from OOB ratio
- POSSIBLE_SWAP: 0.70 base + up to 0.10 from OOB ratio
- SINGLE_ANOMALY: 0.50 base + up to 0.10 from OOB ratio
- REPAN: 0.30 base + up to 0.10 from OOB ratio

### 9.6 Output Schema

**Cell Results:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cell_name` | string | Cell identifier | `CK001L1` |
| `site` | string | Site ID | `CK001` |
| `band` | string | Frequency band | `L1800` |
| `bearing` | float | Cell azimuth (degrees) | `45.0` |
| `total_relations` | integer | Total handover relations | `12` |
| `out_of_beam_relations` | integer | Relations outside beam | `8` |
| `out_of_beam_ratio` | float | Ratio of OOB traffic | `0.67` |
| `dominant_traffic_direction` | float | Weighted mean angle of OOB traffic | `135.0` |
| `confidence_level` | string | Detection confidence | `HIGH_POTENTIAL_SWAP` |
| `severity_score` | float | Composite severity [0-1] | `0.95` |
| `severity_category` | string | Classification | `CRITICAL` |
| `swap_partner` | string | Swap partner cell name (if HIGH) | `CK001L2` |
| `recommendation` | string | Suggested action | `Check feeder connections` |
| `flagged` | boolean | Whether action needed | `True` |

**Swap Pairs Output:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `site` | string | Site ID | `CK001` |
| `band` | string | Frequency band | `L1800` |
| `cell_a` | string | First cell in swap | `CK001L1` |
| `cell_b` | string | Second cell in swap | `CK001L2` |
| `cell_a_azimuth` | float | Cell A azimuth | `45.0` |
| `cell_b_azimuth` | float | Cell B azimuth | `135.0` |
| `cell_a_traffic_dir` | float | Cell A dominant traffic direction | `135.0` |
| `cell_b_traffic_dir` | float | Cell B dominant traffic direction | `45.0` |
| `a_to_b_angle_diff` | float | Angular difference A→B | `0.0` |
| `b_to_a_angle_diff` | float | Angular difference B→A | `0.0` |

---

## 10. Environment-Aware Classification

### 10.1 Environment Detection

Cells are automatically classified by **Inter-Site Distance (ISD)**:

| Environment | ISD Range | Cell Characteristics |
|-------------|-----------|---------------------|
| **Urban** | < 1 km | High density, extensive overlap, small footprints |
| **Suburban** | 1 - 3 km | Moderate density, partial overlap |
| **Rural** | > 3 km | Low density, minimal overlap, large footprints |

### 10.2 Parameter Scaling

Detection thresholds adapt to environment:

| Parameter | Urban | Suburban | Rural | Purpose |
|-----------|-------|----------|-------|---------|
| **Overshooting** |  |  |  |  |
| `min_cell_distance` | 1500m | 3500m | 8000m | Range floor for analysis |
| `min_overshooting_grids` | 25 | 25 | 22 | Absolute grid threshold |
| `edge_traffic_percent` | 0.12 | 0.15 | 0.18 | Edge definition |
| `min_cell_count_in_grid` | 4 | 3 | 2 | Competing cells required |
| `interference_threshold_db` | 8.0 dB | 8.0 dB | 8.5 dB | RSRP competition gap |
| `percentage_overshooting_grids` | 0.08 | 0.10 | 0.07 | Percentage threshold |
| `max_percentage_grid_events` | 0.25 | 0.28 | 0.30 | Max traffic share in grid |
| `min_relative_reach` | 0.70 | 0.68 | 0.65 | Relative reach threshold |
| **Undershooting** |  |  |  |  |
| `max_cell_distance` | 3500m | 7000m | 12000m | Maximum range threshold |
| `max_cell_grid_count` | 4 | 3 | 2 | Max competing cells |
| `max_interference_percentage` | 0.25 | 0.20 | 0.15 | Max edge interference ratio |
| `min_coverage_increase_1deg` | 0.05 | 0.05 | 0.05 | Min coverage gain for 1° uptilt (constant) |
| `min_coverage_increase_2deg` | 0.10 | 0.10 | 0.10 | Min coverage gain for 2° uptilt (constant) |
| `min_distance_gain_1deg_m` | 25m | 50m | 100m | Min distance gain for 1° uptilt |
| `min_distance_gain_2deg_m` | 50m | 100m | 200m | Min distance gain for 2° uptilt |
| `min_new_grids_1deg` | 3 | 5 | 8 | Min new grids for 1° uptilt |
| `min_new_grids_2deg` | 6 | 10 | 15 | Min new grids for 2° uptilt |
| `path_loss_exponent` | 4.0 | 3.5 | 3.0 | Propagation model |
| **Low Coverage** |  |  |  |  |
| `rsrp_threshold_dbm` | -110 | -115 | -120 | Service threshold |
| `min_sample_count` | 10 | 5 | 2 | Minimum UE measurements |
| `min_missing_neighbors` | 15 | 40 | 60 | K-ring density threshold |

### 10.3 Rationale

**Urban:** Dense networks with smaller cells require stricter thresholds. Higher sample counts expected due to more users.

**Rural:** Sparse networks with larger cells use more lenient thresholds. Lower sample counts acceptable due to fewer users.

---

## 11. Validation Framework

All recommendations pass through safety validators before output.

### 11.1 Overshooting Validator

| Rule | Limit | Exceeded Behavior |
|------|-------|-------------------|
| Maximum downtilt change | 2° | Recommendation capped |
| Maximum total electrical tilt | 15° | Warning flag raised |
| Estimated coverage reduction | > 50% | Warning: potential gap creation |

### 11.2 Undershooting Validator

| Rule | Limit | Exceeded Behavior |
|------|-------|-------------------|
| Maximum uptilt change | 2° | Recommendation capped |
| Minimum total electrical tilt | 0° | Cannot go negative |
| Estimated interference increase | > 60% | Warning: pilot pollution risk |
| Predicted coverage increase | > 200% | Warning: unrealistic estimate |

---

## 12. Quick Reference

### 12.1 Algorithm Decision Tree

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
├─► NO COVERAGE ANALYSIS (per band)
│   │
│   ├── Cluster cell hulls (DBSCAN, eps=5km)
│   ├── For each cluster: find gap polygons
│   ├── Extract geohashes in gaps
│   ├── K-ring density filter (≥40 of 49 neighbors)
│   ├── Cluster gaps (HDBSCAN, min_size=10)
│   └── Create alpha shape polygons
│
├─► LOW COVERAGE ANALYSIS (per band)
│   │
│   ├── Find grids with RSRP < threshold (band + environment specific)
│   ├── Apply k-ring density filter (≥30% neighbors)
│   ├── Cluster with HDBSCAN
│   └── Output per-band GeoJSON
│
├─► INTERFERENCE ANALYSIS (per band)
│   │
│   ├── Find grids with ≥4 cells within 5 dB RSRP
│   ├── Cluster cells by RSRP similarity
│   ├── Exclude grids with dominant cell
│   ├── K-ring spatial filter (≥33% neighbors)
│   ├── Cluster grids (HDBSCAN)
│   └── Create polygons, filter by SINR if available
│
├─► PCI PLANNING
│   │
│   ├── Build handover relation graph
│   ├── Detect PCI confusion (serving cell with 2+ neighbors on same PCI+band)
│   ├── Detect PCI collision (relevant pairs with same/mod3/mod30 PCI)
│   └── Suggest blacklists for dead/low-activity relations
│
├─► PCI COLLISION (hull overlap)
│   │
│   ├── Per band: group cells by PCI / PCI mod 3 / PCI mod 30
│   ├── For each group: find overlapping cell pairs
│   ├── Calculate overlap percentage and distance
│   └── Classify: exact (critical), mod3 (high), mod30 (low)
│
├─► CA IMBALANCE
│   │
│   ├── For each CA pair (coverage → capacity):
│   ├── Group by site_sector (parsed from cell name)
│   ├── Calculate intersection area / coverage area
│   └── FLAG if coverage_ratio < threshold (70%)
│
└─► CROSSED FEEDER
    │
    ├── Per cell: calculate traffic direction vs azimuth
    ├── Identify out-of-beam anomalies
    ├── Per site+band: check for reciprocal swap patterns
    └── Classify: HIGH (swap), POSSIBLE (multi), SINGLE (anomaly), REPAN
```

### 12.2 Default Parameters Summary

| Parameter | Default | Algorithm |
|-----------|---------|-----------|
| `edge_traffic_percent` | 0.15 | Overshooting |
| `min_cell_distance` | 4000m | Overshooting |
| `min_cell_count_in_grid` | 4 | Overshooting |
| `max_percentage_grid_events` | 0.25 | Overshooting |
| `interference_threshold_db` | 7.5 dB | Overshooting |
| `min_relative_reach` | 0.70 | Overshooting |
| `rsrp_degradation_db` | 10.0 dB | Overshooting |
| `rsrp_reference_quantile` | 0.85 | Overshooting |
| `rsrp_competition_quantile` | 0.90 | Overshooting |
| `min_overshooting_grids` | 30 | Overshooting |
| `percentage_overshooting_grids` | 0.10 | Overshooting |
| `max_azimuth_deviation_deg` | 90.0° | Overshooting |
| `max_cell_distance` | 7000m | Undershooting |
| `max_interference_percentage` | 0.20 | Undershooting (edge interference) |
| `min_coverage_increase_1deg` | 0.05 | Undershooting (5% coverage gain) |
| `min_coverage_increase_2deg` | 0.10 | Undershooting (10% coverage gain) |
| `min_distance_gain_1deg_m` | 50m | Undershooting (distance gain for 1°) |
| `min_distance_gain_2deg_m` | 100m | Undershooting (distance gain for 2°) |
| `min_new_grids_1deg` | 5 | Undershooting (new grids for 1°) |
| `min_new_grids_2deg` | 10 | Undershooting (new grids for 2°) |
| `path_loss_exponent` | 3.5 | Undershooting |
| `rsrp_threshold` | -115 dBm | Low Coverage (band/environment-specific) |
| `neighbor_density_threshold` | 0.30 (30/100) | Low Coverage |
| `min_area_km2` | 0.5 | Low Coverage (micro-cluster filtering) |
| `rsrp_min_dbm` | -140.0 | Data quality filter (Overshooting, Low Coverage) |
| `rsrp_max_dbm` | -30.0 | Data quality filter (Overshooting, Low Coverage) |
| `cell_cluster_eps_km` | 5.0 | No Coverage |
| `k_ring_steps` | 3 | Coverage Gaps |
| `min_missing_neighbors` | 40 (of 49) | No Coverage |
| `hdbscan_min_cluster_size` | 10 | Coverage Gaps |
| `min_filtered_cells_per_grid` | 4 | Interference |
| `max_rsrp_diff_db` | 5.0 dB | Interference |
| `dominance_diff_db` | 5.0 dB | Interference |
| `perc_interference` | 0.33 | Interference |
| `max_collision_radius_m` | 30000 | PCI Planning |
| `two_hop_factor` | 0.25 | PCI Planning (2-hop severity multiplier) |
| `include_mod3_inter_site` | False | PCI Planning (intra-site mod3 only by default) |
| `mod3_severity_factor` | 0.5 | PCI Planning (PSS interference factor) |
| `mod30_severity_factor` | 0.3 | PCI Planning (RS interference factor) |
| `intra_site_bonus_mod3` | 0.25 | PCI Planning (intra-site mod3 bonus) |
| `intra_site_bonus_mod30` | 0.15 | PCI Planning (intra-site mod30 bonus) |
| `intra_site_penalty_exact` | 0.10 | PCI Planning (intra-site exact penalty) |
| `overlap_threshold` | 0.10 | PCI Collision |
| `min_overlap_area_km2` | 0.0005 | PCI Collision |
| `coverage_threshold` | 0.70 | CA Imbalance |
| `min_out_of_beam_ratio` | 0.50 | Crossed Feeder (UPDATED from 0.60) |
| `min_out_of_beam_weight` | 5.0 | Crossed Feeder (UPDATED from 10.0) |
| `min_total_relations` | 5 | Crossed Feeder |
| `severity_weight_coverage` | 0.30 | Undershooting (severity weight) |
| `severity_weight_new_grids` | 0.25 | Undershooting (severity weight) |
| `severity_weight_low_interference` | 0.20 | Undershooting (severity weight) |
| `severity_weight_distance` | 0.15 | Undershooting (severity weight) |
| `severity_weight_traffic` | 0.10 | Undershooting (severity weight) |

### 12.3 Severity Score Mapping

All algorithms use standardized severity thresholds:

| Category | Score Range | Action Timeline |
|----------|-------------|-----------------|
| **CRITICAL** | 0.80 - 1.00 | Immediate (same day) |
| **HIGH** | 0.60 - 0.79 | Within 7 days |
| **MEDIUM** | 0.40 - 0.59 | Within 30 days |
| **LOW** | 0.20 - 0.39 | Planned maintenance |
| **MINIMAL** | 0.00 - 0.19 | Monitor only |

---

## 13. Document Changelog

### Version 5.0 (2026-01-13)

**CRITICAL FIXES:**
- **Crossed Feeder Detection:**
  - Fixed `min_out_of_beam_ratio`: Updated from 60% to 50% (matches code default 0.5)
  - Fixed `min_out_of_beam_weight`: Updated from 10.0 to 5.0 (matches code default)
  - Added rationale for updated thresholds
- **Undershooting Edge Interference:** Fixed threshold from 40% to 20% (matches code `max_interference_percentage: 0.20`)

**HIGH PRIORITY FIXES:**
- **Overshooting RSRP Reference:** Updated Step 4 to specify P85 quantile (0.85) instead of "max(RSRP)" with enhanced rationale
- **Complete Band-Specific Interference Thresholds:** Added all bands from code:
  - L700/L800: max_rsrp_diff=6.0, dominance_diff=6.0
  - L900: 5.5, 5.5
  - L1800/L2100: 5.0, 5.0
  - L2600: 4.5, 4.5
  - L3500/L3700/N78: 4.0, 4.0
  - N257/N258: 3.5, 3.5
  - Added propagation physics explanation
- **PCI Planner Mod Severity Factors:** Added complete documentation:
  - `mod3_severity_factor: 0.5` (PSS interference)
  - `mod30_severity_factor: 0.3` (RS interference)
  - `intra_site_bonus_mod3: 0.25`
  - `intra_site_bonus_mod30: 0.15`
  - `intra_site_penalty_exact: 0.10`
  - Added example calculation

**MEDIUM PRIORITY ADDITIONS:**
- **Undershooting Coverage Thresholds:** Added complete set:
  - `min_coverage_increase_1deg: 0.05` (5%)
  - `min_coverage_increase_2deg: 0.10` (10%)
  - `min_distance_gain_1deg_m: 50.0`
  - `min_distance_gain_2deg_m: 100.0`
  - `min_new_grids_1deg: 5`
  - `min_new_grids_2deg: 10`
- **Undershooting Severity Weights:** Documented all five components:
  - coverage: 0.30
  - new_grids: 0.25
  - low_interference: 0.20
  - distance: 0.15
  - traffic: 0.10
- **Low Coverage `min_area_km2`:** Added 0.5 km² micro-cluster filtering parameter
- **Physical Constraint Handling:** Documented MIN_TILT_REACHED flag for undershooting
- **Data Quality Filters:** Added RSRP range validation for Overshooting and Low Coverage:
  - `rsrp_min_dbm: -140.0`
  - `rsrp_max_dbm: -30.0`
- **Environment-Specific Thresholds:** Enhanced undershooting environment table with new parameters

**Documentation Improvements:**
- Enhanced rationale sections for all updated parameters
- Added cross-references to actual Python code defaults
- Improved mathematical notation consistency
- Added example calculations for complex formulas

### Version 4.0 (2026-01-13)
- Initial production release
- Comprehensive algorithm documentation for all detectors
- Environment-aware parameter scaling
- Validation framework documentation

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [[CONFIGURATION]] | Parameter tuning guide |
| [[DATA_FORMATS]] | Input/output file specifications |
| [[API_REFERENCE]] | Python API documentation |

---

*Document maintained by the RAN Optimization Team. For questions or corrections, contact the technical documentation group.*
