# RAN Optimizer Configuration Guide

**Document Version:** 3.0
**Last Updated:** 2026-01-13
**Classification:** Operations Manual

---

## Document Purpose

This guide provides comprehensive instructions for configuring the RAN Optimizer system. It covers:

- Configuration file structure and syntax
- Parameter tuning for detection sensitivity
- Environment-specific deployment profiles
- Performance optimization settings

**Audience:** Network Operations Engineers, RF Planning Teams, System Administrators

---

## Table of Contents

1. [Configuration Architecture](#1-configuration-architecture)
2. [Operator Configuration (YAML)](#2-operator-configuration-yaml)
3. [Overshooting Parameters](#3-overshooting-parameters)
4. [Undershooting Parameters](#4-undershooting-parameters)
5. [Coverage Gap Parameters](#5-coverage-gap-parameters)
6. [Environment-Specific Profiles](#6-environment-specific-profiles)
7. [Performance Tuning](#7-performance-tuning)
8. [Deployment Profiles](#8-deployment-profiles)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Configuration Architecture

### 1.1 Configuration Hierarchy

The RAN Optimizer uses a layered configuration system:

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYERS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Priority 1 (Highest)    │   CLI Arguments                 │
│   ─────────────────────   │   --min-overshooting-grids 50   │
│                           │                                  │
│   Priority 2              │   Environment Variables          │
│   ─────────────────────   │   $RAN_MIN_OVERSHOOTING_GRIDS   │
│                           │                                  │
│   Priority 3              │   Operator YAML                  │
│   ─────────────────────   │   operators/vf_ireland.yaml     │
│                           │                                  │
│   Priority 4 (Lowest)     │   System Defaults               │
│   ─────────────────────   │   defaults/default.yaml         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 File Structure

```
config/
├── defaults/
│   └── default.yaml           # Base configuration (do not modify)
├── operators/
│   ├── template.yaml          # Copy this for new deployments
│   ├── vf_ireland_cork.yaml   # Example: Vodafone Ireland, Cork region
│   └── three_uk_london.yaml   # Example: Three UK, London region
└── parameters/
    └── overshooting.json      # Algorithm-specific parameters
```

### 1.3 Configuration File Types

| File Type | Format | Purpose | Modification Frequency |
|-----------|--------|---------|----------------------|
| **Operator Config** | YAML | Deployment-specific settings | Once per deployment |
| **Algorithm Params** | JSON | Detection threshold tuning | As needed for optimization |
| **Environment Vars** | Shell | Runtime overrides | Per execution |

---

## 2. Operator Configuration (YAML)

### 2.1 Complete Reference Template

```yaml
# =============================================================================
# RAN OPTIMIZER CONFIGURATION
# =============================================================================
# Operator: [Your Organization]
# Region:   [Geographic Area]
# Created:  [Date]
# =============================================================================

# -----------------------------------------------------------------------------
# IDENTIFICATION
# -----------------------------------------------------------------------------
# Metadata for reports and audit trails
operator: "Vodafone_Ireland"
region: "Cork"
deployment_id: "VF_IE_CK_2024"

# -----------------------------------------------------------------------------
# DATA SOURCES
# -----------------------------------------------------------------------------
# Path variables: ${DATA_ROOT} is expanded from environment
data:
  # Primary input files (required)
  grid_data: "${DATA_ROOT}/input/grid_cell_data.csv"
  gis_data: "${DATA_ROOT}/input/cell_site_info.csv"
  hull_data: "${DATA_ROOT}/input/coverage_hulls.csv"

  # Optional boundary definition
  boundary_shapefile: "${DATA_ROOT}/input/service_area.shp"

  # Output directory
  output_base: "${DATA_ROOT}/output"

# -----------------------------------------------------------------------------
# FEATURE TOGGLES
# -----------------------------------------------------------------------------
# Enable/disable individual detection algorithms
features:
  overshooters:
    enabled: true
    export_diagnostics: false    # Generate detailed debug output

  undershooters:
    enabled: true
    export_diagnostics: false

  no_coverage:
    enabled: true
    generate_geojson: true       # Create map-ready output

  low_coverage:
    enabled: true
    per_band_analysis: true      # Separate analysis per frequency band

# -----------------------------------------------------------------------------
# PROCESSING OPTIONS
# -----------------------------------------------------------------------------
processing:
  chunk_size: 100000             # Rows per processing batch
  n_workers: 4                   # Parallel threads (match CPU cores)
  timeout_minutes: 60            # Maximum execution time
  cache_intermediate: true       # Cache intermediate results

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging:
  level: "INFO"                  # DEBUG | INFO | WARNING | ERROR
  file: "${DATA_ROOT}/logs/ran_optimizer.log"
  rotate_mb: 100                 # Log rotation threshold
```

### 2.2 Creating a New Deployment

**Step 1:** Copy the template
```bash
cp config/operators/template.yaml config/operators/my_operator.yaml
```

**Step 2:** Configure required fields
```bash
# Edit with your preferred editor
vim config/operators/my_operator.yaml
```

**Step 3:** Validate configuration
```bash
ran-optimize --validate-config config/operators/my_operator.yaml
```

---

## 3. Overshooting Parameters

### 3.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `edge_traffic_percent` | float | 0.15 | 0.05 - 0.30 | Fraction of cell footprint classified as "edge" |
| `min_cell_distance` | int | 4000 | 1000 - 15000 | Minimum cell range (m) for analysis eligibility |
| `min_cell_count_in_grid` | int | 4 | 2 - 10 | Minimum competing cells to flag a grid |
| `max_percentage_grid_events` | float | 0.25 | 0.10 - 0.50 | Maximum traffic share for overshooting grids |
| `interference_threshold_db` | float | 7.5 | 3.0 - 15.0 | RSRP gap from P90 to count as competing (dB) |
| `min_relative_reach` | float | 0.70 | 0.5 - 0.9 | Minimum reach relative to furthest cell |
| `rsrp_degradation_db` | float | 10.0 | 5.0 - 20.0 | Minimum RSRP drop from cell P85 RSRP (dB) |
| `rsrp_reference_quantile` | float | 0.85 | 0.75 - 0.95 | Quantile for cell reference RSRP (P85) |
| `rsrp_competition_quantile` | float | 0.90 | 0.80 - 0.95 | Quantile for grid competition RSRP (P90) |
| `max_azimuth_deviation_deg` | float | 90.0 | 45.0 - 180.0 | Max angular deviation from cell bearing (degrees) |
| `min_overshooting_grids` | int | 30 | 10 - 200 | Minimum grid count to flag cell |
| `percentage_overshooting_grids` | float | 0.10 | 0.05 - 0.25 | Minimum percentage of cell grids overshooting |

### 3.2 Detailed Parameter Specifications

#### `edge_traffic_percent`

**Definition:** The fraction of a cell's total grids, sorted by distance, that constitute the "edge zone."

**Default:** 0.15 (15%)

**Example Calculation:**
```
Cell A serves 400 grids
edge_traffic_percent = 0.15
Edge grids = 400 × 0.15 = 60 grids (the 60 furthest from site)
```

**Tuning Guidelines:**

| Scenario | Recommended Value | Rationale |
|----------|-------------------|-----------|
| Dense urban | 0.15 - 0.20 | Small cells have proportionally larger edges |
| Suburban | 0.12 - 0.15 | Standard configuration |
| Rural | 0.08 - 0.12 | Large cells; avoid flagging normal propagation |

---

#### `min_cell_distance`

**Definition:** Cells with maximum observed range below this threshold are excluded from overshooting analysis.

**Default:** 4000 meters

**Rationale:** Small cells, indoor DAS, and picocells operate at short range by design. Including them would generate false positives.

**Environment-Adjusted Values:**

| Environment | Recommended Value |
|-------------|-------------------|
| Urban | 3000m |
| Suburban | 5000m |
| Rural | 8000m |

---

#### `interference_threshold_db`

**Definition:** Maximum RSRP difference (in dB) from the 90th percentile signal level for a cell to be counted as "competing."

**Default:** 7.5 dB

**Technical Basis:**
- 7.5 dB corresponds to approximately 5.6× power ratio
- Cells within this range can cause meaningful interference
- P90 reference (vs. P100) excludes outlier measurements

**Example:**
```
Grid X measurements:
  Cell A: -85 dBm
  Cell B: -88 dBm
  Cell C: -91 dBm
  Cell D: -95 dBm
  Cell E: -102 dBm

P90_rsrp = -87 dBm (90th percentile)
Threshold = -87 - 7.5 = -94.5 dBm

Competing cells: A, B, C, D (RSRP ≥ -94.5 dBm)
Non-competing: E (RSRP < -94.5 dBm)
```

---

#### `min_relative_reach`

**Definition:** For a cell to be flagged as overshooting into a grid, it must reach at least this fraction of the maximum distance any cell reaches to that grid.

**Default:** 0.7 (70%)

**Purpose:** Prevents false positives where a cell appears in a distant grid, but another cell reaches even further. The truly overshooting cell is the one reaching the farthest.

**Example:**
```
Grid X is 6 km from Cell A
Grid X is 10 km from Cell B (Cell B reaches furthest)

Cell A relative reach = 6 / 10 = 0.6 < 0.7 → Cell A NOT flagged
Cell B relative reach = 10 / 10 = 1.0 ≥ 0.7 → Cell B potentially flagged
```

---

#### `rsrp_degradation_db`

**Definition:** The cell's RSRP in an edge grid must be at least this many dB weaker than its maximum RSRP anywhere.

**Default:** 10.0 dB

**Purpose:** Ensures only genuinely weakened signals at the edge are flagged. A cell with uniformly strong RSRP everywhere has excellent coverage, not overshooting.

**Example:**
```
Cell A max RSRP: -70 dBm (close to site)
Cell A edge RSRP: -85 dBm
Degradation: 15 dB > 10 dB threshold → Passes filter

Cell B max RSRP: -75 dBm
Cell B edge RSRP: -80 dBm
Degradation: 5 dB < 10 dB threshold → Filtered out
```

---

#### `min_overshooting_grids` and `percentage_overshooting_grids`

**Relationship:** Both conditions must be satisfied (AND logic)

| Parameter | Purpose |
|-----------|---------|
| `min_overshooting_grids` | Absolute threshold; prevents flagging cells with isolated edge issues |
| `percentage_overshooting_grids` | Relative threshold; ensures problem is proportionally significant |

**Example:**
```
Cell A: 500 total grids, 45 overshooting
  min_overshooting_grids: 45 ≥ 30 ✓
  percentage: 45/500 = 9% < 10% ✗
  Result: NOT flagged (percentage threshold not met)

Cell B: 300 total grids, 35 overshooting
  min_overshooting_grids: 35 ≥ 30 ✓
  percentage: 35/300 = 11.7% ≥ 10% ✓
  Result: FLAGGED (both thresholds met)
```

---

## 4. Undershooting Parameters

### 4.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_cell_event_count` | int | 200 | 100 - 500 | Minimum traffic samples required for analysis |
| `max_cell_distance` | int | 7000 | 2000 - 15000 | Maximum distance threshold for undershooting (m) |
| `interference_threshold_db` | float | 7.5 | 3.0 - 15.0 | RSRP difference for competing cells (dB) |
| `max_cell_grid_count` | int | 3 | 2 - 10 | Max competing cells in grid for low interference |
| `max_interference_percentage` | float | 0.20 | 0.10 - 0.40 | Maximum allowed fraction of grids with high interference |
| `min_coverage_increase_1deg` | float | 0.04 | 0.02 - 0.10 | Minimum coverage increase for 1° uptilt |
| `min_coverage_increase_2deg` | float | 0.08 | 0.04 - 0.20 | Minimum coverage increase for 2° uptilt |
| `min_distance_gain_1deg_m` | int | 50 | 25 - 200 | Minimum distance gain for 1° uptilt (m) |
| `min_new_grids_1deg` | int | 5 | 2 - 20 | Minimum new grids for 1° recommendation |
| `hpbw_v_deg` | float | 6.5 | 3.0 - 15.0 | Vertical half-power beamwidth (degrees) |
| `path_loss_exponent` | float | 3.5 | 2.5 - 4.5 | Path loss exponent (varies by environment) |

### 4.2 Band-Specific Defaults

Propagation characteristics vary significantly by frequency:

| Parameter | Low Band (700-900 MHz) | Mid Band (1800 MHz) | High Band (2100+ MHz) |
|-----------|------------------------|---------------------|----------------------|
| `max_cell_distance` | 10000m | 7000m | 5000m |
| `path_loss_exponent` | 3.0 (rural) - 3.5 (suburban) | 3.5 | 4.0 (urban) |
| `min_distance_gain_1deg_m` | 100m | 50m | 25m |

**Note:** The `path_loss_exponent` varies by both frequency and environment:
- **Urban (4.0):** High clutter, buildings, dense obstacles
- **Suburban (3.5):** Moderate clutter, mixed terrain
- **Rural (3.0):** Open terrain, minimal obstacles

---

## 5. Coverage Gap Parameters

### 5.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **No Coverage Detection** | | | | |
| `cell_cluster_eps_km` | float | 5.0 | 2.0 - 15.0 | DBSCAN epsilon for clustering cell hulls (km) |
| `cell_cluster_min_samples` | int | 3 | 2 - 5 | Minimum cells per cluster |
| `k_ring_steps` | int | 3 | 2 - 4 | Neighbor rings (3 = 49 neighbors) |
| `min_missing_neighbors` | int | 40 | 20 - 60 | Min missing neighbors (out of 49) |
| `hdbscan_min_cluster_size` | int | 10 | 5 - 25 | Min geohashes to form cluster |
| `alpha_shape_alpha` | float | null | null or 0.1-10.0 | Alpha for concave hull (null = auto) |
| **Low Coverage Detection** | | | | |
| `rsrp_threshold_dbm` | int | -115 | -125 to -105 | Signal level defining "low coverage" (dBm) |
| `k_ring_steps` | int | 3 | 2 - 4 | Neighbor rings for density check |
| `min_missing_neighbors` | int | 30 | 15 - 45 | Min neighbors below threshold (out of 49) |
| `min_area_km2` | float | 0.5 | 0.1 - 2.0 | Minimum cluster area (km²) |
| `max_area_per_point_km2` | float | 2.0 | 0.5 - 5.0 | Max area/point ratio (filters sparse clusters) |

### 5.2 RSRP Reference Scale

| RSRP (dBm) | Signal Quality | User Experience |
|------------|----------------|-----------------|
| > -80 | Excellent | Full speed, seamless service |
| -80 to -90 | Good | Normal operation |
| -90 to -100 | Fair | Possible speed reduction |
| -100 to -110 | Poor | Degraded service likely |
| -110 to -115 | Very Poor | Marginal connectivity |
| < -115 | **Low Coverage** | Service outages expected |

---

## 6. Environment-Specific Profiles

### 6.1 Automatic Environment Detection

Cells are classified by Inter-Site Distance (ISD):

| Environment | ISD Range | Typical Characteristics |
|-------------|-----------|------------------------|
| **Urban** | < 1 km | High-rise, dense deployment |
| **Suburban** | 1 - 3 km | Residential, mixed density |
| **Rural** | > 3 km | Open terrain, sparse deployment |

### 6.2 Profile Configuration

```json
{
  "default": {
    "min_cell_distance": 4000,
    "min_overshooting_grids": 30,
    "edge_traffic_percent": 0.15,
    "min_cell_count_in_grid": 4,
    "max_percentage_grid_events": 0.25,
    "interference_threshold_db": 7.5,
    "min_relative_reach": 0.70
  },
  "environment_profiles": {
    "urban": {
      "min_cell_distance": 1500,
      "min_overshooting_grids": 25,
      "edge_traffic_percent": 0.12,
      "min_cell_count_in_grid": 4,
      "max_percentage_grid_events": 0.25,
      "interference_threshold_db": 8.0,
      "percentage_overshooting_grids": 0.08,
      "min_relative_reach": 0.70,
      "rsrp_competition_quantile": 0.85
    },
    "suburban": {
      "min_cell_distance": 3500,
      "min_overshooting_grids": 25,
      "edge_traffic_percent": 0.15,
      "min_cell_count_in_grid": 3,
      "max_percentage_grid_events": 0.28,
      "interference_threshold_db": 8.0,
      "percentage_overshooting_grids": 0.10,
      "min_relative_reach": 0.68
    },
    "rural": {
      "min_cell_distance": 8000,
      "min_overshooting_grids": 22,
      "edge_traffic_percent": 0.18,
      "min_cell_count_in_grid": 2,
      "max_percentage_grid_events": 0.30,
      "interference_threshold_db": 8.5,
      "percentage_overshooting_grids": 0.07,
      "min_relative_reach": 0.65,
      "rsrp_competition_quantile": 0.80
    }
  }
}
```

---

## 7. Performance Tuning

### 7.1 Processing Parameters

| Parameter | Default | Impact | Tuning Guidance |
|-----------|---------|--------|-----------------|
| `chunk_size` | 100000 | Memory usage | Reduce for memory-constrained systems |
| `n_workers` | 4 | CPU utilization | Set to physical core count |
| `timeout_minutes` | 60 | Execution limit | Increase for large datasets |
| `cache_intermediate` | true | Disk usage | Disable for debugging |

### 7.2 Memory Optimization

**Symptom:** Out-of-memory errors during processing

**Solutions:**
1. Reduce `chunk_size` to 50,000 or 25,000
2. Limit algorithms per run: `--algorithms overshooting`
3. Process regions separately

**Memory Estimation:**
```
Approximate memory = (grid_rows × 500 bytes) + (cell_count × 10 KB)
```

### 7.3 Recommended Hardware Profiles

| Deployment Size | Grid Rows | Recommended RAM | Recommended Cores |
|-----------------|-----------|-----------------|-------------------|
| Small | < 1M | 8 GB | 4 |
| Medium | 1M - 10M | 16 GB | 8 |
| Large | 10M - 100M | 32 GB | 16 |
| Enterprise | > 100M | 64+ GB | 32+ |

---

## 8. Deployment Profiles

### 8.1 Dense Urban Deployment

```yaml
# Profile: Dense Urban (city center)
features:
  overshooters:
    enabled: true
    edge_traffic_percent: 0.18
    min_cell_distance: 3000
    min_overshooting_grids: 25
    min_cell_count_in_grid: 5
    interference_threshold_db: 6.0    # Tighter threshold
```

### 8.2 Rural Deployment

```yaml
# Profile: Rural (countryside)
features:
  overshooters:
    enabled: true
    edge_traffic_percent: 0.08
    min_cell_distance: 8000
    min_overshooting_grids: 100
    min_cell_count_in_grid: 3
    interference_threshold_db: 10.0   # Relaxed threshold
```

### 8.3 High-Sensitivity Mode

```yaml
# Profile: Maximum Detection (find all issues)
features:
  overshooters:
    enabled: true
    min_overshooting_grids: 15        # Lower absolute threshold
    percentage_overshooting_grids: 0.05  # Lower percentage

  low_coverage:
    enabled: true
    rsrp_threshold: -110              # Stricter RSRP threshold
    min_density_ratio: 0.20           # Easier clustering
```

---

## 6. Interference Detection Parameters

### 6.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_filtered_cells_per_grid` | int | 4 | 2 - 10 | Min cells in grid cluster for interference |
| `min_cell_event_count` | int | 2 | 1 - 10 | Min traffic events per cell for quality |
| `perc_grid_events` | float | 0.05 | 0.01 - 0.20 | Min percentage of grid events for cell |
| `max_rsrp_diff_db` | float | 5.0 | 3.0 - 10.0 | Max RSRP diff from strongest cell (dB) |
| `dominance_diff_db` | float | 5.0 | 3.0 - 10.0 | RSRP gap indicating dominance (dB) |
| `dominant_perc_grid_events` | float | 0.30 | 0.20 - 0.50 | Percentage threshold for dominant cell |
| `sinr_threshold_db` | float | 20.0 | 0.0 - 30.0 | Max SINR for interference flag (dB) |
| `k_ring_steps` | int | 3 | 2 - 4 | Geohash neighbor steps (3 = 49 neighbors) |
| `perc_interference` | float | 0.33 | 0.20 - 0.50 | Min fraction of neighbors with interference |

**Environment-Specific:**
- **Urban:** `min_filtered_cells_per_grid: 5`, `perc_interference: 0.40`
- **Rural:** `min_filtered_cells_per_grid: 3`, `perc_interference: 0.25`

---

## 7. PCI Planning Parameters

### 7.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `couple_cosectors` | bool | false | - | Force co-sector cells to share PCI |
| `min_active_neighbors_after_blacklist` | int | 2 | 1 - 5 | Min neighbors after blacklisting |
| `max_collision_radius_m` | float | 30000.0 | 10000 - 50000 | Max distance for collision (m) |
| `two_hop_factor` | float | 0.25 | 0.0 - 1.0 | Severity multiplier for 2-hop |
| `include_mod3_inter_site` | bool | false | - | Include inter-site mod3 conflicts |
| `confusion_alpha` | float | 1.0 | 0.0 - 5.0 | Weight for confusion severity |
| `collision_beta` | float | 1.0 | 0.0 - 5.0 | Weight for collision severity |
| `pci_change_cost` | float | 5.0 | 0.0 - 10.0 | Cost penalty for PCI change |
| `low_activity_max_act` | float | 5.0 | 1.0 - 20.0 | Max HO activity for low-activity flag |

---

## 8. PCI Collision Detection Parameters

### 8.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `overlap_threshold` | float | 0.10 | 0.05 - 0.30 | Min overlap % to flag (10%) |
| `min_overlap_area_km2` | float | 0.0005 | 0.0001 - 0.01 | Min overlap area (500m²) |

**Severity Thresholds:**
- **Critical:** overlap_pct ≥ 50%, distance_km ≤ 2
- **High:** overlap_pct ≥ 30%, distance_km ≤ 3
- **Medium:** overlap_pct ≥ 20%, distance_km ≤ 5

---

## 9. CA Imbalance Parameters

### 9.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `coverage_threshold` | float | 0.70 | 0.50 - 0.90 | Min capacity/coverage ratio (70%) |
| `use_environment_thresholds` | bool | false | - | Enable environment-aware thresholds |
| `cell_name_pattern` | string | `(CK\\d+)[A-Z]+(\\d)` | - | Regex to extract site_id and sector |

**Environment-Aware Thresholds (if enabled):**
- **Urban:** 0.85 (85% - stricter alignment)
- **Suburban:** 0.70 (70% - default)
- **Rural:** 0.60 (60% - more lenient)

**CA Pairs Configuration Example:**
```json
{
  "ca_pairs": [
    {
      "name": "L800-L1800",
      "coverage_band": "L800",
      "capacity_band": "L1800",
      "coverage_threshold": 0.70
    }
  ]
}
```

---

## 10. Crossed Feeder Detection Parameters

### 10.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_out_of_beam_ratio` | float | 0.60 | 0.40 - 0.80 | Min % out-of-beam traffic |
| `min_out_of_beam_weight` | float | 10.0 | 5.0 - 30.0 | Min total out-of-beam weight (%) |
| `min_total_relations` | int | 5 | 3 - 10 | Min neighbor relations required |
| `min_out_of_beam_relations` | int | 3 | 2 - 5 | Min OOB relations to flag |
| `max_radius_m` | float | 30000.0 | 15000 - 50000 | Max relation distance (m) |
| `min_distance_m` | float | 500.0 | 100 - 1000 | Min relation distance (m) |
| `hbw_cap_deg` | float | 60.0 | 30.0 - 120.0 | Cap on half-beamwidth (degrees) |
| `percentile` | float | 0.95 | 0.90 - 0.99 | Percentile for flagging (top 5%) |
| `use_strength_col` | string | `cell_perc_weight` | - | Column for relation strength |
| `distance_weighting` | bool | true | - | Apply distance-based weighting |
| `angle_weighting` | bool | true | - | Apply angle-based weighting |

**Band-Specific Maximum Radius:**
```json
{
  "L700": 32000,
  "L800": 30000,
  "L1800": 25000,
  "L2100": 20000,
  "L2600": 15000
}
```

---

## 11. Troubleshooting

### 11.1 Common Issues

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Too many cells flagged | Thresholds too low | Increase `min_overshooting_grids`, `percentage_overshooting_grids` |
| Too few cells flagged | Thresholds too high | Decrease thresholds; verify data quality |
| Missing expected cells | Filtered by pre-checks | Check `min_cell_distance`, verify cell range in data |
| Memory exhaustion | Large dataset | Reduce `chunk_size`; process in batches |
| Slow execution | Insufficient parallelism | Increase `n_workers` to match CPU cores |

### 11.2 Validation Checklist

Before production deployment:

- [ ] Verify all input file paths exist
- [ ] Confirm environment variables are set (`$DATA_ROOT`)
- [ ] Run `--validate-config` to check syntax
- [ ] Execute on sample data before full dataset
- [ ] Review output row counts against expectations

### 11.3 Diagnostic Mode

Enable detailed logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG ran-optimize \
  --config config/operators/my_operator.yaml \
  --export-diagnostics \
  2>&1 | tee debug.log
```

---

## 12. Operator-Specific Configuration

### 12.1 Multi-Operator Support

The RAN Optimizer supports multiple operators through a standardized directory structure. Each operator's data and configurations are kept separate.

#### Directory Structure

```
project-root/
├── data/
│   ├── vf-ie/              # Vodafone Ireland
│   │   ├── input-data/
│   │   └── output-data/
│   ├── dish/               # Dish Network
│   │   ├── input-data/
│   │   └── output-data/
│   └── three-uk/           # Three UK
│       ├── input-data/
│       └── output-data/
└── config/
    ├── overshooting_params.json      # Global defaults
    ├── pci_planner_params.json
    └── operators/                     # Operator-specific overrides
        ├── dish/
        │   └── pci_planner_params.json
        └── vf-ie/
            └── overshooting_params.json
```

### 12.2 Using the --data-dir Argument

The recommended way to run the optimizer is with the `--data-dir` argument:

```bash
# Run for Vodafone Ireland
python -m ran_optimizer.runner --data-dir data/vf-ie

# Run for Dish Network
python -m ran_optimizer.runner --data-dir data/dish
```

This automatically:
- Sets input directory to `<data-dir>/input-data/`
- Sets output directory to `<data-dir>/output-data/`
- Uses global config files from `config/`

### 12.3 Operator-Specific Parameters

#### When to Use Operator-Specific Configs

Create operator-specific configuration files when:
- The operator uses non-standard PCI ranges or placeholder values
- Network deployment patterns differ significantly from defaults
- Regulatory or business requirements demand different thresholds

#### Example: Dish Network PCI Configuration

Dish Network uses PCI value `0` as a placeholder for unconfigured cells. These should be excluded from collision detection:

**File:** `config/operators/dish/pci_planner_params.json`

```json
{
  "default": {
    "ignore_pcis": [0],
    "max_collision_radius_m": 30000.0,
    "include_mod3_inter_site": false
  }
}
```

**Usage:**
```bash
python -m ran_optimizer.runner \
  --data-dir data/dish \
  --config-dir config/operators/dish
```

#### Example: Vodafone Ireland Environment Tuning

If VF-IE has denser urban deployment than standard:

**File:** `config/operators/vf-ie/overshooting_params.json`

```json
{
  "environment_profiles": {
    "urban": {
      "min_cell_distance": 1200,
      "min_overshooting_grids": 20,
      "percentage_overshooting_grids": 0.08
    }
  }
}
```

**Usage:**
```bash
python -m ran_optimizer.runner \
  --data-dir data/vf-ie \
  --config-dir config/operators/vf-ie
```

### 12.4 Configuration Precedence

When running with operator-specific configs:

```
┌─────────────────────────────────────────────────────────────┐
│             CONFIGURATION PRECEDENCE (Highest → Lowest)     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CLI Arguments                                           │
│     --min-overshooting-grids 50                             │
│                                                             │
│  2. Operator-Specific Config                                │
│     config/operators/<operator>/overshooting_params.json    │
│                                                             │
│  3. Global Config                                           │
│     config/overshooting_params.json                         │
│                                                             │
│  4. Code Defaults                                           │
│     OvershooterParams class default values                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 12.5 Common Operator-Specific Settings

#### PCI Placeholder Values

Different operators use different PCI values for unconfigured/placeholder cells:

| Operator | Placeholder PCI | Config Setting |
|----------|----------------|----------------|
| Dish Network | 0 | `"ignore_pcis": [0]` |
| Vodafone Ireland | None (all valid) | `"ignore_pcis": []` |
| Generic | 0, 504-507 | `"ignore_pcis": [0, 504, 505, 506, 507]` |

#### Carrier Aggregation Pairs

Define operator-specific CA combinations in `ca_imbalance_params.json`:

**Dish Network Example:**
```json
{
  "ca_pairs": [
    {
      "name": "L700-L2100",
      "coverage_band": "L700",
      "capacity_band": "L2100",
      "coverage_threshold": 0.65
    }
  ]
}
```

**Vodafone Ireland Example:**
```json
{
  "ca_pairs": [
    {
      "name": "L800-L1800",
      "coverage_band": "L800",
      "capacity_band": "L1800",
      "coverage_threshold": 0.70
    },
    {
      "name": "L800-L2100",
      "coverage_band": "L800",
      "capacity_band": "L2100",
      "coverage_threshold": 0.70
    }
  ]
}
```

### 12.6 Best Practices

#### Start with Global Defaults

1. **First run:** Use global configs with `--data-dir` only
2. **Review results:** Check if detection quality is acceptable
3. **Only then customize:** Create operator-specific configs if needed

#### Minimal Configuration Changes

Only override parameters that truly differ from defaults:

**Good Example:**
```json
{
  "default": {
    "ignore_pcis": [0]
  }
}
```

**Bad Example (over-specification):**
```json
{
  "default": {
    "couple_cosectors": false,
    "min_active_neighbors_after_blacklist": 2,
    "max_collision_radius_m": 30000.0,
    "ignore_pcis": [0]
  }
}
```

The first example only changes what's needed. The second duplicates defaults unnecessarily, making future updates harder.

#### Document Your Rationale

When creating operator configs, add comments explaining why:

```json
{
  "_comment": "Dish Network uses PCI=0 for unconfigured cells, excluded from collision detection",
  "default": {
    "ignore_pcis": [0]
  }
}
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [[ADDING_NEW_OPERATOR]] | Step-by-step guide for onboarding new operators |
| [[ALGORITHMS]] | Detailed algorithm specifications |
| [[DATA_FORMATS]] | Input/output file schemas |
| [[API_REFERENCE]] | Python API documentation |

---

*Document maintained by the RAN Optimization Team. For configuration questions, contact network.optimization@example.com.*
