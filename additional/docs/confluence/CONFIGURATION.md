# RAN Optimizer Configuration Guide

**Document Version:** 2.0
**Last Updated:** 2024
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
| `min_relative_reach` | float | 0.7 | 0.5 - 0.9 | Minimum reach relative to furthest cell |
| `rsrp_degradation_db` | float | 10.0 | 5.0 - 20.0 | Minimum RSRP drop from cell maximum (dB) |
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
| `min_cell_max_distance` | int | 3000 | 1000 - 10000 | Range below this indicates undershooting (m) |
| `min_grid_count` | int | 100 | 50 - 500 | Minimum grids required for analysis |
| `interference_pct_threshold` | float | 0.40 | 0.20 - 0.60 | Edge interference ratio to flag |
| `traffic_pct_threshold` | float | 0.15 | 0.05 - 0.30 | Minimum edge traffic to indicate demand |
| `min_rsrp_threshold` | int | -105 | -120 to -90 | Ignore grids weaker than this (dBm) |

### 4.2 Band-Specific Defaults

Propagation characteristics vary significantly by frequency:

| Parameter | Low Band (700-900 MHz) | Mid Band (1800 MHz) | High Band (2100+ MHz) |
|-----------|------------------------|---------------------|----------------------|
| `min_cell_max_distance` | 5000m | 3500m | 2500m |
| `traffic_pct_threshold` | 0.10 | 0.15 | 0.20 |

---

## 5. Coverage Gap Parameters

### 5.1 Parameter Reference

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `rsrp_threshold` | int | -115 | -125 to -105 | Signal level defining "low coverage" (dBm) |
| `min_cluster_points` | int | 10 | 5 - 50 | Minimum geohashes to form a cluster |
| `cluster_eps` | float | 0.003 | 0.001 - 0.01 | HDBSCAN clustering distance |
| `min_cluster_area_km2` | float | 0.1 | 0.01 - 1.0 | Minimum reportable cluster area (km²) |
| `k_ring_size` | int | 2 | 1 - 4 | Neighbor ring size for density validation |
| `min_density_ratio` | float | 0.30 | 0.20 - 0.50 | Required neighbor ratio for retention |

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
    "edge_traffic_percent": 0.15
  },
  "environment_profiles": {
    "urban": {
      "min_cell_distance": 3000,
      "min_overshooting_grids": 30,
      "edge_traffic_percent": 0.15,
      "min_cell_count_in_grid": 5
    },
    "suburban": {
      "min_cell_distance": 5000,
      "min_overshooting_grids": 50,
      "edge_traffic_percent": 0.12,
      "min_cell_count_in_grid": 4
    },
    "rural": {
      "min_cell_distance": 8000,
      "min_overshooting_grids": 80,
      "edge_traffic_percent": 0.08,
      "min_cell_count_in_grid": 3
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

## 9. Troubleshooting

### 9.1 Common Issues

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Too many cells flagged | Thresholds too low | Increase `min_overshooting_grids`, `percentage_overshooting_grids` |
| Too few cells flagged | Thresholds too high | Decrease thresholds; verify data quality |
| Missing expected cells | Filtered by pre-checks | Check `min_cell_distance`, verify cell range in data |
| Memory exhaustion | Large dataset | Reduce `chunk_size`; process in batches |
| Slow execution | Insufficient parallelism | Increase `n_workers` to match CPU cores |

### 9.2 Validation Checklist

Before production deployment:

- [ ] Verify all input file paths exist
- [ ] Confirm environment variables are set (`$DATA_ROOT`)
- [ ] Run `--validate-config` to check syntax
- [ ] Execute on sample data before full dataset
- [ ] Review output row counts against expectations

### 9.3 Diagnostic Mode

Enable detailed logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG ran-optimize \
  --config config/operators/my_operator.yaml \
  --export-diagnostics \
  2>&1 | tee debug.log
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [[ALGORITHMS]] | Detailed algorithm specifications |
| [[DATA_FORMATS]] | Input/output file schemas |
| [[API_REFERENCE]] | Python API documentation |

---

*Document maintained by the RAN Optimization Team. For configuration questions, contact network.optimization@example.com.*
