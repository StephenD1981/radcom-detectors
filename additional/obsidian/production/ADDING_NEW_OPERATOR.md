# Adding a New Operator

**Document Version:** 1.0
**Last Updated:** 2026-01-17
**Classification:** Operations Manual

---

## Document Purpose

This guide provides step-by-step instructions for adding a new mobile network operator to the RAN Optimizer system. It covers directory setup, data ingestion, configuration, and validation procedures.

**Audience:** Network Operations Engineers, Data Engineers, System Administrators

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure Setup](#2-directory-structure-setup)
3. [Data Preparation](#3-data-preparation)
4. [Configuration Files](#4-configuration-files)
5. [Running the Optimizer](#5-running-the-optimizer)
6. [Validation and Testing](#6-validation-and-testing)
7. [Operator-Specific Tuning](#7-operator-specific-tuning)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Overview

### 1.1 What You'll Need

Before starting, ensure you have:

- [ ] Input data files in the required format (see [Data Formats](DATA_FORMATS.md))
- [ ] Access to the RAN Optimizer installation
- [ ] Write permissions to the `data/` directory
- [ ] Understanding of your network's frequency bands and technologies

### 1.2 Time Estimate

| Task | Duration |
|------|----------|
| Directory setup | 5 minutes |
| Data preparation | 30-120 minutes (depends on data conversion) |
| Initial test run | 10-30 minutes |
| Configuration tuning | 2-8 hours (iterative) |
| Validation | 1-2 hours |

### 1.3 Process Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   NEW OPERATOR ONBOARDING FLOW                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Create Directory Structure                                  │
│     ├── data/<operator>/input-data/                            │
│     └── data/<operator>/output-data/                           │
│                                                                 │
│  2. Prepare Input Files                                         │
│     ├── cell_coverage.csv     (grid measurements)              │
│     ├── cell_gis.csv          (cell configuration)             │
│     ├── cell_hulls.csv        (coverage polygons)              │
│     └── cell_impacts.csv      (cell relations) [optional]      │
│                                                                 │
│  3. Configure Parameters                                        │
│     ├── Update config/*.json files if needed                   │
│     └── Set operator-specific settings                         │
│                                                                 │
│  4. Run Initial Analysis                                        │
│     └── python -m ran_optimizer.runner --data-dir data/<op>    │
│                                                                 │
│  5. Review & Tune                                               │
│     ├── Validate output files                                  │
│     ├── Adjust thresholds                                      │
│     └── Iterate until results are optimal                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure Setup

### 2.1 Create Operator Directory

Choose a short, descriptive name for your operator (lowercase, hyphens for spaces):

**Examples:**
- `vf-ie` (Vodafone Ireland)
- `dish` (Dish Network)
- `three-uk` (Three UK)
- `att-us` (AT&T United States)

```bash
# Navigate to the project root
cd /path/to/ran-optimizer

# Create operator directories
OPERATOR="your-operator-name"
mkdir -p "data/${OPERATOR}/input-data"
mkdir -p "data/${OPERATOR}/output-data"

# Verify structure
tree "data/${OPERATOR}"
```

Expected output:
```
data/your-operator-name/
├── input-data/
└── output-data/
```

### 2.2 Directory Layout Reference

The complete directory structure for an operator looks like this:

```
data/
└── <operator>/
    ├── input-data/              # Source data files (READ-ONLY)
    │   ├── cell_coverage.csv    # Required: Grid measurements
    │   ├── cell_gis.csv         # Required: Cell configuration
    │   ├── cell_hulls.csv       # Required: Coverage polygons
    │   ├── cell_impacts.csv     # Optional: Cell relations
    │   └── county_bounds/       # Optional: Boundary shapefile
    │       ├── bounds.shp
    │       ├── bounds.dbf
    │       ├── bounds.prj
    │       └── bounds.shx
    └── output-data/             # Generated results (WRITE)
        ├── overshooting_cells.csv
        ├── undershooting_cells.csv
        ├── no_coverage_*.geojson
        ├── low_coverage.geojson
        ├── interference_clusters.geojson
        ├── pci_*.csv
        ├── ca_imbalance.csv
        ├── crossed_feeder_*.csv
        ├── cell_environment.csv
        ├── maps/
        │   └── enhanced_dashboard.html
        └── pg_tables/
            └── *.csv
```

---

## 3. Data Preparation

### 3.1 Required Input Files

The RAN Optimizer requires data in a standardized schema. You'll need to transform your source data to match these specifications.

#### File 1: cell_coverage.csv

**Purpose:** Grid-level RF measurements (RSRP, distance, traffic per grid-cell pair)

**Minimum Required Columns:**

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `grid` | string | 7-character geohash identifier | `gc1zpnu` |
| `cilac` | integer | 9-digit cell identifier | `328167169` |
| `cell_name` | string | Human-readable cell name | `CK002H1` |
| `avg_rsrp` | float | Average RSRP (dBm) | `-107.0` |
| `distance_to_cell` | float | Distance from grid to cell (meters) | `3087.8` |
| `event_count` | integer | Traffic events in this grid-cell pair | `1` |
| `latitude` | float | Grid center latitude (WGS84) | `51.92099` |
| `longitude` | float | Grid center longitude (WGS84) | `-8.47447` |

**Full schema:** See [DATA_FORMATS.md](DATA_FORMATS.md#1-cell_coveragecsv)

#### File 2: cell_gis.csv

**Purpose:** Cell site and antenna configuration data

**Minimum Required Columns:**

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `Name` | string | Cell name (must match coverage data) | `CK002H1` |
| `CILAC` | integer | 9-digit cell identifier | `328167169` |
| `Latitude` | float | Cell site latitude (WGS84) | `51.92099` |
| `Longitude` | float | Cell site longitude (WGS84) | `-8.47447` |
| `Bearing` | integer | Antenna azimuth (degrees, 0-360) | `130` |
| `TiltE` | float | Electrical tilt (degrees) | `2.0` |
| `TiltM` | float | Mechanical tilt (degrees) | `0` |
| `Band` | string | Frequency band identifier | `L1800` |
| `Scr_Freq` | integer | Physical Cell ID (PCI) | `453` |

**Full schema:** See [DATA_FORMATS.md](DATA_FORMATS.md#2-cork-giscsv)

#### File 3: cell_hulls.csv

**Purpose:** Coverage boundary polygons for each cell

**Minimum Required Columns:**

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `cell_name` | string | Cell name | `CK002H1` |
| `cilac` | integer | 9-digit cell identifier | `328167169` |
| `geometry` | string | WKT polygon | `POLYGON ((-8.49... 51.90...))` |
| `area_km2` | float | Hull area (km²) | `17.757` |

**Full schema:** See [DATA_FORMATS.md](DATA_FORMATS.md#3-cell_hullscsv)

#### File 4: cell_impacts.csv (Optional)

**Purpose:** Cell-to-cell relationships for PCI planning and crossed feeder detection

**Required if running:** `pci`, `pci_conflict`, or `crossed_feeder` algorithms

**Minimum Required Columns:**

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `cell_name` | string | Source cell name | `CK093K3` |
| `cell_impact_name` | string | Target cell name | `CK652L1` |
| `cell_pci` | integer | Source cell PCI | `12` |
| `cell_impact_pci` | integer | Target cell PCI | `205` |
| `cell_band` | string | Source frequency band | `L700` |
| `cell_impact_band` | string | Target frequency band | `L800` |
| `distance` | float | Distance between cells (m) | `15190.47` |
| `relation_impact_data_perc` | float | Traffic percentage on relation | `0.43` |
| `co_site` | string | Same site? (Y/N) | `N` |
| `co_sectored` | string | Same cell ID? (Y/N) | `N` |

**Full schema:** See [DATA_FORMATS.md](DATA_FORMATS.md#4-cell_impactscsv)

### 3.2 Data Transformation Script

Here's a template script to help transform your data:

```python
import pandas as pd

# Load your source data (adjust paths and formats as needed)
raw_coverage = pd.read_csv('your_source_data/coverage.csv')
raw_gis = pd.read_csv('your_source_data/cells.csv')
raw_hulls = pd.read_csv('your_source_data/hulls.csv')

# Transform to RAN Optimizer schema
coverage_df = pd.DataFrame({
    'grid': raw_coverage['your_geohash_column'],
    'cilac': raw_coverage['your_cell_id_column'],
    'cell_name': raw_coverage['your_cell_name_column'],
    'avg_rsrp': raw_coverage['your_rsrp_column'],
    'distance_to_cell': raw_coverage['your_distance_column'],
    'event_count': raw_coverage['your_traffic_column'],
    'latitude': raw_coverage['your_lat_column'],
    'longitude': raw_coverage['your_lon_column'],
    # Add all other required columns...
})

gis_df = pd.DataFrame({
    'Name': raw_gis['your_cell_name_column'],
    'CILAC': raw_gis['your_cell_id_column'],
    'Latitude': raw_gis['your_lat_column'],
    'Longitude': raw_gis['your_lon_column'],
    'Bearing': raw_gis['your_azimuth_column'],
    'TiltE': raw_gis['your_etilt_column'],
    'TiltM': raw_gis['your_mtilt_column'],
    'Band': raw_gis['your_band_column'],
    'Scr_Freq': raw_gis['your_pci_column'],
    # Add all other required columns...
})

# Save to operator input directory
OPERATOR = "your-operator-name"
coverage_df.to_csv(f'data/{OPERATOR}/input-data/cell_coverage.csv', index=False)
gis_df.to_csv(f'data/{OPERATOR}/input-data/cell_gis.csv', index=False)
# ... save other files
```

### 3.3 Data Validation Checklist

Before running the optimizer, validate your data:

- [ ] **File Existence:** All required files are present
- [ ] **Column Names:** Match the expected schema exactly (case-sensitive)
- [ ] **Cell ID Consistency:** Same `cilac`/`CILAC` values appear in all files
- [ ] **Data Types:** Numeric columns contain numbers, not strings
- [ ] **RSRP Range:** Values are between -140 and -30 dBm
- [ ] **Distance Units:** All distances are in meters
- [ ] **PCI Range:** PCI values are between 0 and 503 (for LTE)
- [ ] **Azimuth Range:** Bearing values are between 0 and 360 degrees
- [ ] **Geohash Length:** All geohash values are exactly 7 characters
- [ ] **Geometry Format:** Hull geometries are valid WKT POLYGON strings

**Quick validation script:**

```python
import pandas as pd

OPERATOR = "your-operator-name"

# Load files
coverage = pd.read_csv(f'data/{OPERATOR}/input-data/cell_coverage.csv')
gis = pd.read_csv(f'data/{OPERATOR}/input-data/cell_gis.csv')

# Check column presence
required_coverage = ['grid', 'cilac', 'cell_name', 'avg_rsrp', 'distance_to_cell', 'event_count']
required_gis = ['Name', 'CILAC', 'Latitude', 'Longitude', 'Bearing', 'Band']

missing_coverage = [c for c in required_coverage if c not in coverage.columns]
missing_gis = [c for c in required_gis if c not in gis.columns]

print(f"Missing coverage columns: {missing_coverage}")
print(f"Missing GIS columns: {missing_gis}")

# Check cell ID overlap
coverage_cells = set(coverage['cilac'].unique())
gis_cells = set(gis['CILAC'].unique())
overlap = len(coverage_cells & gis_cells)
print(f"Cell overlap: {overlap}/{len(coverage_cells)} coverage cells found in GIS")

# Check RSRP range
rsrp_min, rsrp_max = coverage['avg_rsrp'].min(), coverage['avg_rsrp'].max()
print(f"RSRP range: {rsrp_min} to {rsrp_max} dBm")
if rsrp_min < -140 or rsrp_max > -30:
    print("WARNING: RSRP values outside expected range!")
```

---

## 4. Configuration Files

### 4.1 Global Configuration Files

The RAN Optimizer uses JSON configuration files in the `config/` directory. These are shared across all operators but can be customized if needed.

**Standard config files:**

```
config/
├── overshooting_params.json      # Overshooting detection thresholds
├── undershooting_params.json     # Undershooting detection thresholds
├── coverage_gaps.json            # No/low coverage detection
├── interference_params.json      # Interference cluster detection
├── pci_planner_params.json       # PCI planning parameters
├── pci_conflict_params.json      # PCI conflict detection
├── ca_imbalance_params.json      # Carrier aggregation analysis
└── crossed_feeder_params.json    # Crossed feeder detection
```

### 4.2 Operator-Specific Settings

Most operators can use the default configurations, but you may need to customize:

#### PCI Planning: Ignore PCIs

Some operators use placeholder PCI values (e.g., `0` for unconfigured cells). Update `config/pci_planner_params.json`:

```json
{
  "default": {
    "ignore_pcis": [0]
  }
}
```

**Example for Dish Network:**
- Dish uses PCI `0` as a placeholder
- Set `"ignore_pcis": [0]` to exclude these from collision detection

**Example for Vodafone Ireland:**
- All cells have valid PCIs
- Set `"ignore_pcis": []` (empty list)

#### Overshooting: Environment Thresholds

If your network has unusual inter-site distances, adjust environment classification in `config/overshooting_params.json`:

```json
{
  "environment_profiles": {
    "urban": {
      "min_cell_distance": 1500
    },
    "suburban": {
      "min_cell_distance": 3500
    },
    "rural": {
      "min_cell_distance": 8000
    }
  }
}
```

#### CA Imbalance: Band Pairs

Define which bands you use for carrier aggregation in `config/ca_imbalance_params.json`:

```json
{
  "ca_pairs": [
    {
      "name": "L700-L2100",
      "coverage_band": "L700",
      "capacity_band": "L2100",
      "coverage_threshold": 0.70
    },
    {
      "name": "L800-L1800",
      "coverage_band": "L800",
      "capacity_band": "L1800",
      "coverage_threshold": 0.70
    }
  ]
}
```

### 4.3 When to Create Operator-Specific Configs

**You probably don't need operator-specific configs if:**
- Your network uses standard frequency bands (700/800/1800/2100/2600 MHz)
- Inter-site distances are typical (urban: 0.5-1.5km, suburban: 1.5-5km, rural: 5-15km)
- All cells have valid PCI assignments
- You're using standard LTE carrier aggregation pairs

**Create operator-specific config files only if:**
- Your network uses non-standard PCI ranges
- You need dramatically different detection thresholds
- You have unique band combinations not covered by defaults

**To create operator-specific configs:**

```bash
# Create operator config directory
mkdir -p "config/operators/${OPERATOR}"

# Copy default configs
cp config/overshooting_params.json "config/operators/${OPERATOR}/"
cp config/pci_planner_params.json "config/operators/${OPERATOR}/"

# Edit as needed
vim "config/operators/${OPERATOR}/overshooting_params.json"

# Reference when running
python -m ran_optimizer.runner \
  --data-dir "data/${OPERATOR}" \
  --config-dir "config/operators/${OPERATOR}"
```

---

## 5. Running the Optimizer

### 5.1 Basic Usage (Recommended)

The simplest way to run the optimizer is using `--data-dir`:

```bash
# Run all algorithms with default settings
python -m ran_optimizer.runner --data-dir "data/your-operator-name"

# Or use the installed CLI command
ran-optimize --data-dir "data/your-operator-name"
```

This automatically:
- Looks for input files in `data/your-operator-name/input-data/`
- Saves results to `data/your-operator-name/output-data/`
- Uses environment-aware detection parameters
- Runs all available algorithms

### 5.2 Running Specific Algorithms

If you only need certain analyses:

```bash
# Coverage optimization only
ran-optimize \
  --data-dir "data/your-operator-name" \
  --algorithms overshooting undershooting

# Coverage gaps only
ran-optimize \
  --data-dir "data/your-operator-name" \
  --algorithms no_coverage low_coverage

# PCI analysis only
ran-optimize \
  --data-dir "data/your-operator-name" \
  --algorithms pci pci_conflict

# All physical layer checks
ran-optimize \
  --data-dir "data/your-operator-name" \
  --algorithms ca_imbalance crossed_feeder
```

**Available algorithms:**
- `overshooting` - Cells transmitting too far
- `undershooting` - Cells not reaching far enough
- `no_coverage` - Areas with zero coverage
- `no_coverage_per_band` - Band-specific coverage gaps
- `low_coverage` - Areas with weak signal (< -115 dBm)
- `interference` - High interference zones
- `pci` - PCI confusion and collision detection
- `pci_conflict` - Hull overlap-based PCI conflicts
- `ca_imbalance` - Carrier aggregation footprint mismatches
- `crossed_feeder` - Swapped antenna feeders

### 5.3 Advanced Options

```bash
# Specify custom config directory
ran-optimize \
  --data-dir "data/your-operator-name" \
  --config-dir "config/operators/your-operator-name"

# Use standard (non-environment-aware) detection
ran-optimize \
  --data-dir "data/your-operator-name" \
  --no-environment-aware

# Specify input/output explicitly (instead of --data-dir)
ran-optimize \
  --input-dir "data/your-operator-name/input-data" \
  --output-dir "data/your-operator-name/output-data"
```

### 5.4 Expected Execution Time

| Dataset Size | Grid Rows | Algorithms | Execution Time |
|--------------|-----------|------------|----------------|
| Small | < 1M | All | 5-10 minutes |
| Medium | 1M - 10M | All | 15-45 minutes |
| Large | 10M - 50M | All | 1-3 hours |
| Very Large | > 50M | All | 3-8 hours |

**Time-saving tips:**
- Run algorithms incrementally (coverage first, then PCI, etc.)
- Use `--algorithms` to limit scope
- Process regions separately if dataset is huge

---

## 6. Validation and Testing

### 6.1 Check Output Files

After running, verify that output files were created:

```bash
OPERATOR="your-operator-name"
ls -lh "data/${OPERATOR}/output-data/"
```

**Expected files:**

| File | Expected If... | Typical Size |
|------|----------------|--------------|
| `overshooting_cells.csv` | Ran overshooting algorithm | 10-500 KB |
| `undershooting_cells.csv` | Ran undershooting algorithm | 10-500 KB |
| `no_coverage_clusters.geojson` | Ran no_coverage algorithm | 1-100 KB |
| `no_coverage_<band>.geojson` | Ran no_coverage_per_band | 1-50 KB each |
| `low_coverage.geojson` | Ran low_coverage algorithm | 10-200 KB |
| `interference_clusters.geojson` | Ran interference algorithm | 5-100 KB |
| `pci_confusions.csv` | Ran pci algorithm | 5-100 KB |
| `pci_collisions.csv` | Ran pci algorithm | 5-100 KB |
| `pci_conflicts.csv` | Ran pci_conflict algorithm | 5-100 KB |
| `ca_imbalance.csv` | Ran ca_imbalance algorithm | 5-50 KB |
| `crossed_feeder_*.csv` | Ran crossed_feeder algorithm | 5-100 KB |
| `cell_environment.csv` | Always created | 50-500 KB |
| `maps/enhanced_dashboard.html` | Always created | 500 KB - 5 MB |

### 6.2 Validate Results

#### Check Record Counts

```bash
# Count overshooting cells
wc -l "data/${OPERATOR}/output-data/overshooting_cells.csv"

# Count undershooting cells
wc -l "data/${OPERATOR}/output-data/undershooting_cells.csv"

# Count PCI conflicts
wc -l "data/${OPERATOR}/output-data/pci_conflicts.csv"
```

**Sanity check expectations:**

| Metric | Typical Range | Flag If... |
|--------|---------------|------------|
| Overshooting cells | 2-15% of total cells | > 30% (thresholds too loose) |
| Undershooting cells | 1-10% of total cells | > 25% (thresholds too loose) |
| PCI conflicts | 0-5% of cell pairs | > 10% (PCI planning issue) |
| Coverage gaps | < 5% of service area | > 20% (data quality issue) |

#### Visual Inspection

Open the interactive dashboard:

```bash
# macOS
open "data/${OPERATOR}/output-data/maps/enhanced_dashboard.html"

# Linux
xdg-open "data/${OPERATOR}/output-data/maps/enhanced_dashboard.html"

# Windows
start "data/${OPERATOR}/output-data/maps/enhanced_dashboard.html"
```

**What to check:**
- [ ] Cell markers appear in the correct geographic region
- [ ] Overshooting/undershooting cells are highlighted
- [ ] Coverage gap polygons look reasonable (not scattered pixels)
- [ ] PCI conflict pairs are shown with connecting lines
- [ ] Map layers can be toggled on/off

#### Data Quality Checks

```python
import pandas as pd

OPERATOR = "your-operator-name"

# Load overshooting results
over_df = pd.read_csv(f'data/{OPERATOR}/output-data/overshooting_cells.csv')

# Check for suspicious patterns
print("Overshooting severity distribution:")
print(over_df['severity_tier'].value_counts())

print("\nTop 10 cells by overshooting grids:")
print(over_df.nlargest(10, 'overshooting_grids')[
    ['cell_name', 'overshooting_grids', 'percentage_overshooting', 'environment']
])

# Check environment distribution
print("\nEnvironment distribution:")
env_df = pd.read_csv(f'data/{OPERATOR}/output-data/cell_environment.csv')
print(env_df['environment'].value_counts())
```

**Red flags:**
- All cells flagged as same severity tier (config too aggressive/loose)
- All cells in same environment (classification not working)
- Extremely high percentages (> 80% overshooting) suggest data issue
- Zero results when issues are expected (thresholds too strict)

---

## 7. Operator-Specific Tuning

### 7.1 Iterative Refinement Process

1. **Run with defaults** - Get baseline results
2. **Review output counts** - Are too many/too few cells flagged?
3. **Adjust thresholds** - Make incremental changes
4. **Re-run and compare** - Did results improve?
5. **Validate with RF engineers** - Get domain expert feedback
6. **Document final settings** - Record what works for this operator

### 7.2 Common Tuning Scenarios

#### Scenario 1: Too Many Overshooting Cells Flagged

**Symptom:** > 30% of cells flagged, including cells that look reasonable

**Solution:** Tighten thresholds in `config/overshooting_params.json`

```json
{
  "default": {
    "min_overshooting_grids": 50,           // Increased from 30
    "percentage_overshooting_grids": 0.15,  // Increased from 0.10
    "min_cell_count_in_grid": 5             // Increased from 4
  }
}
```

#### Scenario 2: No Undershooting Cells Found (But Network Has Coverage Issues)

**Symptom:** Zero undershooting cells, but you know there are coverage gaps

**Solution:** Relax thresholds in `config/undershooting_params.json`

```json
{
  "default": {
    "min_new_grids_1deg": 3,                // Decreased from 5
    "min_coverage_increase_1deg": 0.02,     // Decreased from 0.04
    "max_interference_percentage": 0.30     // Increased from 0.20
  }
}
```

#### Scenario 3: Too Many PCI Conflicts (Including False Positives)

**Symptom:** Thousands of PCI conflicts, many at long distances

**Solution:** Reduce collision radius in `config/pci_planner_params.json`

```json
{
  "default": {
    "max_collision_radius_m": 20000.0,      // Decreased from 30000.0
    "ignore_pcis": [0, 504, 505]            // Add your placeholders
  }
}
```

#### Scenario 4: Coverage Gaps Look Noisy (Many Small Clusters)

**Symptom:** Hundreds of tiny coverage gap polygons

**Solution:** Increase clustering parameters in `config/coverage_gaps.json`

```json
{
  "no_coverage_detection": {
    "hdbscan_min_cluster_size": 20,         // Increased from 10
    "min_missing_neighbors": 45             // Increased from 40
  },
  "low_coverage_detection": {
    "min_area_km2": 1.0                     // Increased from 0.5
  }
}
```

### 7.3 Environment-Specific Tuning

If your network has atypical environment characteristics:

```json
{
  "environment_profiles": {
    "urban": {
      "min_cell_distance": 1500,
      "min_overshooting_grids": 25,
      "percentage_overshooting_grids": 0.08
    },
    "suburban": {
      "min_cell_distance": 3500,
      "min_overshooting_grids": 25,
      "percentage_overshooting_grids": 0.10
    },
    "rural": {
      "min_cell_distance": 8000,
      "min_overshooting_grids": 22,
      "percentage_overshooting_grids": 0.07
    }
  }
}
```

**When to adjust:**
- **Dense urban (< 500m ISD):** Decrease `min_cell_distance` to 1000m
- **Sparse rural (> 10km ISD):** Increase `min_cell_distance` to 10000m
- **Mixed network:** Create custom thresholds per environment

---

## 8. Troubleshooting

### 8.1 Common Errors

#### Error: "FileNotFoundError: Grid data not found"

**Cause:** Input file missing or wrong path

**Solution:**
```bash
# Check files exist
ls "data/${OPERATOR}/input-data/"

# Verify filenames match exactly (case-sensitive)
# Required: cell_coverage.csv, cell_gis.csv, cell_hulls.csv
```

#### Error: "KeyError: 'cilac'"

**Cause:** Column name mismatch in input files

**Solution:**
```python
# Check column names
import pandas as pd
df = pd.read_csv('data/your-operator/input-data/cell_coverage.csv')
print(df.columns.tolist())

# Must include: grid, cilac, cell_name, avg_rsrp, distance_to_cell, etc.
```

#### Error: "Cell overlap: 0/1660 coverage cells found in GIS"

**Cause:** Cell IDs don't match between files

**Solution:**
```python
import pandas as pd

# Load both files
coverage = pd.read_csv('data/your-operator/input-data/cell_coverage.csv')
gis = pd.read_csv('data/your-operator/input-data/cell_gis.csv')

# Check data types
print(f"Coverage cilac type: {coverage['cilac'].dtype}")
print(f"GIS CILAC type: {gis['CILAC'].dtype}")

# Check sample values
print(f"Coverage sample: {coverage['cilac'].head()}")
print(f"GIS sample: {gis['CILAC'].head()}")

# Solution: Convert to same type
coverage['cilac'] = coverage['cilac'].astype(str)
gis['CILAC'] = gis['CILAC'].astype(str)
```

#### Warning: "Cell hulls not available - coverage gap detection will be skipped"

**Cause:** `cell_hulls.csv` file missing

**Impact:** No coverage and low coverage detection won't run

**Solution:**
```bash
# Verify file exists
ls "data/${OPERATOR}/input-data/cell_hulls.csv"

# If missing, you need to generate coverage hulls from your data
# Or run without coverage gap detection:
ran-optimize \
  --data-dir "data/${OPERATOR}" \
  --algorithms overshooting undershooting pci
```

### 8.2 Performance Issues

#### Issue: Execution Takes Too Long (> 2 hours for medium dataset)

**Solutions:**

1. **Run algorithms separately:**
```bash
# Split into multiple runs
ran-optimize --data-dir "data/${OPERATOR}" --algorithms overshooting
ran-optimize --data-dir "data/${OPERATOR}" --algorithms undershooting
ran-optimize --data-dir "data/${OPERATOR}" --algorithms pci
```

2. **Reduce parallelism if hitting memory limits:**
```json
// In config files
{
  "processing": {
    "n_workers": 2,       // Reduce from 4
    "chunk_size": 50000   // Reduce from 100000
  }
}
```

3. **Process regions separately:**
```python
# Filter input data by region
import pandas as pd
import geopandas as gpd

coverage = pd.read_csv('data/operator/input-data/cell_coverage.csv')

# Filter to specific region
region_coverage = coverage[
    (coverage['latitude'] >= 51.0) &
    (coverage['latitude'] <= 52.0)
]

region_coverage.to_csv('data/operator/input-data/cell_coverage_region1.csv', index=False)
```

### 8.3 Data Quality Issues

#### Issue: RSRP Values Look Wrong (e.g., all positive numbers)

**Cause:** RSRP exported in linear scale instead of dBm

**Solution:**
```python
import pandas as pd
import numpy as np

coverage = pd.read_csv('data/operator/input-data/cell_coverage.csv')

# If RSRP is in linear scale (0-100), convert to dBm
if coverage['avg_rsrp'].min() > 0:
    coverage['avg_rsrp'] = 10 * np.log10(coverage['avg_rsrp']) - 140

coverage.to_csv('data/operator/input-data/cell_coverage.csv', index=False)
```

#### Issue: Distance Units Wrong (kilometers instead of meters)

**Cause:** Source data uses different units

**Solution:**
```python
import pandas as pd

coverage = pd.read_csv('data/operator/input-data/cell_coverage.csv')

# If distance is in km, convert to meters
if coverage['distance_to_cell'].max() < 100:  # Likely in km
    coverage['distance_to_cell'] = coverage['distance_to_cell'] * 1000

coverage.to_csv('data/operator/input-data/cell_coverage.csv', index=False)
```

#### Issue: Geohash Column Missing

**Cause:** Source data doesn't include geohash encoding

**Solution:**
```python
import pandas as pd
import geohash2

coverage = pd.read_csv('data/operator/input-data/cell_coverage.csv')

# Generate geohash from lat/lon (precision 7)
coverage['grid'] = coverage.apply(
    lambda row: geohash2.encode(row['latitude'], row['longitude'], precision=7),
    axis=1
)

coverage.to_csv('data/operator/input-data/cell_coverage.csv', index=False)
```

---

## 9. Checklist: New Operator Onboarding

Use this checklist to track your progress:

### Pre-Flight
- [ ] Python 3.11+ installed
- [ ] RAN Optimizer installed (`ran-optimize --help` works)
- [ ] Operator name chosen (lowercase, hyphens)
- [ ] Source data files obtained

### Directory Setup
- [ ] Created `data/<operator>/input-data/`
- [ ] Created `data/<operator>/output-data/`

### Data Preparation
- [ ] `cell_coverage.csv` prepared and validated
- [ ] `cell_gis.csv` prepared and validated
- [ ] `cell_hulls.csv` prepared and validated
- [ ] `cell_impacts.csv` prepared (if needed)
- [ ] Cell ID consistency verified across all files
- [ ] Column names match schema exactly
- [ ] Data types are correct (numeric columns are numbers)
- [ ] RSRP values in valid range (-140 to -30 dBm)
- [ ] Distance values in meters
- [ ] Geohash values are 7 characters

### Configuration
- [ ] Reviewed default config files
- [ ] Updated `ignore_pcis` if needed
- [ ] Created operator-specific configs (if needed)
- [ ] Updated CA band pairs (if needed)

### First Run
- [ ] Ran optimizer with `--data-dir`
- [ ] Execution completed without errors
- [ ] Output files created
- [ ] Dashboard HTML generated

### Validation
- [ ] Checked record counts are reasonable
- [ ] Reviewed dashboard visually
- [ ] Validated overshooting results with RF team
- [ ] Validated undershooting results with RF team
- [ ] Checked PCI conflict counts
- [ ] Reviewed coverage gap polygons

### Tuning (Iterative)
- [ ] Adjusted thresholds based on results
- [ ] Re-ran and compared outputs
- [ ] Documented final configuration
- [ ] Obtained RF engineer sign-off

### Production Readiness
- [ ] Created runbook for this operator
- [ ] Scheduled automated runs (if applicable)
- [ ] Set up result delivery/integration
- [ ] Trained operations team

---

## 10. Quick Reference

### Directory Structure
```
data/<operator>/
├── input-data/
│   ├── cell_coverage.csv (required)
│   ├── cell_gis.csv (required)
│   ├── cell_hulls.csv (required)
│   └── cell_impacts.csv (optional)
└── output-data/ (generated)
```

### Run Commands
```bash
# Full run with all algorithms
ran-optimize --data-dir data/<operator>

# Specific algorithms only
ran-optimize --data-dir data/<operator> --algorithms overshooting undershooting

# Custom config directory
ran-optimize --data-dir data/<operator> --config-dir config/operators/<operator>
```

### Key Config Files
- `config/pci_planner_params.json` → `ignore_pcis` for placeholder PCIs
- `config/overshooting_params.json` → environment-specific thresholds
- `config/ca_imbalance_params.json` → CA band pair definitions

### Validation Quick Checks
```bash
# File existence
ls data/<operator>/input-data/

# Record counts
wc -l data/<operator>/output-data/*.csv

# Open dashboard
open data/<operator>/output-data/maps/enhanced_dashboard.html
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [DATA_FORMATS](DATA_FORMATS.md) | Complete data schema specifications |
| [CONFIGURATION](CONFIGURATION.md) | Detailed parameter tuning guide |
| [README](README.md) | Quick start and overview |
| [ALGORITHMS](ALGORITHMS.md) | Algorithm specifications |

---

*For questions or issues, contact the RAN Optimization Team.*
