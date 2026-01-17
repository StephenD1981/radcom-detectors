# Production Readiness Plan

## Overview

This document outlines the plan to make the RAN Optimizer production-ready for AWS deployment as a Docker image. The system will support multiple input sources (CSV files, PostgreSQL) and produce standardized outputs for 8 detection algorithms.

---

## Current State Assessment

### Project Structure
- **Location:** `ran_optimizer/` package
- **Existing Detectors:** 7 partially implemented (overshooters, undershooters, coverage_gaps, interference, ca_imbalance, crossed_feeder, pci_conflict)
- **Data Layer:** Pydantic schemas, loaders, adapters
- **Configuration:** YAML-based with Pydantic validation
- **Runner:** Basic CLI in `runner.py` and `cli.py`

### Input Data (Vodafone Ireland)
| File | Rows | Description |
|------|------|-------------|
| `cell_coverage.csv` | 1.95M | Grid-level coverage metrics |
| `cell_gis.csv` | 1,661 | Cell master directory |
| `cell_hulls.csv` | 1,659 | Convex hull coverage areas |
| `cell_impacts.csv` | 142K | Cell-to-cell impact relationships |

---

## Phase 1: Configuration System

### 1.1 JSON Configuration File Structure

Create `config/pipeline_config.json`:

```json
{
  "version": "1.0",
  "operator": "vodafone_ireland",
  "region": "cork",

  "inputs": {
    "source_type": "csv",
    "csv": {
      "base_path": "data/vf-ie/input-data",
      "files": {
        "coverage": "cell_coverage.csv",
        "gis": "cell_gis.csv",
        "hulls": "cell_hulls.csv",
        "impacts": "cell_impacts.csv"
      }
    },
    "postgres": {
      "enabled": false,
      "host": "${DB_HOST}",
      "port": 5432,
      "database": "${DB_NAME}",
      "tables": {
        "coverage": "cell_coverage",
        "gis": "cell_gis",
        "hulls": "cell_hulls",
        "impacts": "cell_impacts"
      }
    }
  },

  "outputs": {
    "base_path": "data/vf-ie/output-data",
    "formats": {
      "geojson": true,
      "csv": true
    }
  },

  "detectors": {
    "low_coverage": {"enabled": true},
    "no_coverage": {"enabled": true},
    "interference": {"enabled": true},
    "overshooters": {"enabled": true},
    "undershooters": {"enabled": true},
    "ca_imbalance": {"enabled": true},
    "crossed_feeders": {"enabled": true},
    "pci_conflict": {"enabled": true}
  },

  "processing": {
    "chunk_size": 100000,
    "n_workers": 4
  }
}
```

### 1.2 Implementation Tasks

| Task | File | Description |
|------|------|-------------|
| 1.2.1 | `ran_optimizer/utils/pipeline_config.py` | Create Pydantic models for pipeline config |
| 1.2.2 | `ran_optimizer/data/sources.py` | Abstract data source interface (CSV/Postgres) |
| 1.2.3 | `ran_optimizer/data/postgres_loader.py` | PostgreSQL data loader |
| 1.2.4 | `config/pipeline_config.json` | Default configuration file |

---

## Phase 2: Detector Upgrades

### 2.1 Detectors from Reference Project (ARANO)

These detectors should be upgraded from the reference project at `/Users/stephendillon/Library/Mobile Documents/com~apple~CloudDocs/5-deployments/1-arano`:

#### 2.1.1 Low Coverage Detector
- **Source:** `hackathon-backend/python/detectors/coverage_gaps.py` → `LowCoverageDetector`
- **Target:** `ran_optimizer/recommendations/low_coverage.py`
- **Algorithm:**
  1. Find single-server regions (cell area minus overlapping cells)
  2. Extract geohashes in single-server areas
  3. Filter by RSRP threshold per band (-115 dBm default)
  4. Apply k-ring density filtering
  5. Cluster with HDBSCAN
  6. Create alpha shape polygons
- **Inputs:** `cell_coverage.csv`, `cell_hulls.csv`
- **Outputs:** GeoJSON + Summary CSV

#### 2.1.2 No Coverage Detector
- **Source:** `hackathon-backend/python/detectors/coverage_gaps.py` → `CoverageGapDetector`
- **Target:** `ran_optimizer/recommendations/no_coverage.py`
- **Algorithm:**
  1. Cluster cell hulls using DBSCAN
  2. Create cluster convex hulls
  3. Find gap polygons (cluster minus cell coverage)
  4. K-ring density filtering
  5. HDBSCAN clustering
  6. Alpha shape polygons
- **Inputs:** `cell_hulls.csv`
- **Outputs:** GeoJSON + Summary CSV

#### 2.1.3 Interference Detector
- **Source:** `hackathon-backend/python/detectors/interference/`
- **Target:** `ran_optimizer/recommendations/interference.py`
- **Algorithm:**
  1. Band-level processing
  2. RSRP similarity filtering (within 5dB of P90)
  3. Clustering by RSRP similarity
  4. Dominance detection
  5. K-ring spatial clustering
  6. Per-cell aggregation
- **Inputs:** `cell_coverage.csv`
- **Outputs:** GeoJSON + Summary CSV

#### 2.1.4 Overshooters Detector
- **Source:** `hackathon-backend/python/detectors/overshooters.py`
- **Target:** `ran_optimizer/recommendations/overshooters.py`
- **Algorithm:**
  1. Calculate grid-to-cell distances
  2. Identify edge traffic bins (top 15% distance)
  3. Calculate per-cell metrics
  4. Apply overshooting filters
  5. Calculate tilt recommendations (3GPP antenna pattern)
  6. Severity scoring
- **Inputs:** `cell_coverage.csv`, `cell_gis.csv`
- **Outputs:** Grid CSV + Summary CSV

#### 2.1.5 Undershooters Detector
- **Source:** `hackathon-backend/python/detectors/undershooters.py`
- **Target:** `ran_optimizer/recommendations/undershooters.py`
- **Algorithm:**
  1. Filter by max distance threshold
  2. Calculate RSRP-based interference
  3. Estimate uptilt impact
  4. Recommend uptilt if criteria met
- **Inputs:** `cell_coverage.csv`, `cell_gis.csv`
- **Outputs:** Grid CSV + Summary CSV

### 2.2 Existing Detector (Keep As-Is)

#### 2.2.1 CA Imbalance Detector
- **Location:** `ran_optimizer/recommendations/ca_imbalance.py`
- **Status:** Uses correct schema, integrate as-is
- **Inputs:** `cell_hulls.csv` (multi-band)
- **Outputs:** GeoJSON (hulls) + Summary CSV

### 2.3 Detectors Requiring Schema Migration

These detectors need migration from `relations.csv` schema to `cell_impacts.csv` schema:

#### Column Mapping: relations.csv → cell_impacts.csv

| Old Column (relations.csv) | New Column (cell_impacts.csv) | Notes |
|---------------------------|-------------------------------|-------|
| `cell_name` | `cell_name` | Same |
| `to_cell_name` | `cell_impact_name` | Renamed |
| `distance` | `distance` | Same |
| `pci` | `cell_pci` | Renamed |
| `to_pci` | `cell_impact_pci` | Renamed |
| `band` | `cell_band` | Renamed |
| `to_band` | `cell_impact_band` | Renamed |
| `intra_site` | `co_site` | Renamed |
| `intra_cell` | `co_sectored` | Renamed |
| `weight` | `total_cell_traffic_data` | Traffic metric |
| `cell_perc_weight` | `relation_impact_data_perc` | Percentage |

#### 2.3.1 Crossed Feeders Detector
- **Location:** `ran_optimizer/recommendations/crossed_feeder.py`
- **Changes Required:**
  1. Update column name references per mapping above
  2. Filter to cells in `cell_gis.csv`
  3. Add bearing calculations from `cell_gis.csv`
- **Inputs:** `cell_impacts.csv`, `cell_gis.csv`
- **Outputs:** Relational CSV + Summary CSV

#### 2.3.2 PCI Conflict Detector
- **Location:** `ran_optimizer/recommendations/pci_conflict.py`
- **Changes Required:**
  1. Update column name references per mapping above
  2. Filter to cells in `cell_gis.csv`
  3. **Filter out rows where `cell_pci` OR `cell_impact_pci` = 'N/A'**
  4. Detect: PCI Confusion, PCI Collision, Blacklist Suggestions
- **Inputs:** `cell_impacts.csv`, `cell_gis.csv`
- **Outputs:** PCI Issues CSV + Summary CSV

---

## Phase 3: Output Specifications

### 3.1 Output Directory Structure

```
data/vf-ie/output-data/
├── low_coverage/
│   ├── low_coverage_areas.geojson
│   └── low_coverage_summary.csv
├── no_coverage/
│   ├── no_coverage_areas.geojson
│   └── no_coverage_summary.csv
├── interference/
│   ├── interference_areas.geojson
│   └── interference_summary.csv
├── overshooters/
│   ├── overshooter_grids.csv
│   └── overshooter_summary.csv
├── undershooters/
│   ├── undershooter_grids.csv
│   └── undershooter_summary.csv
├── ca_imbalance/
│   ├── ca_imbalance_hulls.geojson
│   └── ca_imbalance_summary.csv
├── crossed_feeders/
│   ├── crossed_feeder_relations.csv
│   └── crossed_feeder_summary.csv
└── pci/
    ├── pci_confusion.csv
    ├── pci_collision.csv
    ├── pci_blacklist.csv
    └── pci_summary.csv
```

### 3.2 Output Schema Specifications

#### Low Coverage / No Coverage GeoJSON
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [...]},
    "properties": {
      "cluster_id": 1,
      "area_km2": 2.5,
      "centroid_lat": 51.89,
      "centroid_lon": -8.47,
      "n_points": 45,
      "serving_cells": ["CK001H1", "CK002H1"],
      "severity": "HIGH"
    }
  }]
}
```

#### Overshooter/Undershooter Summary CSV
```csv
cell_name,total_grids,affected_grids,percentage,max_distance_m,recommended_tilt_change,severity_score,severity_category
CK001H1,500,75,0.15,8500,2,0.72,HIGH
```

#### PCI Issues CSV
```csv
cell_name,cell_pci,cell_impact_name,cell_impact_pci,issue_type,distance,cell_band,severity
CK001H1,123,CK045H1,123,collision,2500,L1800,HIGH
```

---

## Phase 4: Implementation Order

### Stage 1: Foundation (Tasks 1-5)
1. [ ] Create `ran_optimizer/utils/pipeline_config.py` - Pydantic config models
2. [ ] Create `config/pipeline_config.json` - Default configuration
3. [ ] Create `ran_optimizer/data/sources.py` - Data source abstraction
4. [ ] Update `ran_optimizer/runner.py` - Load config and dispatch
5. [ ] Create output directory structure and writers

### Stage 2: Detector Upgrades from ARANO (Tasks 6-10)
6. [ ] Upgrade `low_coverage.py` from reference
7. [ ] Upgrade `no_coverage.py` from reference
8. [ ] Upgrade `interference.py` from reference
9. [ ] Upgrade `overshooters.py` from reference
10. [ ] Upgrade `undershooters.py` from reference

### Stage 3: Schema Migration (Tasks 11-12)
11. [ ] Migrate `crossed_feeder.py` to cell_impacts.csv schema
12. [ ] Migrate `pci_conflict.py` to cell_impacts.csv schema

### Stage 4: Integration (Tasks 13-15)
13. [ ] Integrate `ca_imbalance.py` (existing)
14. [ ] Create unified runner with all 8 detectors
15. [ ] Add Docker configuration

---

## Phase 5: Docker Deployment

### 5.1 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ran_optimizer/ ran_optimizer/
COPY config/ config/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "ran_optimizer.runner"]
CMD ["--config", "/config/pipeline_config.json"]
```

### 5.2 Docker Compose (for local testing)
```yaml
version: '3.8'
services:
  ran-optimizer:
    build: .
    volumes:
      - ./data:/data
      - ./config:/config
    environment:
      - DB_HOST=${DB_HOST}
      - DB_NAME=${DB_NAME}
```

---

## Clarified Requirements

| Question | Answer |
|----------|--------|
| PostgreSQL Schema | Exact mapping to CSV column names |
| Interference Output | GeoJSON convex hulls (same as low/no coverage) |
| Band Processing | Per-band: low coverage, interference, PCI |
| Severity Scoring | Consistent scale: CRITICAL/HIGH/MEDIUM/LOW/MINIMAL |

---

## PCI Detector - Complete Specification

The PCI detector uses **relation-based analysis** from `cell_impacts.csv` (adapted from `pci_planner.py` in reference):

### 1. PCI Confusion
- **Definition:** Serving cell has 2+ neighbors sharing the same PCI on the same band
- **Impact:** UE measurement ambiguity during handovers
- **Detection:** Group neighbors by (PCI, band), flag groups with size ≥ 2
- **Severity:** Sum of HO activity excluding strongest neighbor
- **Output columns:** `serving, confusion_pci, band, group_size, neighbors, severity`

### 2. PCI Collision
- **Definition:** Cell pairs with same PCI on same band that are mobility-relevant
- **Types:**
  - **1-hop:** Direct neighbor relations
  - **2-hop:** Neighbors of neighbors (weighted at 0.25x)
- **Distance filter:** ≤30km (configurable)
- **Severity:** Logarithmic weighting of handover volumes
- **Output columns:** `cell_a, cell_b, pci, band, collision_type, pair_weight, severity`

### 3. Blacklist Suggestions
- **AUTO_DEAD_RELATION:** HO = 0 in both directions → auto-apply
- **SUGGEST_LOW_ACTIVITY:** Low activity (≤5 HO, ≤0.1% share) → manual review
- **REJECT_INTRA_SITE:** Never blacklist intra-site relations
- **REJECT_MIN_NEIGHBORS:** Would leave <2 active neighbors
- **Output columns:** `serving, neighbor, reason, out_ho, in_ho, act_ho, share, confusion_pci`

---

## Updated Output Directory Structure

```
data/vf-ie/output-data/
├── low_coverage/
│   ├── low_coverage_areas.geojson      # Per-band alpha shape polygons
│   └── low_coverage_summary.csv
├── no_coverage/
│   ├── no_coverage_areas.geojson       # Gap alpha shape polygons
│   └── no_coverage_summary.csv
├── interference/
│   ├── interference_areas.geojson      # Per-band interference hulls
│   └── interference_summary.csv
├── overshooters/
│   ├── overshooter_grids.csv           # Grid-level data
│   └── overshooter_summary.csv
├── undershooters/
│   ├── undershooter_grids.csv          # Grid-level data
│   └── undershooter_summary.csv
├── ca_imbalance/
│   ├── ca_imbalance_hulls.geojson      # Coverage/capacity hull pairs
│   └── ca_imbalance_summary.csv
├── crossed_feeders/
│   ├── crossed_feeder_relations.csv    # Relation-level scores
│   ├── crossed_feeder_cells.csv        # Cell-level aggregation
│   └── crossed_feeder_summary.csv
└── pci/
    ├── pci_confusion.csv               # Per-band confusion issues
    ├── pci_collision.csv               # Per-band collision issues
    ├── pci_blacklist_suggestions.csv   # Blacklist recommendations
    └── pci_summary.csv
```

---

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 (Configuration System)
3. Proceed through phases sequentially
4. Validate each detector output before moving to next
5. Final integration testing with all 8 detectors
