# Data Format Specification

| Document Information |                                      |
|---------------------|--------------------------------------|
| **Version**         | 2.0                                  |
| **Classification**  | Data Integration Specification       |
| **Last Updated**    | November 2024                        |
| **Audience**        | Data Engineers, System Integrators   |

---

## Table of Contents

1. [Overview](#overview)
2. [Input Data Requirements](#input-data-requirements)
   - [Grid Data](#input-file-1-grid-data)
   - [GIS Data](#input-file-2-gis-data)
   - [Hull Data](#input-file-3-hull-data)
   - [Boundary Shapefile](#input-file-4-boundary-shapefile)
3. [Output Data Specifications](#output-data-specifications)
   - [Overshooting Results](#output-file-1-overshooting-results)
   - [Undershooting Results](#output-file-2-undershooting-results)
   - [Daily Resolution Recommendations](#output-file-3-daily-resolution-recommendations)
   - [Environment Classification](#output-file-4-environment-classification)
   - [Coverage Gap Results](#output-file-5-coverage-gap-results)
   - [Interactive Dashboard](#output-file-6-interactive-dashboard)
4. [Data Validation](#data-validation)
5. [Data Type Reference](#data-type-reference)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│   Grid Data     │    GIS Data     │   Hull Data     │   Boundary    │
│   (Required)    │   (Required)    │   (Required)    │   (Optional)  │
│                 │                 │                 │               │
│ cell_coverage   │     gis.csv     │ cell_hulls.csv  │  bounds.shp   │
│     .csv        │                 │                 │               │
└────────┬────────┴────────┬────────┴────────┬────────┴───────┬───────┘
         │                 │                 │                │
         └─────────────────┴─────────────────┴────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       RAN OPTIMIZER           │
                    │   Detection & Analysis        │
                    └───────────────────────────────┘
                                    │
         ┌─────────────────┬────────┴────────┬─────────────────┐
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Overshooting│   │Undershooting│   │   Daily     │   │  Coverage   │
│   Results   │   │   Results   │   │ Resolution  │   │    Gaps     │
│    .csv     │   │    .csv     │   │    .csv     │   │  .geojson   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Interactive Dashboard     │
                    │         .html                 │
                    └───────────────────────────────┘
```

### File Summary

| File Type | Format | Required | Purpose |
|-----------|--------|----------|---------|
| Grid Data | CSV | Yes | Signal measurements from UE devices |
| GIS Data | CSV | Yes | Cell site locations and configurations |
| Hull Data | CSV | Yes | Cell coverage boundary polygons |
| Boundary | Shapefile | No | Analysis region boundary |

---

## Input Data Requirements

### Input File 1: Grid Data

**Filename:** `cell_coverage.csv` (configurable)

#### Purpose

Contains radio frequency measurements aggregated by cell and geographic grid. Each row represents signal quality measurements from a specific cell at a specific location.

#### Data Sources

| Source | Description | Typical Volume |
|--------|-------------|----------------|
| MDT (Minimization of Drive Tests) | Automated UE measurements | Millions/day |
| Drive Tests | Manual test vehicle measurements | Thousands/campaign |
| Network Probes | Fixed measurement points | Continuous |

#### Required Schema

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `cilac` OR `cell_id` | String/Integer | Cell identifier | Must match GIS data |
| `grid` OR `geohash7` | String | 7-character geohash | Length = 7 |
| `rsrp_mean` | Float | Mean RSRP (dBm) | Range: -140 to -40 |
| `distance_m` | Float | Distance from cell (meters) | ≥ 0 |

#### Optional Schema (Recommended)

| Column | Data Type | Description | Default |
|--------|-----------|-------------|---------|
| `Latitude` | Float | Grid center latitude | Decoded from geohash |
| `Longitude` | Float | Grid center longitude | Decoded from geohash |
| `rsrp_std` | Float | RSRP standard deviation | - |
| `sample_count` | Integer | Number of measurements | 1 |
| `event_count` | Integer | Traffic event count | 1 |
| `ta_mean` | Float | Mean timing advance | - |
| `traffic_pct` | Float | Traffic distribution % | - |
| `band` | Integer | Frequency band (MHz) | - |

#### Example

```csv
cilac,grid,rsrp_mean,distance_m,Latitude,Longitude,event_count,band
328576779,gc7x9r5,-95.2,1250.5,51.8976,-8.4723,156,1800
328576779,gc7x9r4,-98.7,1450.2,51.8965,-8.4712,89,1800
328825100,gc7x9r5,-102.3,2100.8,51.8976,-8.4723,234,800
328825100,gc7x9pm,-105.1,2800.3,51.8945,-8.4698,67,800
```

#### Geohash Precision Reference

| Precision | Cell Size | Use Case |
|-----------|-----------|----------|
| 5 | ~4.9km × 4.9km | Regional overview |
| 6 | ~1.2km × 0.6km | Urban planning |
| **7** | **~153m × 153m** | **Standard (required)** |
| 8 | ~38m × 19m | Dense urban |

---

### Input File 2: GIS Data

**Filename:** `gis.csv` (configurable)

#### Purpose

Contains physical and configuration parameters for each cell site in the network.

#### Data Sources

| Source | Description |
|--------|-------------|
| Network Inventory | Asset management database |
| OSS (Operations Support System) | Configuration management |
| RF Planning Tools | Network design outputs |

#### Required Schema

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `CellName` | String | Cell identifier | Must match grid data |
| `Latitude` | Float | Site latitude (WGS84) | Range: -90 to +90 |
| `Longitude` | Float | Site longitude (WGS84) | Range: -180 to +180 |
| `Bearing` | Float | Antenna azimuth (°) | Range: 0 to 360 |

#### Optional Schema (Recommended)

| Column | Data Type | Description | Default |
|--------|-----------|-------------|---------|
| `SiteName` | String | Site identifier | Derived from CellName |
| `MechanicalTilt` | Float | Physical tilt (°) | 0 |
| `ElectricalTilt` | Float | Electrical tilt (°) | 0 |
| `Height` | Float | Antenna height AGL (m) | - |
| `Band` | Integer | Frequency band (MHz) | - |
| `Technology` | String | LTE, NR, etc. | - |
| `TxPower` | Float | Transmit power (dBm) | - |
| `Bandwidth` | Float | Channel bandwidth (MHz) | - |

#### Example

```csv
CellName,SiteName,Latitude,Longitude,Bearing,MechanicalTilt,ElectricalTilt,Height,Band
CK089L1,CK089,51.8932,-8.4567,120.0,2.0,4.0,25.0,1800
CK089L2,CK089,51.8932,-8.4567,240.0,2.0,3.0,25.0,800
CK089L3,CK089,51.8932,-8.4567,0.0,2.0,5.0,25.0,2100
CK090L1,CK090,51.9012,-8.4234,90.0,3.0,2.0,30.0,1800
```

#### Bearing Reference

```
                    N (0°/360°)
                         │
                         │
         NW (315°)       │       NE (45°)
                    ╲    │    ╱
                      ╲  │  ╱
                        ╲│╱
       W (270°) ─────────┼───────── E (90°)
                        ╱│╲
                      ╱  │  ╲
                    ╱    │    ╲
         SW (225°)       │       SE (135°)
                         │
                         │
                    S (180°)
```

#### Tilt Calculation

**Total Tilt = Mechanical Tilt + Electrical Tilt**

| Mechanical | Electrical | Total | Effect |
|------------|------------|-------|--------|
| 2° | 4° | 6° | Standard suburban |
| 3° | 6° | 9° | Dense urban |
| 1° | 2° | 3° | Rural long-range |

---

### Input File 3: Hull Data

**Filename:** `cell_hulls.csv` (configurable)

#### Purpose

Contains coverage boundary polygons for each cell, representing the geographic area where the cell provides service.

#### Data Sources

| Source | Method |
|--------|--------|
| Generated | Convex/concave hull from grid data |
| RF Planning | Propagation model outputs |
| Coverage Predictions | Planning tool exports |

#### Required Schema

| Column | Data Type | Description | Constraints |
|--------|-----------|-------------|-------------|
| `cell_id` | String | Cell identifier | Must match GIS data |
| `geometry` | String | Polygon boundary (WKT) | Valid WKT POLYGON |

#### Optional Schema

| Column | Data Type | Description |
|--------|-----------|-------------|
| `cell_name` | String | Cell display name |
| `band` | Integer | Frequency band (MHz) |
| `area_km2` | Float | Coverage area (km²) |
| `n_grids` | Integer | Grid count in hull |

#### Example

```csv
cell_id,cell_name,geometry,band,area_km2
CK089L1,CK089L1,"POLYGON((-8.48 51.88, -8.45 51.91, -8.42 51.89, -8.44 51.86, -8.48 51.88))",1800,2.54
CK089L2,CK089L2,"POLYGON((-8.50 51.87, -8.46 51.92, -8.41 51.88, -8.45 51.84, -8.50 51.87))",800,4.21
```

#### WKT (Well-Known Text) Format

```
POLYGON((
  lon1 lat1,    ← First point
  lon2 lat2,    ← Second point
  lon3 lat3,    ← Third point
  ...
  lon1 lat1     ← Close polygon (repeat first point)
))
```

**Coordinate Order:** Longitude, Latitude (opposite of typical lat/lon convention)

---

### Input File 4: Boundary Shapefile

**Directory:** `county_bounds/` (configurable)

#### Purpose

Defines the analysis region boundary. Used to:
- Clip coverage gap analysis to service area
- Prevent false positives at network edges
- Focus analysis on specific geographic regions

#### Required Files

| File | Purpose |
|------|---------|
| `bounds.shp` | Geometry data |
| `bounds.shx` | Shape index |
| `bounds.dbf` | Attribute data |
| `bounds.prj` | Projection definition |

#### Requirements

| Attribute | Requirement |
|-----------|-------------|
| Geometry Type | Polygon or MultiPolygon |
| Coordinate System | WGS84 (EPSG:4326) |
| Coverage | Must fully contain analysis area |

---

## Output Data Specifications

### Output File 1: Overshooting Results

**Filename:** `overshooting_cells_environment_aware.csv`

#### Purpose

Identifies cells transmitting beyond optimal range, causing interference in distant grids.

#### Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `cell_id` | String | Cell identifier | `CK089L1` |
| `cell_name` | String | Cell display name | `CK089L1` |
| `cilac` | Integer | Numeric cell identifier | `328576779` |
| `environment` | String | Classification | `urban`, `suburban`, `rural` |
| `intersite_distance_km` | Float | Distance to nearest site (km) | `1.2` |
| `latitude` | Float | Cell latitude | `51.8932` |
| `longitude` | Float | Cell longitude | `-8.4567` |
| `azimuth_deg` | Float | Antenna bearing (°) | `120.0` |
| `current_tilt` | Float | Total tilt (°) | `6.0` |
| `band` | Integer | Frequency band (MHz) | `1800` |
| `overshooting_grids` | Integer | Count of overshooting grids | `45` |
| `total_grids` | Integer | Total grids served | `375` |
| `percentage_overshooting` | Float | Proportion overshooting | `0.12` |
| `tier_1_grids` | Integer | Critical (>2km beyond) | `8` |
| `tier_2_grids` | Integer | Moderate (1-2km beyond) | `15` |
| `tier_3_grids` | Integer | Minor (<1km beyond) | `22` |
| `max_overshoot_distance` | Float | Maximum overshoot (m) | `3200` |
| `edge_traffic_events` | Integer | Traffic in edge zone | `1250` |
| `total_traffic_events` | Integer | Total cell traffic | `8500` |
| `avg_competing_cells` | Float | Mean competing cells | `5.2` |
| `severity_score` | Float | Severity (0-1) | `0.75` |
| `severity_category` | String | Severity label | `High` |
| `recommended_tilt_change` | Float | Suggested downtilt (°) | `2.0` |
| `estimated_coverage_reduction_pct` | Float | Expected coverage change | `-15.0` |

#### Severity Categories

| Category | Score Range | Description |
|----------|-------------|-------------|
| Critical | 0.85 - 1.0 | Immediate action required |
| High | 0.65 - 0.84 | Priority remediation |
| Medium | 0.45 - 0.64 | Scheduled optimization |
| Low | 0.0 - 0.44 | Monitor only |

---

### Output File 2: Undershooting Results

**Filename:** `undershooting_cells_environment_aware.csv`

#### Purpose

Identifies cells not reaching full coverage potential, with neighboring cells filling gaps.

#### Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `cell_id` | String | Cell identifier | `CK089L2` |
| `cell_name` | String | Cell display name | `CK089L2` |
| `cilac` | Integer | Numeric cell identifier | `328825100` |
| `environment` | String | Classification | `rural` |
| `max_distance_m` | Float | Current maximum reach (m) | `2500` |
| `edge_traffic_pct` | Float | Traffic at coverage edge | `0.18` |
| `edge_interference_pct` | Float | Interference at edge | `0.55` |
| `competing_cells` | String | Comma-separated list | `CK090L1,CK091L1` |
| `severity_score` | Float | Severity (0-1) | `0.65` |
| `severity_category` | String | Severity label | `High` |
| `recommended_uptilt` | Float | Suggested uptilt (°) | `1.5` |
| `predicted_distance_gain_m` | Float | Expected range increase (m) | `400` |
| `predicted_coverage_increase_pct` | Float | Expected coverage change | `+25.0` |

---

### Output File 3: Daily Resolution Recommendations

**Filename:** `daily_overshooter_resolution_recommendations.csv`

#### Purpose

Unified export combining overshooting and undershooting recommendations in NMS-compatible format for automated tilt change workflows.

#### Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `analysisdate` | Date | Analysis date (ISO 8601) | `2025-11-27` |
| `cell_name` | String | Human-readable name | `CK162L1` |
| `cilac` | Integer | Numeric identifier | `328576779` |
| `parameter` | String | Tilt parameter to modify | `Electrical_tilt` |
| `category` | String | Recommendation type | `overshooter` |
| `parameter_new_value` | Float | Recommended tilt value (°) | `6.0` |
| `cycle_start_date` | Date | Cycle start | `2025-11-27` |
| `cycle_end_date` | Date | Cycle end (if closed) | `` |
| `cycle_status` | String | Workflow status | `PENDING` |
| `conditions` | String | Special conditions | `AUTO_GENERATED` |
| `current_tilt` | Float | Current total tilt (°) | `4.0` |
| `min_tilt` | Float | Minimum allowed (°) | `0.0` |
| `max_tilt` | Float | Maximum allowed (°) | `15.0` |
| `tier_3_sectors_count` | Integer | Affected same-band cells | `12` |
| `tier_3_cells_count` | Integer | Same as sectors_count | `12` |
| `tier_3_traffic_total` | Integer | Traffic in problematic grids | `1840` |
| `tier_3_drops_total` | Integer | Drops in problematic grids | `45` |
| `tier3_traffic_perc` | Float | Traffic % in problem grids | `19.62` |
| `tier3_drops_perc` | Float | Drops % in problem grids | `8.5` |

#### Parameter Selection Logic

```
┌────────────────────┐
│ Recommendation     │
│ Type               │
└─────────┬──────────┘
          │
          ▼
    ┌─────────────┐
    │ Overshooter │──────────────────────► Electrical_tilt
    │ (downtilt)  │
    └─────────────┘
          │
          │ NO
          ▼
    ┌─────────────┐     ┌──────────────┐
    │ Undershooter│────►│ E-tilt = 0? │
    │ (uptilt)    │     └──────┬───────┘
    └─────────────┘            │
                          YES  │  NO
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Manual_tilt           Electrical_tilt
```

#### Tier 3 Metrics Explained

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `tier_3_sectors_count` | Distinct same-band cells (excluding recommendation cell) serving overshooting grids | Count of unique cells |
| `tier_3_traffic_total` | Offending cell's traffic in problematic grids | Sum of event_count |
| `tier3_traffic_perc` | Traffic concentration in problem area | `(tier_3_traffic / total_traffic) × 100` |

#### Workflow States

| Status | Description | Next Action |
|--------|-------------|-------------|
| `PENDING` | Awaiting review | Engineer approval |
| `APPROVED` | Approved for implementation | Apply change |
| `APPLIED` | Change implemented | Monitor results |
| `REJECTED` | Not approved | Document reason |
| `EXPIRED` | Cycle ended without action | Generate new |

---

### Output File 4: Environment Classification

**Filename:** `cell_environment.csv`

#### Purpose

Classification of all cells by deployment environment based on inter-site distance.

#### Schema

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `cell_id` | String | Cell identifier | `CK089L1` |
| `environment` | String | Classification | `urban` |
| `intersite_distance_km` | Float | Distance to nearest site (km) | `0.8` |

#### Classification Criteria

| Environment | ISD Range | Typical Deployment |
|-------------|-----------|-------------------|
| `urban` | < 1.0 km | City centers, commercial districts |
| `suburban` | 1.0 - 3.0 km | Residential, light industrial |
| `rural` | > 3.0 km | Countryside, highways |

---

### Output File 5: Coverage Gap Results

#### No Coverage Clusters

**Filename:** `no_coverage_clusters.geojson`

Geographic areas with no signal from any cell.

**GeoJSON Schema:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "cluster_id": 1,
        "n_points": 25,
        "area_km2": 0.45,
        "centroid_lat": 51.8765,
        "centroid_lon": -8.4321
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-8.43, 51.87], [-8.42, 51.88], ...]]
      }
    }
  ]
}
```

**Properties:**

| Property | Data Type | Description |
|----------|-----------|-------------|
| `cluster_id` | Integer | Unique cluster identifier |
| `n_points` | Integer | Geohash count in cluster |
| `area_km2` | Float | Cluster area (km²) |
| `centroid_lat` | Float | Cluster center latitude |
| `centroid_lon` | Float | Cluster center longitude |

#### Low Coverage Clusters

**Filename Pattern:** `low_coverage_band_{BAND}.geojson`

Example: `low_coverage_band_800.geojson`, `low_coverage_band_1800.geojson`

**Additional Properties:**

| Property | Data Type | Description |
|----------|-----------|-------------|
| `band` | Integer | Frequency band (MHz) |
| `avg_rsrp` | Float | Mean RSRP in cluster (dBm) |
| `min_rsrp` | Float | Minimum RSRP in cluster (dBm) |

---

### Output File 6: Interactive Dashboard

**Filename:** `maps/enhanced_dashboard.html`

#### Purpose

Self-contained interactive map visualization combining all analysis results.

#### Features

| Layer | Symbol | Description |
|-------|--------|-------------|
| Overshooting Cells | Red marker | Cells with severity info |
| Undershooting Cells | Blue marker | Cells with severity info |
| No Coverage | Yellow polygon | Complete coverage gaps |
| Low Coverage | Orange polygon | Weak signal areas (per band) |
| All Cells | Grey triangle | Network cell locations |
| Coverage Hulls | Colored boundary | Per-band coverage boundaries |

#### Browser Requirements

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 80+ |
| Firefox | 75+ |
| Safari | 13+ |
| Edge | 80+ |

---

## Data Validation

### Automatic Validation Checks

| Check | Validation | Error Level |
|-------|------------|-------------|
| Required columns | Column exists in dataset | ERROR |
| Data types | Correct type conversion | ERROR |
| Value ranges | Within expected bounds | WARNING |
| Completeness | Null value percentage | WARNING |
| Referential integrity | Cell IDs match between files | ERROR |

### Value Range Validation

| Field | Valid Range | Unit |
|-------|-------------|------|
| `rsrp_mean` | -140 to -40 | dBm |
| `distance_m` | 0 to 100,000 | meters |
| `Latitude` | -90 to +90 | degrees |
| `Longitude` | -180 to +180 | degrees |
| `Bearing` | 0 to 360 | degrees |
| `MechanicalTilt` | 0 to 15 | degrees |
| `ElectricalTilt` | 0 to 15 | degrees |

### Completeness Thresholds

| Level | Null % | Action |
|-------|--------|--------|
| OK | < 5% | None |
| WARNING | 5-10% | Log warning |
| ERROR | > 10% | Halt processing |

---

## Data Type Reference

### Cell ID Matching

The tool automatically handles cell ID format variations:

| Grid Data | GIS Data | Match Method |
|-----------|----------|--------------|
| `cilac` (Integer) | `CellName` (String) | Integer → String conversion |
| `cell_id` (String) | `CellName` (String) | Direct string match |

### Coordinate Systems

All geographic data must use **WGS84 (EPSG:4326)**:

| Coordinate | Range | Convention |
|------------|-------|------------|
| Latitude | -90 to +90 | Positive = North |
| Longitude | -180 to +180 | Positive = East |

### Geohash Reference

| Character | Precision (m) | Total Chars |
|-----------|---------------|-------------|
| 1 | 5,000,000 × 5,000,000 | 1 |
| 2 | 1,250,000 × 625,000 | 2 |
| 3 | 156,000 × 156,000 | 3 |
| 4 | 39,000 × 19,500 | 4 |
| 5 | 4,900 × 4,900 | 5 |
| 6 | 1,200 × 610 | 6 |
| **7** | **153 × 153** | **7** |
| 8 | 38 × 19 | 8 |

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'cilac'` | Missing cell ID column | Add `cilac` or `cell_id` column |
| `ValueError: Invalid RSRP` | RSRP out of range | Verify values are -140 to -40 |
| `TypeError: Cannot convert` | Wrong data type | Ensure numeric columns contain numbers |
| `No matching cells` | Cell IDs don't match | Verify ID format consistency |
| `Empty results` | No cells meet criteria | Review threshold parameters |

### Data Quality Checklist

- [ ] All required columns present
- [ ] Cell IDs match between grid and GIS data
- [ ] Coordinates in WGS84 (EPSG:4326)
- [ ] Geohashes are 7 characters
- [ ] RSRP values in valid range (-140 to -40)
- [ ] No significant null values (< 5%)
- [ ] Hull geometries are valid WKT polygons

---

## Related Documentation

- [[CONFIGURATION]] — Parameter configuration guide
- [[ALGORITHMS]] — Algorithm technical specifications
- [[API_REFERENCE]] — Programmatic data loading
