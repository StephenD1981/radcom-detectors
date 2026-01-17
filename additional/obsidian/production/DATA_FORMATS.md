# Data Format Specification

| Document Information |                                      |
|---------------------|--------------------------------------|
| **Version**         | 3.1                                  |
| **Classification**  | Data Integration Specification       |
| **Last Updated**    | 15 January 2026                      |
| **Audience**        | Data Engineers, System Integrators   |

---

## Table of Contents

1. [Overview](#overview)
2. [Input Data Requirements](#input-data-requirements)
   - [Grid Data](#input-file-1-grid-data)
   - [GIS Data](#input-file-2-gis-data)
   - [Hull Data](#input-file-3-hull-data)
   - [Cell Impacts Data](#input-file-4-cell-impacts-data)
   - [Boundary Shapefile](#input-file-5-boundary-shapefile)
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
| Cell Impacts | CSV | Conditional | Cell-to-cell relationships and traffic (required for CA imbalance, PCI conflicts, crossed feeder detection) |
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
| `avg_rsrp_grid` | Float | Grid-level average RSRP (dBm) | - |
| `avg_rsrq_grid` | Float | Grid-level average RSRQ (dB) | - |
| `avg_sinr_grid` | Float | Grid-level average SINR (dB) | - |
| `grid_event_count` | Integer | Total events in grid | - |
| `avg_rsrp_cell` | Float | Cell-level average RSRP (dBm) | - |
| `avg_rsrq_cell` | Float | Cell-level average RSRQ (dB) | - |
| `avg_sinr_cell` | Float | Cell-level average SINR (dB) | - |
| `cell_event_count` | Integer | Total events for cell | - |
| `perc_cell_events` | Float | % of cell's total events | - |
| `perc_grid_events` | Float | % of grid's total events | - |
| `grid_max_distance_to_cell` | Float | Max distance from grid to any cell (m) | - |
| `grid_min_distance_to_cell` | Float | Min distance from grid to any cell (m) | - |
| `cell_max_distance_to_cell` | Float | Max distance for this cell (m) | - |
| `perc_cell_max_dist` | Float | Distance as % of cell max | - |
| `grid_bearing_diff` | Float | Bearing difference (°) | - |
| `geometry` | String | Grid polygon (WKT) | - |
| `band_cell_count` | Integer | Cells on same band in grid | - |
| `cell_count` | Integer | Total cells in grid | - |
| `grid_cell` | String | Composite key (grid_cilac) | - |

#### Field Categories

**Grid-Level Aggregations** - Metrics aggregated across all cells in a grid:
- `avg_rsrp_grid`, `avg_rsrq_grid`, `avg_sinr_grid` - RF quality metrics averaged over all cells serving the grid
- `grid_event_count` - Total traffic events across all cells in this grid
- `grid_max_distance_to_cell`, `grid_min_distance_to_cell` - Distance range from grid to any serving cell
- `band_cell_count` - Count of cells on same frequency band serving this grid
- `cell_count` - Total count of all cells serving this grid

**Cell-Level Baselines** - Metrics aggregated across all grids served by a cell:
- `avg_rsrp_cell`, `avg_rsrq_cell`, `avg_sinr_cell` - RF quality metrics averaged over all grids the cell serves
- `cell_event_count` - Total traffic events across all grids this cell serves
- `cell_max_distance_to_cell` - Maximum distance this cell reaches to any grid

**Coverage Extent Metrics** - Relationships between cell reach and grid position:
- `perc_cell_max_dist` - This grid's distance as percentage of cell's maximum reach (0.0 to 1.0)
- `perc_cell_events` - This grid's events as percentage of cell's total traffic (used for undershooting detection)
- `perc_grid_events` - This cell's events as percentage of grid's total traffic (used for interference analysis)

**Angular Metrics**:
- `grid_bearing_diff` - Angular difference between cell azimuth and actual bearing to grid (used for overshooting)

**Composite Keys**:
- `grid_cell` - Concatenation of grid and cilac for unique row identification

#### Example

```csv
cilac,grid,rsrp_mean,distance_m,Latitude,Longitude,event_count,band,avg_rsrp_grid,avg_rsrq_grid,avg_sinr_grid,grid_event_count,avg_rsrp_cell,avg_rsrq_cell,avg_sinr_cell,cell_event_count,perc_cell_events,perc_grid_events,cell_max_distance_to_cell,perc_cell_max_dist,grid_bearing_diff
328576779,gc7x9r5,-95.2,1250.5,51.8976,-8.4723,156,1800,-104.8,-14.4,13.6,1630,-102.1,-13.0,15.0,3633,0.000275,0.000613,3424.9,0.902,51.6
328576779,gc7x9r4,-98.7,1450.2,51.8965,-8.4712,89,1800,-106.2,-15.1,12.8,1420,-102.1,-13.0,15.0,3633,0.000245,0.000627,3424.9,0.964,48.2
328825100,gc7x9r5,-102.3,2100.8,51.8976,-8.4723,234,800,-108.5,-16.2,11.4,1630,-99.8,-11.5,16.2,4821,0.000485,0.001435,5680.2,0.740,32.8
328825100,gc7x9pm,-105.1,2800.3,51.8945,-8.4698,67,800,-110.1,-17.0,10.2,890,-99.8,-11.5,16.2,4821,0.000139,0.000753,5680.2,0.928,28.4
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
| `uarfcn` | Integer | UARFCN/EARFCN channel number | - |
| `cell_type` | String | Outdoor/Indoor classification | - |

#### Example

```csv
CellName,SiteName,Latitude,Longitude,Bearing,MechanicalTilt,ElectricalTilt,Height,Band,uarfcn,cell_type
CK089L1,CK089,51.8932,-8.4567,120.0,2.0,4.0,25.0,1800,1450,Outdoor
CK089L2,CK089,51.8932,-8.4567,240.0,2.0,3.0,25.0,800,6300,Outdoor
CK089L3,CK089,51.8932,-8.4567,0.0,2.0,5.0,25.0,2100,9360,Outdoor
CK090L1,CK090,51.9012,-8.4234,90.0,3.0,2.0,30.0,1800,1450,Indoor
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

### Input File 4: Cell Impacts Data

**Filename:** `cell_impacts.csv` (operator-specific)

#### Purpose

Contains cell-to-cell relationship data including traffic handovers, interference patterns, and neighbor relationships. Used for advanced interference detection, carrier aggregation imbalance analysis, PCI conflict detection, and crossed feeder identification.

#### Data Sources

| Source | Description | Update Frequency |
|--------|-------------|------------------|
| OSS/NMS | Cell configuration and relationships | Daily |
| PM Counters | Traffic volumes, handover statistics | Hourly |
| CDR/XDR | Call detail records, event data | Real-time |

#### Complete Schema (63 Fields)

##### Identity & Relationship Fields

| Column | Data Type | Required | Description | Example |
|--------|-----------|----------|-------------|---------|
| `cell_name` | String | Yes | Source cell name | `CK093K3` |
| `cell` | String | Yes | Source cell identifier | `CK093_3` |
| `site` | String | Yes | Source site identifier | `CK093` |
| `cell_impact_name` | String | Yes | Target cell name | `CK652L1` |
| `impact_cell` | String | Yes | Target cell identifier | `CK652_1` |
| `impact_site` | String | Yes | Target site identifier | `CK652` |
| `co_sectored` | String | Yes | Same cell ID? (Y/N) | `N` |
| `co_site` | String | Yes | Same site? (Y/N) | `N` |
| `neighbor_relation` | String | Yes | Defined neighbor? (Y/N/N/A) | `N` |

##### RF Configuration Fields

| Column | Data Type | Required | Description | Example | Valid Range |
|--------|-----------|----------|-------------|---------|-------------|
| `cell_tech` | String | Yes | Source cell technology | `LTE` | LTE, UMTS, GSM |
| `cell_band` | String | Yes | Source frequency band | `L700` | L700, L800, L1800, L2100, U900, G900 |
| `cell_pci` | Integer | Yes | Source Physical Cell ID | `12` | 0-503 (LTE), N/A (2G/3G) |
| `cell_mech_tilt` | Float | Yes | Source mechanical tilt (°) | `0` | 0-15 |
| `cell_elec_tilt` | Float | Yes | Source electrical tilt (°) | `0` | 0-15 |
| `cell_impact_tech` | String | Yes | Target cell technology | `LTE` | LTE, UMTS, GSM |
| `cell_impact_band` | String | Yes | Target frequency band | `L800` | L700, L800, L1800, L2100, U900, G900 |
| `cell_impact_pci` | Integer | Yes | Target Physical Cell ID | `205` | 0-503 (LTE), N/A (2G/3G) |
| `cell_impact_mech_tilt` | Float | Yes | Target mechanical tilt (°) | `0` | 0-15 |
| `cell_impact_elec_tilt` | Float | Yes | Target electrical tilt (°) | `4` | 0-15 |

##### Technology Indicator Fields

| Column | Data Type | Required | Description | Example |
|--------|-----------|----------|-------------|---------|
| `lte_on_cell` | String | Yes | LTE enabled on source? | `Y` |
| `lte_on_impact_cell` | String | Yes | LTE enabled on target? | `Y` |
| `gsm_on_cell` | String | Yes | GSM enabled on source? | `Y` |
| `gsm_on_impact_cell` | String | Yes | GSM enabled on target? | `Y` |
| `umts_on_cell` | String | Yes | UMTS enabled on source? | `Y` |
| `umts_on_impact_cell` | String | Yes | UMTS enabled on target? | `Y` |
| `nsa_on_cell` | String | Yes | NSA (5G) enabled on source? | `N` |
| `nsa_on_impact_cell` | String | Yes | NSA (5G) enabled on target? | `N` |
| `sa_on_cell` | String | Yes | SA (5G) enabled on source? | `N` |
| `sa_on_impact_cell` | String | Yes | SA (5G) enabled on target? | `N` |

##### Traffic & Performance Fields - Data

| Column | Data Type | Required | Description | Example | Unit |
|--------|-----------|----------|-------------|---------|------|
| `traffic_data` | Integer | Yes | Data traffic on this relation | `5` | Events |
| `total_cell_traffic_data` | Integer | Yes | Total data traffic on source cell | `1166` | Events |
| `relation_impact_data_perc` | Float | Yes | Relation data as % of cell total | `0.43` | % |
| `total_cell_traffic_data_lte` | Integer | Yes | LTE data traffic on source | `74` | Events |
| `total_lte_impact_data_perc` | Float | Yes | LTE data as % of cell total | `6.35` | % |
| `total_cell_traffic_data_umts` | Integer | Yes | UMTS data traffic on source | `541` | Events |
| `total_UMTS_impact_data_perc` | Float | Yes | UMTS data as % of cell total | `46.4` | % |
| `total_cell_traffic_data_gsm` | Integer | Yes | GSM data traffic on source | `551` | Events |
| `total_gsm_impact_data_perc` | Float | Yes | GSM data as % of cell total | `47.26` | % |

##### Traffic & Performance Fields - Voice

| Column | Data Type | Required | Description | Example | Unit |
|--------|-----------|----------|-------------|---------|------|
| `traffic_voice` | Integer | Yes | Voice traffic on this relation | `1` | Events |
| `total_cell_traffic_voice` | Integer | Yes | Total voice traffic on source cell | `20` | Events |
| `relation_impact_voice_perc` | Float | Yes | Relation voice as % of cell total | `5` | % |
| `total_cell_traffic_voice_lte` | Integer | Yes | LTE voice traffic on source | `13` | Events |
| `total_LTE_impact_voice_perc` | Float | Yes | LTE voice as % of cell total | `65` | % |
| `total_cell_traffic_voice_umts` | Integer | Yes | UMTS voice traffic on source | `2` | Events |
| `total_umts_impact_voice_perc` | Float | Yes | UMTS voice as % of cell total | `10` | % |
| `total_cell_traffic_voice_gsm` | Integer | Yes | GSM voice traffic on source | `5` | Events |
| `total_gsm_impact_voice_perc` | Float | Yes | GSM voice as % of cell total | `25` | % |

##### Performance Quality Fields

| Column | Data Type | Required | Description | Example | Unit |
|--------|-----------|----------|-------------|---------|------|
| `drops_voice` | Integer | Yes | Voice call drops on relation | `0` | Drops |

##### Geographic Fields

| Column | Data Type | Required | Description | Example | Valid Range |
|--------|-----------|----------|-------------|---------|-------------|
| `cell_lat` | Float | Yes | Source cell latitude | `51.558639` | -90 to +90 |
| `cell_lon` | Float | Yes | Source cell longitude | `-9.540553` | -180 to +180 |
| `cell_impact_lat` | Float | Yes | Target cell latitude | `51.5043459` | -90 to +90 |
| `cell_impact_lon` | Float | Yes | Target cell longitude | `-9.741992441` | -180 to +180 |
| `distance` | Float | Yes | Distance between cells (m) | `15190.47082` | ≥ 0 |

##### Administrative Fields

| Column | Data Type | Required | Description | Example |
|--------|-----------|----------|-------------|---------|
| `market` | String | Yes | Market/region identifier | `Cork` |
| `vendor` | String | Yes | Equipment vendor | `ERICSSON` |

##### 5G SA/NSA Traffic Fields

| Column | Data Type | Required | Description | Example | Unit |
|--------|-----------|----------|-------------|---------|------|
| `total_cell_traffic_data_nsa` | Integer | Yes | NSA (5G) data traffic on source | `0` | Events |
| `total_nsa_impact_data_perc` | Float | Yes | NSA data as % of cell total | `0` | % |
| `total_cell_traffic_voice_nsa` | Integer | Yes | NSA (5G) voice traffic on source | `0` | Events |
| `total_nsa_impact_voice_perc` | Float | Yes | NSA voice as % of cell total | `0` | % |
| `total_cell_traffic_data_sa` | Integer | Yes | SA (5G) data traffic on source | `0` | Events |
| `total_sa_impact_data_perc` | Float | Yes | SA data as % of cell total | `0` | % |
| `total_cell_traffic_voice_sa` | Integer | Yes | SA (5G) voice traffic on source | `0` | Events |
| `total_sa_impact_voice_perc` | Float | Yes | SA voice as % of cell total | `0` | % |

#### Usage by Algorithm

| Algorithm | Key Fields Used | Purpose |
|-----------|----------------|---------|
| **CA Imbalance Detection** | `cell_band`, `cell_impact_band`, `traffic_data`, `total_cell_traffic_data_lte`, `co_site`, `distance` | Identifies unbalanced traffic between carrier bands at same site |
| **PCI Conflict Detection** | `cell_pci`, `cell_impact_pci`, `cell_band`, `cell_impact_band`, `distance`, `co_site` | Detects PCI collisions, mod3/mod30 conflicts within interference range |
| **Crossed Feeder Detection** | `cell_name`, `cell_impact_name`, `bearing`, `distance`, `traffic_data`, `co_site` | Identifies physically swapped antenna connections |
| **Interference Analysis** | `traffic_data`, `traffic_voice`, `drops_voice`, `distance`, `cell_band`, `cell_impact_band` | Quantifies inter-cell interference impact |

#### Data Quality Notes

1. **Technology Indicators**: All technology flags (lte_on_cell, gsm_on_cell, etc.) use Y/N values
2. **Missing PCIs**: GSM and UMTS cells have `N/A` for PCI fields (LTE-only parameter)
3. **Percentage Fields**: All percentage fields are in decimal format (0.43 = 0.43%, not 43%)
4. **Distance Calculations**: All distances use Haversine formula on WGS84 coordinates
5. **Traffic Volumes**: Traffic counts represent aggregated events over collection period (typically 24 hours)

#### Example

```csv
cell_name,cell,site,cell_tech,cell_band,cell_pci,lte_on_cell,cell_mech_tilt,cell_elec_tilt,cell_impact_name,impact_cell,impact_site,cell_impact_tech,cell_impact_band,cell_impact_pci,lte_on_impact_cell,cell_impact_mech_tilt,cell_impact_elec_tilt,co_sectored,co_site,neighbor_relation,traffic_data,total_cell_traffic_data,relation_impact_data_perc,cell_lat,cell_lon,cell_impact_lat,cell_impact_lon,distance,vendor
CK093K3,CK093_3,CK093,LTE,L700,12,Y,0,0,CK652L1,CK652_1,CK652,LTE,L800,205,Y,0,4,N,N,N,5,1166,0.43,51.558639,-9.540553,51.5043459,-9.741992441,15190.47082,ERICSSON
CK366K1,CK366_1,CK366,LTE,L700,354,Y,0,6,CK366S1,CK366_1,CK366,GSM,G900,N/A,Y,0,6,Y,Y,N/A,153,700,21.86,51.91169478,-8.277230183,51.91169478,-8.277230183,0,ERICSSON
```

---

### Input File 5: Boundary Shapefile

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
