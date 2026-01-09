# Data Ingestion & Field Standardisation

This document describes the data ingestion process for the RAN Optimizer, including source files, column mappings, and validation requirements.

## Overview

The RAN Optimizer requires three input datasets plus a boundary file to generate recommendations for overshooting, undershooting, and coverage gap detection.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT DATA FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  cell_coverage.csv ──┐                                              │
│                      │                                              │
│  cork-gis.csv ───────┼──► VodafoneIrelandAdapter ──► Standardised  │
│                      │         (adapters.py)          DataFrames    │
│  cell_hulls.csv ─────┘                                              │
│                                                                     │
│  county_bounds/ ─────────► GeoDataFrame (boundary filtering)        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Source Files

### Location

```
data/vf-ie/input-data/
├── cell_coverage.csv      # Grid-level RF measurements (1.7GB)
├── cell_hulls.csv         # Cell coverage polygons (1.2MB)
├── cork-gis.csv           # Cell site/antenna configuration (900KB)
└── county_bounds/
    ├── bounds.shp         # Cork county boundary polygon
    ├── bounds.dbf
    ├── bounds.prj
    ├── bounds.shx
    └── bounds.cpg
```

### Data Relationships

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  cell_coverage   │     │    cork-gis      │     │   cell_hulls     │
│                  │     │                  │     │                  │
│  cilac (FK) ─────┼─────┼─► CILAC (PK)  ◄──┼─────┼── cilac (FK)     │
│  grid            │     │  Name            │     │  cell_name       │
│  cell_name       │     │  SiteID          │     │  geometry        │
│  avg_rsrp        │     │  Latitude        │     │  area_km2        │
│  avg_rsrq        │     │  Longitude       │     │                  │
│  ...             │     │  Bearing         │     │                  │
│                  │     │  TiltE, TiltM    │     │                  │
│                  │     │  ...             │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘

Join Key: cilac/CILAC (9-digit cell identifier, e.g., 328167169)
```

---

## File Specifications

### 1. cell_coverage.csv

Grid-level RF measurements enriched with cell information. Each row represents a unique (grid, cell) combination.

**Record Count**: ~8.5M rows
**Unique Cells**: 1,660
**Unique Grids**: ~500K geohash7 bins

#### Source Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `grid_cell` | string | Composite key (grid_cilac) | `gc1zpnu_328167169` |
| `grid` | string | 7-character geohash | `gc1zpnu` |
| `cell_name` | string | Human-readable cell name | `CK002H1` |
| `cilac` | integer | 9-digit cell identifier | `328167169` |
| `avg_rsrp` | float | Average RSRP (dBm) | `-107.0` |
| `avg_rsrq` | float | Average RSRQ (dB) | `-15.0` |
| `avg_sinr` | float | Average SINR (dB) | `13.0` |
| `event_count` | integer | Number of measurements | `1` |
| `grid_event_count` | integer | Total events in grid | `1630` |
| `cell_event_count` | integer | Total events for cell | `3633` |
| `distance_to_cell` | float | Distance to serving cell (m) | `3087.8` |
| `cell_angle_to_grid` | float | Bearing from cell to grid (°) | `181.6` |
| `perc_cell_events` | float | % of cell's total events | `0.000275` |
| `perc_grid_events` | float | % of grid's total events | `0.000613` |

#### Pre-computed Tilt Impact Columns

| Column | Type | Description |
|--------|------|-------------|
| `max_dist_1_dt` | float | Max distance after 1° downtilt |
| `perc_dist_reduct_1_dt` | float | % distance reduction (1° DT) |
| `max_dist_2_dt` | float | Max distance after 2° downtilt |
| `perc_dist_reduct_2_dt` | float | % distance reduction (2° DT) |
| `max_dist_1_ut` | float | Max distance after 1° uptilt |
| `perc_dist_inc_1_ut` | float | % distance increase (1° UT) |
| `max_dist_2_ut` | float | Max distance after 2° uptilt |
| `perc_dist_inc_2_ut` | float | % distance increase (2° UT) |
| `avg_rsrp_1_degree_downtilt` | float | Estimated RSRP after 1° DT |
| `avg_rsrp_2_degree_downtilt` | float | Estimated RSRP after 2° DT |
| `avg_rsrp_1_degree_uptilt` | float | Estimated RSRP after 1° UT |
| `avg_rsrp_2_degree_uptilt` | float | Estimated RSRP after 2° UT |

---

### 2. cork-gis.csv

Cell site and antenna configuration data from the network inventory system.

**Record Count**: 1,919 cells
**Encoding**: UTF-8 with BOM (`\ufeff`)

#### Source Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Name` | string | Cell name | `CK002H1` |
| `CILAC` | integer | 9-digit cell identifier | `328167169` |
| `SectorID` | integer | Sector identifier | `28167169` |
| `SiteID` | string | Site identifier (lat_lon format) | `51.921_-8.47447` |
| `Latitude` | float | Cell latitude (WGS84) | `51.92099042` |
| `Longitude` | float | Cell longitude (WGS84) | `-8.474466964` |
| `Bearing` | integer | Antenna azimuth (degrees) | `130` |
| `TiltE` | float | Electrical tilt (degrees) | `2.0` |
| `TiltM` | float | Mechanical tilt (degrees) | `0` |
| `Height` | float | Antenna height AGL (meters) | `27.8` |
| `Tech` | string | Technology type | `LTE_Ericsson` |
| `Band` | string | Frequency band | `L1800` |
| `FreqMHz` | integer | Frequency (MHz) | `1800` |
| `HBW` | integer | Bandwidth (MHz) | `20` |
| `MaxTransPwr` | integer | Max TX power (dBm) | `60` |
| `Scr_Freq` | integer | Physical Cell ID (PCI) | `453` |
| `AdminCellState` | integer | On-air status (1=active) | `1` |
| `Vendor` | string | Equipment vendor | `Ericsson` |
| `TAC` | integer | Tracking Area Code | `51001` |

---

### 3. cell_hulls.csv

Pre-computed coverage hull polygons for each cell.

**Record Count**: 1,660 cells

#### Source Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `cell_name` | string | Cell name | `CK002H1` |
| `cilac` | integer | 9-digit cell identifier | `328167169` |
| `geometry` | string | WKT polygon | `POLYGON ((-8.49... 51.90...))` |
| `area_km2` | float | Hull area (km²) | `17.757` |

---

### 4. county_bounds/bounds.shp

Cork county boundary shapefile for spatial filtering.

**Format**: ESRI Shapefile
**CRS**: EPSG:4326 (WGS84)
**Bounds**: `[-10.248, 51.420, -7.841, 52.388]`

---

## Field Standardisation

The `VodafoneIrelandAdapter` class (`ran_optimizer/data/adapters.py`) maps source columns to standardised schema fields.

### Grid Column Mappings

```python
GRID_COLUMN_MAP = {
    'geohash7': 'grid',                    # grid -> geohash7
    'cell_id': 'cilac',                    # cilac -> cell_id
    'cell_name': 'cell_name',              # cell_name -> cell_name
    'rsrp': 'avg_rsrp',                    # avg_rsrp -> rsrp
    'rsrq': 'avg_rsrq',                    # avg_rsrq -> rsrq
    'sinr': 'avg_sinr',                    # avg_sinr -> sinr
    'total_traffic': 'event_count',        # event_count -> total_traffic
    'distance_m': 'distance_to_cell',      # distance_to_cell -> distance_m
    'bearing_deg': 'cell_angle_to_grid',   # cell_angle_to_grid -> bearing_deg
}
```

### GIS Column Mappings

```python
GIS_COLUMN_MAP = {
    'cell_id': 'CILAC',                    # CILAC -> cell_id
    'cell_name': 'Name',                   # Name -> cell_name
    'site_name': 'SiteID',                 # SiteID -> site_name
    'sector_id': 'SectorID',               # SectorID -> sector_id
    'cell_pci': 'Scr_Freq',                # Scr_Freq -> cell_pci
    'latitude': 'Latitude',                # Latitude -> latitude
    'longitude': 'Longitude',              # Longitude -> longitude
    'azimuth_deg': 'Bearing',              # Bearing -> azimuth_deg
    'mechanical_tilt': 'TiltM',            # TiltM -> mechanical_tilt
    'electrical_tilt': 'TiltE',            # TiltE -> electrical_tilt
    'height_m': 'Height',                  # Height -> height_m
    'on_air': 'AdminCellState',            # AdminCellState -> on_air
    'technology': 'Tech',                  # Tech -> technology
    'frequency_mhz': 'FreqMHz',            # FreqMHz -> frequency_mhz
    'bandwidth_mhz': 'HBW',                # HBW -> bandwidth_mhz
    'tx_power_dbm': 'MaxTransPwr',         # MaxTransPwr -> tx_power_dbm
    'band': 'Band',                        # Band -> band
    'vendor': 'Vendor',                    # Vendor -> vendor
}
```

### Hull Column Mappings

```python
HULL_COLUMN_MAP = {
    'cell_id': 'cilac',                    # cilac -> cell_id
    'cell_name': 'cell_name',              # cell_name -> cell_name
    'geometry': 'geometry',                # geometry -> geometry
    'area_km2': 'area_km2',                # area_km2 -> area_km2
}
```

---

## Data Transformations

The adapter applies the following transformations after column renaming:

### Grid Data Transformations

```python
def _transform_grid_data(df):
    # Convert cell_id to string
    df['cell_id'] = df['cell_id'].astype(str)

    # Set cell_pci placeholder (not available in VF data)
    df['cell_pci'] = 0

    return df
```

### GIS Data Transformations

```python
def _transform_gis_data(df):
    # Convert identifiers to strings
    df['cell_id'] = df['cell_id'].astype(str)
    df['sector_id'] = df['sector_id'].astype(str)
    df['site_name'] = df['site_name'].astype(str)

    # Convert on_air to boolean (1 = True, 0 = False)
    df['on_air'] = df['on_air'] == 1

    # Fill missing tilt values with 0
    df['electrical_tilt'] = df['electrical_tilt'].fillna(0.0)
    df['mechanical_tilt'] = df['mechanical_tilt'].fillna(0.0)

    return df
```

### Hull Data Transformations

```python
def _transform_hull_data(df):
    # Convert cell_id to string
    df['cell_id'] = df['cell_id'].astype(str)

    return df
```

---

## Validation Requirements

### Join Key Validation

All three datasets must share the same `cell_id` (cilac) format:

| Requirement | Specification |
|-------------|---------------|
| Format | 9-digit integer (as string after transformation) |
| Example | `328167169` |
| Validation | All coverage/hull cilacs must exist in GIS |

### Current Data Statistics

| Metric | Value |
|--------|-------|
| Cells in coverage data | 1,660 |
| Cells in hull data | 1,660 |
| Cells in GIS data | 1,919 |
| Coverage → GIS match rate | 100% |
| Hull → GIS match rate | 100% |
| GIS cells without coverage | 259 (off-air or outside boundary) |

### RF Metric Ranges

| Field | Valid Range | Unit |
|-------|-------------|------|
| `rsrp` | -140 to -30 | dBm |
| `rsrq` | -40 to 0 | dB |
| `sinr` | -20 to 40 | dB |
| `cell_pci` | 0 to 503 | - |
| `azimuth_deg` | 0 to 360 | degrees |
| `mechanical_tilt` | -30 to 30 | degrees |
| `electrical_tilt` | -30 to 30 | degrees |
| `height_m` | 0 to 200 | meters |

---

## Usage Example

```python
import pandas as pd
from ran_optimizer.data.adapters import get_adapter

# Get the Vodafone Ireland adapter
adapter = get_adapter('Vodafone_Ireland')

# Load and adapt grid data
grid_raw = pd.read_csv('data/vf-ie/input-data/cell_coverage.csv')
grid_df = adapter.adapt_grid_data(grid_raw)

# Load and adapt GIS data
gis_raw = pd.read_csv('data/vf-ie/input-data/cork-gis.csv')
gis_df = adapter.adapt_gis_data(gis_raw)

# Load and adapt hull data
hull_raw = pd.read_csv('data/vf-ie/input-data/cell_hulls.csv')
hull_df = adapter.adapt_hull_data(hull_raw)

# Join datasets on standardised cell_id
merged = grid_df.merge(gis_df, on='cell_id', suffixes=('', '_gis'))
```

---

## Future: PostgreSQL Migration

When migrating to PostgreSQL, the same adapter pattern will apply:

```python
# Future database loader
from ran_optimizer.data.loaders import load_from_postgres

grid_df = load_from_postgres(
    table='cell_coverage',
    adapter='Vodafone_Ireland',
    connection_string='postgresql://...'
)
```

The adapter mappings will translate database column names to the standardised schema, maintaining consistency across data sources.

---

## Changelog

| Date | Change |
|------|--------|
| 2024-11-26 | Fixed `cilac` format mismatch in cell_coverage.csv (added leading `3`) |
| 2024-11-26 | Added `cilac` column to cell_hulls.csv |
| 2024-11-26 | Updated VodafoneIrelandAdapter with correct column mappings |
| 2024-11-26 | Added HULL_COLUMN_MAP and adapt_hull_data() method |
| 2024-11-26 | Added frequency_mhz, bandwidth_mhz, tx_power_dbm to GIS mappings |
