# RAN Optimizer - Production Documentation

## What is the RAN Optimizer?

The RAN Optimizer is a software tool that analyzes mobile phone network data to find problems with cell tower antennas.

**Think of it like this:** Imagine cell towers as flashlights pointing at the ground. Some flashlights might be:
- Pointing too far (overshooting) - wasting light where it's not needed
- Not pointing far enough (undershooting) - leaving dark spots
- Missing areas entirely (coverage gaps) - no light at all

This tool finds these problems and tells engineers exactly how to fix them (usually by adjusting the angle of the antenna).

---

## Who Should Use This Tool?

| Role | How They Use It |
|------|-----------------|
| **RAN Engineers** | Run the tool, review recommendations, implement tilt changes |
| **Network Planners** | Analyze coverage gaps, prioritize optimization projects |
| **Data Scientists** | Customize parameters, integrate with data pipelines |
| **Operations Teams** | Schedule automated runs, monitor network health |

---

## What Problems Does It Solve?

### Problem 1: Overshooting Cells
**What it means:** A cell antenna is sending signal too far, interfering with neighboring cells.

**Real-world impact:**
- Users experience dropped calls when their phone switches between cells
- Network capacity is wasted
- Pilot pollution in distant areas

**What the tool does:** Identifies which cells are overshooting and recommends how many degrees to tilt the antenna DOWN.

### Problem 2: Undershooting Cells
**What it means:** A cell antenna isn't reaching far enough, leaving gaps in coverage.

**Real-world impact:**
- Users have weak signal or no signal in certain areas
- Call quality suffers
- Neighboring cells compensate, becoming overloaded

**What the tool does:** Identifies which cells are undershooting and recommends how many degrees to tilt the antenna UP.

### Problem 3: No Coverage Areas
**What it means:** Geographic areas with absolutely no mobile signal from any cell.

**Real-world impact:**
- Users have no service at all in these zones
- Emergency calls impossible

**What the tool does:** Maps these areas on an interactive map so planners can decide if new cell sites are needed.

### Problem 4: Low Coverage Areas
**What it means:** Areas where signal exists but is too weak for reliable service (below -115 dBm).

**Real-world impact:**
- Slow data speeds
- Poor call quality
- Frequent disconnections

**What the tool does:** Maps these weak signal zones by frequency band.

### Problem 5: High Interference Zones
**What it means:** Multiple cells provide similar signal strength in the same area, causing confusion.

**Real-world impact:**
- Poor signal quality (low SINR)
- Handover failures
- Reduced throughput

**What the tool does:** Identifies geographic zones with high pilot pollution where multiple cells compete.

### Problem 6: PCI Conflicts
**What it means:** Cells with identical or conflicting Physical Cell IDs cause measurement ambiguity.

**Real-world impact:**
- Cell confusion during initial access
- Measurement errors
- Failed handovers

**What the tool does:** Detects PCI confusion (same PCI in neighbor list), collisions (coverage overlap), and mod 3/30 conflicts.

### Problem 7: Carrier Aggregation Imbalance
**What it means:** Coverage band and capacity band footprints don't align properly.

**Real-world impact:**
- Users can't aggregate carriers for higher speeds
- Wasted capacity spectrum
- Suboptimal throughput

**What the tool does:** Identifies sites where coverage and capacity bands have mismatched footprints.

### Problem 8: Crossed Feeders
**What it means:** Antenna cables are physically swapped between sectors.

**Real-world impact:**
- Wrong cell serves wrong area
- Severe performance degradation
- Handover issues

**What the tool does:** Detects reciprocal swap patterns by analyzing traffic direction vs. antenna azimuth.

---

## Documentation Index

| Document | What You'll Learn | Who Should Read It |
|----------|-------------------|-------------------|
| [[INSTALLATION]] | How to install the software | Everyone (first step) |
| [[ADDING_NEW_OPERATOR]] | How to add a new mobile operator to the system | Operations Teams, Data Engineers |
| [[CONFIGURATION]] | How to adjust parameters for your network | Engineers, Data Scientists |
| [[ALGORITHMS]] | Exactly how each detection algorithm works | Engineers, Data Scientists |
| [[DATA_FORMATS]] | What data format the tool expects | Data Engineers, Analysts |
| [[DATA_INGESTION]] | How to prepare and transform source data | Data Engineers |
| [[API_REFERENCE]] | How to use the Python code directly | Developers |
| [[DEPLOYMENT]] | How to run this in production | Operations Teams |

---

## Quick Start Guide

### Step 1: Install the Tool

```bash
pip install -e .
```

### Step 2: Prepare Your Data

You need three input files (see [[DATA_FORMATS]] for details):
1. **Grid data** - Measurements from user devices (signal strength, distance)
2. **GIS data** - Cell tower locations and antenna configurations
3. **Hull data** - Coverage area polygons for each cell

### Step 3: Run the Analysis

```bash
# RECOMMENDED: Use --data-dir for simplified usage
ran-optimize --data-dir data/vf-ie

# This automatically uses:
#   - Input from: data/vf-ie/input-data/
#   - Output to: data/vf-ie/output-data/

# For other operators:
ran-optimize --data-dir data/dish
ran-optimize --data-dir data/three-uk

# Run specific algorithms only
ran-optimize --data-dir data/vf-ie --algorithms overshooting undershooting

# Or use explicit paths (legacy method)
ran-optimize --input-dir data/vf-ie/input-data --output-dir data/vf-ie/output-data

# Available algorithms:
# - overshooting
# - undershooting
# - no_coverage (areas with zero coverage)
# - no_coverage_per_band (band-specific coverage gaps)
# - low_coverage (areas with weak signal)
# - interference
# - pci (confusion + collision detection)
# - pci_conflict (hull overlap analysis)
# - ca_imbalance
# - crossed_feeder
```

### Step 4: Review Results

1. Open `data/output/maps/enhanced_dashboard.html` in a web browser
2. Review CSV files with specific recommendations
3. Export to your network management system

---

## What Files Does It Produce?

| Output File | What It Contains | Format |
|-------------|------------------|--------|
| **Coverage Optimization** | | |
| `overshooting_cells_environment_aware.csv` | List of cells to tilt DOWN, with severity scores | CSV |
| `undershooting_cells_environment_aware.csv` | List of cells to tilt UP, with severity scores | CSV |
| `no_coverage_clusters.geojson` | Map polygons showing zero coverage areas | GeoJSON |
| `low_coverage_band_{700/800/1800/etc}.geojson` | Map polygons showing weak signal areas (per band) | GeoJSON |
| **Interference & PCI** | | |
| `interference_clusters.geojson` | Map polygons showing high interference zones | GeoJSON |
| `pci_confusion.csv` | Cells with neighbor PCI ambiguity | CSV |
| `pci_collisions.csv` | Cell pairs with same/conflicting PCIs | CSV |
| `pci_conflict_overlaps.csv` | Hull overlap analysis for PCI conflicts | CSV |
| **Physical Layer** | | |
| `ca_imbalance.csv` | Sites with mismatched CA band footprints | CSV |
| `crossed_feeder_results.csv` | Cells with suspected feeder swaps | CSV |
| `crossed_feeder_swap_pairs.csv` | Reciprocal swap patterns (HIGH confidence) | CSV |
| **Supporting Files** | | |
| `cell_environment.csv` | Classification of each cell (urban/suburban/rural) | CSV |
| `maps/enhanced_dashboard.html` | Interactive map showing all findings | HTML |

---

## Key Concepts Explained

### What is RSRP?
**RSRP (Reference Signal Received Power)** is how strong the cell signal is, measured in dBm (decibel-milliwatts).

| RSRP Value | Signal Quality | User Experience |
|------------|----------------|-----------------|
| -80 dBm or better | Excellent | Fast data, clear calls |
| -80 to -100 dBm | Good | Reliable service |
| -100 to -115 dBm | Fair | May experience slowdowns |
| Worse than -115 dBm | Poor | Likely to have problems |

### What is a Geohash?
A **geohash** is a way to represent a geographic area as a short code. The tool uses 7-character geohashes, which represent squares approximately 153m x 153m.

Example: `gc7x9r5` represents a specific 153m x 153m square somewhere on Earth.

### What is Antenna Tilt?
**Tilt** is the angle of the antenna pointing downward from horizontal.
- **Downtilt** = pointing more toward the ground (reduces coverage distance)
- **Uptilt** = pointing more toward the horizon (increases coverage distance)

The tool recommends tilt changes in degrees (e.g., "increase downtilt by 2°").

### What is Environment Classification?
The tool automatically classifies each cell as:
- **Urban** - Dense city areas with many cells close together
- **Suburban** - Medium density areas
- **Rural** - Sparse areas with cells far apart

This matters because optimal parameters differ by environment type.

---

## System Architecture

```
ran_optimizer/
├── cli.py                 # Command-line interface (how you run it)
├── runner.py              # Main coordinator (orchestrates everything)
├── core/                  # Core utilities
│   ├── environment_classifier.py  # Determines urban/suburban/rural
│   └── geometry.py                # Distance and angle calculations
├── data/                  # Data handling
│   ├── loaders.py         # Reads your input files
│   ├── schemas.py         # Validates data structure
│   └── adapters.py        # Converts between formats
├── recommendations/       # The detection algorithms
│   ├── overshooters.py    # Finds cells transmitting too far
│   ├── undershooters.py   # Finds cells not reaching far enough
│   ├── coverage_gaps.py   # Finds no/low coverage areas
│   └── environment_aware.py  # Adjusts parameters by environment
├── validation/            # Quality checks
│   └── validators.py      # Ensures recommendations make sense
├── visualization/         # Map generation
│   ├── enhanced_map.py    # Creates interactive HTML dashboard
│   └── unified_map.py     # Combines all layers
└── utils/                 # Helper utilities
    ├── config.py          # Reads configuration files
    ├── logging_config.py  # Sets up logging
    └── geohash.py         # Geohash encoding/decoding
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.11+ |
| RAM | 4 GB | 8+ GB (for large datasets) |
| Disk Space | 1 GB | 10+ GB (depends on data size) |
| OS | macOS, Linux, Windows | Any |

---

## Getting Help

- **Installation problems?** See [[INSTALLATION]] troubleshooting section
- **Data format questions?** See [[DATA_FORMATS]]
- **Understanding results?** See [[ALGORITHMS]]
- **Running in production?** See [[DEPLOYMENT]]

---

## Glossary

| Term | Definition |
|------|------------|
| **Cell** | A single antenna sector on a cell tower (towers typically have 3 cells) |
| **Cilac** | Cell ID + Location Area Code - a unique identifier for each cell |
| **dBm** | Decibel-milliwatts - unit for measuring signal strength |
| **GIS** | Geographic Information System - cell location and antenna data |
| **Grid** | A small geographic area (geohash) where measurements are aggregated |
| **Hull** | A polygon boundary around a cell's coverage area |
| **ISD** | Inter-Site Distance - distance between cell tower sites |
| **RSRP** | Reference Signal Received Power - signal strength metric |
| **Tilt** | The downward angle of an antenna from horizontal |

