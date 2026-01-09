from pathlib import Path

# project root = 2 levels up from this file (adjust as needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

###########################
### DISH DATA LOCATIONS ###
###########################
INPUT_PATH_DISH  = DATA_ROOT / "input-data" / "dish" / "grid" / "denver"
GIS_PATH_DISH    = DATA_ROOT / "input-data" / "dish" / "gis"
OUTPUT_PATH_DISH = DATA_ROOT / "output-data" / "dish" / "denver" / "recommendations" / "created_datasets"

INPUT_GRID_PATH_DISH = DATA_ROOT  / "output-data" / "dish" / "denver" / "recommendations" / "created_datasets"
OUTPUT_INTERFERENCE_PATH_DISH = DATA_ROOT / "output-data" / "dish" / "denver" / "recommendations" / "features" / "interference"

###############################
### VODAFONE DATA LOCATIONS ###
###############################
INPUT_PATH_VF  = DATA_ROOT / "output-data" / "vf-ie" / "grid"
GIS_PATH_VF    = DATA_ROOT / "input-data" / "vf-ie" / "gis"
OUTPUT_PATH_VF = DATA_ROOT / "output-data" / "vf-ie" / "recommendations" / "created_datasets"

INPUT_GRID_PATH_VF = DATA_ROOT  / "output-data" / "vf-ie" / "recommendations" / "created_datasets"
OUTPUT_INTERFERENCE_PATH_VF = DATA_ROOT / "output-data" / "vf-ie" /  "recommendations" / "features" / "interference"





########################
### Module Variables ###
########################

# Interference module
min_filtered_cells_per_grid = 3
min_cell_event_count = 25
perc_grid_events = 0.05
dominant_perc_grid_events = 0.3
max_rsrp_diff = 5
grid_ring = 3
perc_interference = 0.33
dominance_diff = 10