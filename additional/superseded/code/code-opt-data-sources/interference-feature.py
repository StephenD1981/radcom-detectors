from config import INPUT_PATH_DISH, INPUT_GRID_PATH_DISH, OUTPUT_INTERFERENCE_PATH_DISH, INPUT_PATH_VF, INPUT_GRID_PATH_VF, OUTPUT_INTERFERENCE_PATH_VF, min_filtered_cells_per_grid, \
					min_cell_event_count, perc_grid_events, dominant_perc_grid_events, max_rsrp_diff, grid_ring, perc_interference, dominance_diff
from utils import clean_clamp_metrics, create_grid_geo_df, find_interference_cells
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads
from shapely import wkt
import shapely.geometry
from shapely.geometry import Polygon
import time

def main():
	'''
	This is an optimization feature with the goal of finding network interference based on geospatial data. 
	Input datasets:
		- Grid-Cell data: Provides all relevant Grid-Cell info created by the geolocation process
		- Grid-Cell data (perceived coverage): Provides perceived Grid-Cell info created from 'Grid-Cell data'
		- Grid-Cell data (perceived coverage - 1 degree downtilt): Provides perceived Grid-Cell info with 1 degree towntilt applied
		- Grid-Cell data (perceived coverage - 1 degree downtilt): Provides perceived Grid-Cell info with 1 degree towntilt applied

	Output datastes:
		- Hull-projection: Shows the potential coverage of cells to grids with current tilt settings

		- Grid-Cell-1-degree-down-tilt: Estimated coverage grids and RSRP if 1 degree down-tilt is applied
		- Grid-Cell-2-degree-down-tilt: Estimated coverage grids and RSRP if 2 degree down-tilt is applied
		- Grid-Cell-1-degree-up-tilt: Estimated coverage grids and RSRP if 1 degree up-tilt is applied
		- Grid-Cell-2-degree-up-tilt: Estimated coverage grids and RSRP if 2 degree up-tilt is applied

		- Hull-projection-1-degree-down-tilt: Estimated coverage grids and RSRP if 1 degree down-tilt is applied
		- Hull-projection-2-degree-down-tilt: Estimated coverage grids and RSRP if 2 degree down-tilt is applied
		- Hull-projection-1-degree-up-tilt: Estimated coverage grids and RSRP if 1 degree up-tilt is applied
		- Hull-projection-2-degree-up-tilt: Estimated coverage grids and RSRP if 2 degree up-tilt is applied
	'''

	while True:
		operator = input("Please select an operator:\n\t1. DISH\n\t2. VODAFONE\nPlease select (e.g. 1 or 2): ")
		if (operator == "1") or (operator == "2"):
			break
		else:
			print("Please select a valid operator number..\n")
	
	if operator == "1":
		INPUT_PATH               = INPUT_PATH_DISH
		INPUT_GRID_PATH          = INPUT_GRID_PATH_DISH
		OUTPUT_INTERFERENCE_PATH = OUTPUT_INTERFERENCE_PATH_DISH
	elif operator == "2":
		INPUT_PATH               = INPUT_PATH_VF
		INPUT_GRID_PATH          = INPUT_GRID_PATH_VF
		OUTPUT_INTERFERENCE_PATH = OUTPUT_INTERFERENCE_PATH_VF

	###########################
	### Load Grid-Cell data ###
	###########################
	while True:
		clustering_algo = input("Please select fixed or floating clustering approach;\n\t1. Fixed\n\t2. Floating\nType 1 or 2 now..")
		if clustering_algo == "1":
			clustering_algo = "fixed"
			break
		elif clustering_algo == "2":
			clustering_algo = "dynamic-sklearn"
			break
		else:
			print("Incorrect input, please swelet 1 or 2..\n")
	try:
		print("Load Grid-Cell data..")
		grid_data = pd.read_csv(INPUT_PATH / "bins_enrichment_dn.csv")
		print("\tClean Grid-Cell data..")
		grid_data = clean_clamp_metrics(grid_data)

	except Exception as e:
		print(f"Processing Grid-Cell failed: {e}")

	#############################################
	### Transform 'grid_data' 'grid_geo_data' ###
	#############################################
	try:
		print("\tTransform to geodataframe..")
		grid_geo_data = create_grid_geo_df(grid_data)
		# Add grid_count
		grid_count = grid_geo_data.groupby('cell_name')['grid'].transform('count')
		grid_geo_data['grid_count'] = grid_count
		print(f"\t There are {len(list(set(grid_geo_data.cell_name.to_list())))} cells in this dataset..")

		del grid_data
	except Exception as e:
		print(f"Transform to geodataframe failed: {e}")

	#####################################
	### Load Potential Grid-Cell data ###
	#####################################
	try:
		print("Load Potential Grid-Cell data..")
		grid_data_perceived = pd.read_csv(INPUT_GRID_PATH / "cell_coverage_complete.csv")

	except Exception as e:
		print(f"Processing Potential Grid-Cell failed: {e}")

	#################################################################
	### Transform 'grid_data_perceived' 'grid_geo_data_perceived' ###
	#################################################################
	try:
		print("\tTransform to geodataframe..")
		grid_geo_data_perceived = create_grid_geo_df(grid_data_perceived)
		# Add grid_count
		grid_count = grid_geo_data_perceived.groupby('cell_name')['grid'].transform('count')
		grid_geo_data_perceived['grid_count'] = grid_count
		del grid_data_perceived
	except Exception as e:
		print(f"Transform to geodataframe failed: {e}")
	
	#######################################################
	### Load Potential Grid-Cell 1 degree Downtilt data ###
	#######################################################
	'''try:
		print("Load Potential Grid-Cell 1 degree Downtilt data..")
		grid_data_perceived_1_dt = pd.read_csv(INPUT_GRID_PATH / "cell_coverage_complete_1_degree_dt.csv")

	except Exception as e:
		print(f"Processing Potential Grid-Cell 1 degree Downtilt failed: {e}")

	###########################################################################
	### Transform 'grid_data_perceived_1_dt' 'grid_geo_data_perceived_1_dt' ###
	###########################################################################
	try:
		print("\tTransform to geodataframe..")
		grid_geo_data_perceived_1_dt = create_grid_geo_df(grid_data_perceived_1_dt)
		# Add grid_count
		grid_count = grid_geo_data_perceived_1_dt.groupby('cell_name')['grid'].transform('count')
		grid_geo_data_perceived_1_dt['grid_count'] = grid_count
		del grid_data_perceived_1_dt
	except Exception as e:
		print(f"Transform to geodataframe failed: {e}")

	#######################################################
	### Load Potential Grid-Cell 2 degree Downtilt data ###
	#######################################################
	try:
		print("Load Potential Grid-Cell 2 degree Downtilt data..")
		grid_data_perceived_2_dt = pd.read_csv(INPUT_GRID_PATH / "cell_coverage_complete_2_degree_dt.csv")

	except Exception as e:
		print(f"Processing Potential Grid-Cell 2 degree Downtilt failed: {e}")

	###########################################################################
	### Transform 'grid_data_perceived_2_dt' 'grid_geo_data_perceived_2_dt' ###
	###########################################################################
	try:
		print("\tTransform to geodataframe..")
		grid_geo_data_perceived_2_dt = create_grid_geo_df(grid_data_perceived_2_dt)
		# Add grid_count
		grid_count = grid_geo_data_perceived_2_dt.groupby('cell_name')['grid'].transform('count')
		grid_geo_data_perceived_2_dt['grid_count'] = grid_count
		del grid_data_perceived_2_dt
	except Exception as e:
		print(f"Transform to geodataframe failed: {e}")'''

	#############################################
	### Calcuate interference cells and grids ###
	#############################################

	# Calculate interference cells and grids for 'grid_geo_data'
	interference_cell_list, interference_grids = find_interference_cells(grid_geo_data, min_filtered_cells_per_grid, min_cell_event_count, \
																			perc_grid_events, dominant_perc_grid_events, dominance_diff, max_rsrp_diff, grid_ring, \
																			perc_interference, 'actual', clustering_algo)
	####################
	### Save to File ###
	####################
	interference_cell_list.to_csv(OUTPUT_INTERFERENCE_PATH / f"interference-cell-list.csv", index = False)
	interference_grids.to_csv(OUTPUT_INTERFERENCE_PATH / f"interference-grid-cell.csv", index = False)

	#######################################################
	### Calcuate perceived interference cells and grids ###
	#######################################################
	# Find 95th Percentile 'cell_count' for 'grid_geo_data_perceived'
	# -> We want to tackle interference gradually, always targeting the worst 5% of interference grids
	min_cell_grid_count_perceived = grid_geo_data_perceived.groupby('grid')['cell_count'].min().quantile(0.95)
	#print(f"95th percentile 'cell_count' = {min_cell_grid_count_perceived} cells per grid")


	# Calculate interference cells and grids for 'grid_geo_data'
	interference_cell_list_perceived, interference_grids_perceived = find_interference_cells(grid_geo_data_perceived, min_filtered_cells_per_grid, 0, \
																			perc_grid_events, dominant_perc_grid_events, dominance_diff, max_rsrp_diff, grid_ring, \
																			perc_interference, 'perceived', clustering_algo)

	####################
	### Save to File ###
	####################
	interference_cell_list_perceived.to_csv(OUTPUT_INTERFERENCE_PATH / f"interference-cell-list-perceived.csv", index = False)
	interference_grids_perceived.to_csv(OUTPUT_INTERFERENCE_PATH / f"interference-grid-cell-perceived.csv", index = False)



if __name__ == "__main__":
    main()