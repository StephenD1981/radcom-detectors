from config import INPUT_PATH_DISH, GIS_PATH_DISH, OUTPUT_PATH_DISH, INPUT_PATH_VF, GIS_PATH_VF, OUTPUT_PATH_VF
from utils import polysTriangle, clean_clamp_metrics, create_grid_geo_df, create_convex_hulls, geohash7_inside_hulls_fast, clean_hull_hashes_gdf, predict_grid_rsrp_wgs84_same_cell_only, calculate_distances, save_file, \
					calc_cell_dist_metrics, predict_rsrp_existing_bins_vec, create_tilt_files, build_new_grids, build_extended_grid, build_extended_hulls_grid, add_required_columns
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
	This program creates required data sets from available Grid-Cell data to be used by multiple optimization features. 
	Input datasets:
		- Grid-Cell data: Provides all relevant Grid-Cell info created by thegeolocation process
		- Cell GIS info (enrichment data)

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
		INPUT_PATH  = INPUT_PATH_DISH
		GIS_PATH    = GIS_PATH_DISH
		OUTPUT_PATH = OUTPUT_PATH_DISH
		# Files
		input_grid_file = "bins_enrichment_dn.csv"
		gis_file = "gis.csv"
	elif operator == "2":
		INPUT_PATH  = INPUT_PATH_VF
		GIS_PATH    = GIS_PATH_VF
		OUTPUT_PATH = OUTPUT_PATH_VF
		# Files
		input_grid_file = "grid-cell-data-150m.csv"
		gis_file = "cork-gis.csv"
	

	#################
	### Variables ###
	#################
	extended_cell_calc = "hull" # or end_dist
	start = time.perf_counter()
	##########################
	### Load Cell GIS data ###
	##########################
	try:
		print("Load Cell GIS data..")
		if operator == "1":
			gis_df = pd.read_csv(GIS_PATH / f"{gis_file}", names = ['Name','CILAC', 'SectorID', 'RNC_BSC', 'LAC', 'SectorType', 'Scr_Freq', \
																'UARFCN', 'BSIC', 'Tech', 'Latitude', 'Longitude','Bearing','AvgNeighborDist', \
																'MaxNeighborDist', 'NeighborsCount', 'Eng', 'TiltE','TiltM', 'SiteID', \
																'AdminCellState', 'Asset', 'Asset_Configuration', 'Cell_Type', 'Cell_Name', \
																'City', 'Height', 'RF_Team', 'Asset_Calc', 'Sector_uniq', 'FreqType', 'TAC', \
																'RAC', 'Band', 'Vendor', 'CPICHPwr', 'MaxTransPwr', 'FreqMHz', 'HBW', \
																'VBW', 'Antenna'])
		elif operator == "2":
			gis_df = pd.read_csv(GIS_PATH / f"{gis_file}")
			gis_df = gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', 'Latitude', 'Longitude', 'Bearing', 'CILAC', \
															'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']]

		print("\tCell GIS data Loaded..\n\tCreate Cell Polygons..")
		#gis_df = gis_df[(gis_df.Name == "CK002H1") | (gis_df.Name == "CK002H2")].copy() ## REMOVE!
		gis_df['geometry'] = gis_df.apply(lambda x: polysTriangle(x['Latitude'], x['Longitude'], x['Bearing'], x['HBW']), axis = 1)
		gis_df['geometry'] = gis_df['geometry'].apply(lambda x: shapely.wkt.loads(x))
		gis_df = gpd.GeoDataFrame(gis_df, geometry='geometry', crs='EPSG:4326') 
		print(f"\tCell Polygons created for {gis_df.shape[0]} cells..")
	except Exception as e:
		print(f"Processing Cell GIS failed: {e}")

	###########################
	### Load Grid-Cell data ###
	###########################
	try:
		print("Load Grid-Cell data..")
		grid_data = pd.read_csv(INPUT_PATH / f"{input_grid_file}")
		#grid_data = grid_data[(grid_data.cell_name == "CK002H1") | (grid_data.cell_name == "CK002H2")].copy() ## REMOVE!
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
		if operator == "1":
			grid_geo_data = grid_geo_data.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		elif operator == "2":
			grid_geo_data = grid_geo_data.merge(gis_df[['Name', 'Latitude', 'Longitude', 'RF_Team']], left_on = 'cell_name', right_on = 'Name', how = 'inner')

		grid_geo_data.drop(columns = ['Name'], inplace = True)
		del grid_data
	except Exception as e:
		print(f"Transform to geodataframe failed: {e}")

	########################################
	### Create pontential coverage hulls ###
	########################################
	try:
		print("Create pontential coverage hulls..")
		hulls = create_convex_hulls(grid_geo_data, gis_df)
		print(f"Saving 'hulls' to file..")
		hulls_copy = hulls.copy()
		hulls_copy = hulls_copy.to_crs(4326)
		hulls_copy.to_csv(OUTPUT_PATH / f"cell_hulls.csv", index = False)
		del hulls_copy
	except Exception as e:
		print(f"Creating pontential coverage hulls failed: {e}")

	#############################################
	### Find grids for pontial coverage hulls ###
	#############################################
	try:
		print("Create a dataframe of geohashes residing within the hulls..")
		potential_coverage_start = time.perf_counter()
		#hull_hashes = geohash7_inside_hulls_fast(hulls.head(100), precision=7, geometry_mode="cell", simplify_tolerance_m=10) #### REVOME HEAD!
		hull_hashes = geohash7_inside_hulls_fast(hulls, precision=7, geometry_mode="cell", simplify_tolerance_m=10)
		potential_coverage_end = time.perf_counter()
		print(f"\tTime takein to run: {potential_coverage_end - potential_coverage_start:.3f} seconds")
	except Exception as e:
		print(f"Creating a dataframe of geohashes residing within the hulls failed: {e}")

	#################################################
	### Transform 'hull_hashes' 'hull_hashes_gdf' ###
	#################################################
	try:
		print("\tTransform 'hull_hashes' to geodataframe 'hull_hashes_gdf'..")
		hull_hashes_gdf = create_grid_geo_df(hull_hashes)
		del hull_hashes
		print("\tClean 'hull_hashes_gdf'..")
		hull_hashes_gdf = clean_hull_hashes_gdf(hull_hashes_gdf, grid_geo_data, gis_df)
		#print(hull_hashes_gdf[~hull_hashes_gdf.avg_rsrp.isna()].head())
	except Exception as e:
		print(f"Transform 'hull_hashes' to geodataframe 'hull_hashes_gdf' failed: {e}")

	########################################################
	### Predict RSRP based on surrounding grids for cell ###
	########################################################
	try:
		print("\tPredict RSRP based on surrounding grids for cell")
		hull_hashes_gdf = predict_grid_rsrp_wgs84_same_cell_only(hull_hashes_gdf)
	except Exception as e:
		print(f"Predict RSRP based on surrounding grids for cell failed: {e}")

	######################################################################
	### Add columns 'distance_to_cell' and 'cell_max_distance_to_cell' ###
	######################################################################
	try:
		print("\tAdd columns 'distance_to_cell' and 'cell_max_distance_to_cell'..")
		hull_hashes_gdf = calculate_distances(hull_hashes_gdf)
	except Exception as e:
		print(f"Add columns 'distance_to_cell' and 'cell_max_distance_to_cell' failed: {e}")

	############################################
	### Merge 'hull_hashes_gdf' and 'gis_df' ###
	############################################
	try:
		print("\tMerge 'hull_hashes_gdf' and 'gis_df'..")
		hull_hashes_gdf = hull_hashes_gdf.merge(gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', \
														'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		hull_hashes_gdf.drop(columns = ['Name'], inplace = True)
	except Exception as e:
		print(f"Merge 'hull_hashes_gdf' and 'gis_df' failed: {e}")

	
	#########################################################################################
	### Add additional columns;                                                           ###
	###  - 'perc_grid_events': We will estimate this based on 'avg_rsrp' of cells in grid ###
	###  - 'avg_rsrp_grid': Sum ('perc_grid_events' * 'avg_rsrp')                         ###
	###  - 'avg_rsrp_cell': Avg('avg_rsrp') of the cell                                   ###
	###  - 'grid_max_distance_to_cell': Standard calculation                              ###
	###  - 'grid_min_distance_to_cell': Standard calculation                              ###
	###  - 'perc_cell_max_dist': Standard calculation                                     ###
	###  - 'cell_angle_to_grid': Standard calculation                                     ###
	###  - 'grid_bearing_diff': Standard calculation                                      ###
	###  - 'cell_count': Standard calculation                                             ###
	###  - 'same_pci_cell_count': Standard calculation                                    ###
	### 'hull_hashes_gdf' to file                                                         ###
	#########################################################################################
	hull_hashes_gdf = add_required_columns(hull_hashes_gdf)

	######################################
	### Save 'hull_hashes_gdf' to file ###
	######################################
	try:
		print(f"Saving dataframe 'hull_hashes_gdf' to file..")
		hull_hashes_gdf_copy = hull_hashes_gdf.copy()
		save_file(hull_hashes_gdf_copy, "cell_coverage_complete", OUTPUT_PATH, "csv")
		del hull_hashes_gdf_copy
	except Exception as e:
		print(f"Saving dataframe {hull_hashes_gdf} to file failed: {e}")
	
	##########################################################
	### Delete newly created columns for 'hull_hashes_gdf' ###
	##########################################################
	hull_hashes_gdf.drop(columns = ['perc_grid_events', 'avg_rsrp_grid', 'avg_rsrp_cell', 'grid_max_distance_to_cell', 'grid_min_distance_to_cell', \
									'perc_cell_max_dist', 'cell_angle_to_grid', 'grid_bearing_diff', 'cell_count', 'same_pci_cell_count'], inplace = True)
	

	####################################################################################
	### Create a dataframe containing one row for each valid cell containing columns ###
	### 'cell_name', 'max_dist_1_dt', 'perc_dist_reduct_1_dt', 'max_dist_2_dt',      ###
	### 'perc_dist_reduct_2_dt', 'max_dist_1_ut', 'perc_dist_inc_1_ut',              ###
	### 'max_dist_2_ut', 'perc_dist_inc_2_ut'                                        ###
	####################################################################################
	try:
		print("Create dataframe 'cell_dist_metrics' containing max distance with applied tilts..")
		cell_dist_metrics = calc_cell_dist_metrics(grid_geo_data)
		cell_dist_metrics.to_csv(OUTPUT_PATH / f"cell_distance_metrics.csv", index = False)
	except Exception as e:
		print(f"Create dataframe 'cell_dist_metrics' failed: {e}")

	############################################################################
	### Merge 'cell_dist_metrics' with 'grid_geo_data' and 'hull_hashes_gdf' ###
	############################################################################
	try:
		print("Merge 'cell_dist_metrics' with 'grid_geo_data' and 'hull_hashes_gdf'")
		grid_geo_data = grid_geo_data.merge(cell_dist_metrics[['cell_name', 'max_dist_1_dt', 'perc_dist_reduct_1_dt', 'max_dist_2_dt', 'perc_dist_reduct_2_dt', 'max_dist_1_ut', \
															'perc_dist_inc_1_ut', 'max_dist_2_ut', 'perc_dist_inc_2_ut']], on = "cell_name", how = "left")

		hull_hashes_gdf = hull_hashes_gdf.merge(cell_dist_metrics[['cell_name', 'max_dist_1_dt', 'perc_dist_reduct_1_dt', 'max_dist_2_dt', 'perc_dist_reduct_2_dt', 'max_dist_1_ut', \
															'perc_dist_inc_1_ut', 'max_dist_2_ut', 'perc_dist_inc_2_ut']], on = "cell_name", how = "left")

	except Exception as e:
		print(f"Merge 'cell_dist_metrics' with 'grid_geo_data' and 'hull_hashes_gdf' failed: {e}")

	#####################################################################################
	### Get RSRP values for 'grid_geo_data' and 'hull_hashes_gdf' when tilt is appled ###
	#####################################################################################
	try:
		print("Get RSRP values for 'grid_geo_data' and 'hull_hashes_gdf' when tilt is appled")
		grid_geo_data = predict_rsrp_existing_bins_vec(grid_geo_data)
		hull_hashes_gdf = predict_rsrp_existing_bins_vec(hull_hashes_gdf)
	except Exception as e:
		print(f"Get RSRP values for 'grid_geo_data' and 'hull_hashes_gdf' when tilt is appled failed: {e}")

	##################################################################
	### Create and save output files with 1 and 2 degrees downtilt ###
	##################################################################
	#try:
	print("Create and save output files with 1 and 2 degrees downtilt..")
	print(f"\t'grid_geo_data_1_dt' pre removing na RSRP rows =  {grid_geo_data.shape[0]}")
	grid_geo_data_1_dt = create_tilt_files(grid_geo_data, 'max_dist_1_dt', 'avg_rsrp_1_degree_downtilt')
	print(f"\t'grid_geo_data_1_dt' post removing na RSRP rows = {grid_geo_data_1_dt.shape[0]}")
	print("\tAdd additional columns")
	grid_geo_data_1_dt = add_required_columns(grid_geo_data_1_dt)
	
	print(f"\tSave 'grid_geo_data_1_dt' to file 'cell_coverage_1_degree_dt'..\n")
	save_file(grid_geo_data_1_dt, "cell_coverage_1_degree_dt", OUTPUT_PATH, "csv")
	
	print(f"\t'grid_geo_data_2_dt' pre removing na RSRP rows =  {grid_geo_data.shape[0]}")
	grid_geo_data_2_dt = create_tilt_files(grid_geo_data, 'max_dist_2_dt', 'avg_rsrp_2_degree_downtilt')
	print(f"\t'grid_geo_data_2_dt' post removing na RSRP rows = {grid_geo_data_2_dt.shape[0]}")
	print(f"\tSave 'grid_geo_data_2_dt' to file 'cell_coverage_2_degree_dt'..\n")
	
	print("\tAdd additional columns")
	grid_geo_data_2_dt = add_required_columns(grid_geo_data_2_dt)
	
	save_file(grid_geo_data_2_dt, "cell_coverage_2_degree_dt", OUTPUT_PATH, "csv")

	print(f"\t'hull_hashes_gdf_1_dt' pre removing na RSRP rows =  {hull_hashes_gdf.shape[0]}")
	hull_hashes_gdf_1_dt = create_tilt_files(hull_hashes_gdf, 'max_dist_1_dt', 'avg_rsrp_1_degree_downtilt')
	print(f"\t'hull_hashes_gdf_1_dt' post removing na RSRP rows = {hull_hashes_gdf_1_dt.shape[0]}")
	print(f"\tSave 'hull_hashes_gdf_1_dt' to file 'cell_coverage_complete_1_degree_dt'..\n")
	
	print("\tAdd additional columns")
	hull_hashes_gdf_1_dt = add_required_columns(hull_hashes_gdf_1_dt)

	save_file(hull_hashes_gdf_1_dt, "cell_coverage_complete_1_degree_dt", OUTPUT_PATH, "csv")

	print(f"\t'hull_hashes_gdf_2_dt' pre removing na RSRP rows =   {hull_hashes_gdf.shape[0]}")
	hull_hashes_gdf_2_dt = create_tilt_files(hull_hashes_gdf, 'max_dist_2_dt', 'avg_rsrp_2_degree_downtilt')
	print(f"\t'hull_hashes_gdf_2_dt' post removing na RSRP rows = {hull_hashes_gdf_2_dt.shape[0]}")
	print(f"\tSave 'hull_hashes_gdf_2_dt' to file 'cell_coverage_complete_2_degree_dt'..\n")
	
	print("\tAdd additional columns")
	hull_hashes_gdf_2_dt = add_required_columns(hull_hashes_gdf_2_dt)
	save_file(hull_hashes_gdf_2_dt, "cell_coverage_complete_2_degree_dt", OUTPUT_PATH, "csv")

	# delete unrequired dataframes
	del grid_geo_data_1_dt
	del grid_geo_data_2_dt
	del hull_hashes_gdf_1_dt
	del hull_hashes_gdf_2_dt

	#except Exception as e:
	#	print(f"Create and save output files with 1 and 2 degrees downtilt failed: {e}")

	############################################################
	### Build extended grid files for 1 and 2 degrees uptilt ###
	############################################################
	try:
		print("Build extended grids for uptilt..")
		if extended_cell_calc == "hull":
			extended_grid_1_ut_gdf, extended_grid_2_ut_gdf = build_extended_hulls_grid(cell_dist_metrics, gis_df, hulls)
		else:
			extended_grid_1_ut_gdf, extended_grid_2_ut_gdf = build_extended_grid(cell_dist_metrics, gis_df)
	except Exception as e:
		print(f"Build extended grids for uptilt failed: {e}")

	#######################################################
	### Create output files with 1 and 2 degrees uptilt ###
	#######################################################
	try:
		print("Create files with 1 and 2 degrees uptilt..")
		print(f"\tCreate 'grid_geo_data_1_ut'")
		grid_geo_data_1_ut = create_tilt_files(grid_geo_data, 'max_dist_1_ut', 'avg_rsrp_1_degree_uptilt')
		
		print(f"\tCreate 'grid_geo_data_2_ut'")
		grid_geo_data_2_ut = create_tilt_files(grid_geo_data, 'max_dist_2_ut', 'avg_rsrp_2_degree_uptilt')

		print(f"\tCreate 'hull_hashes_gdf_1_ut'")
		hull_hashes_gdf_1_ut = create_tilt_files(hull_hashes_gdf, 'max_dist_1_ut', 'avg_rsrp_1_degree_uptilt')
		print(f"\tCreate 'hull_hashes_gdf_2_ut'")
		hull_hashes_gdf_2_ut = create_tilt_files(hull_hashes_gdf, 'max_dist_2_ut', 'avg_rsrp_2_degree_uptilt')

	except Exception as e:
		print(f"Create files with 1 and 2 degrees uptilt failed: {e}")

	#######################################################################################################################
	### Concat 'grid_geo_data_1_ut' and 'grid_geo_data_2_ut' with 'extended_grid_1_ut_gdf' and 'extended_grid_2_ut_gdf' ###
	#######################################################################################################################		
	try:
		print("Concat uptilt dataframes with extended coverage grids..")
		if extended_cell_calc == "hull":
			mask = ~extended_grid_1_ut_gdf['grid_cell'].isin(grid_geo_data_1_ut['grid_cell'])
			extended_filtered = extended_grid_1_ut_gdf.loc[mask]
			grid_geo_data_1_ut = pd.concat([grid_geo_data_1_ut, extended_filtered], ignore_index=True)
			hull_hashes_gdf_1_ut = pd.concat([hull_hashes_gdf_1_ut, extended_filtered], ignore_index=True)

			mask = ~extended_grid_2_ut_gdf['grid_cell'].isin(grid_geo_data_2_ut['grid_cell'])
			extended_filtered = extended_grid_2_ut_gdf.loc[mask]
			grid_geo_data_2_ut = pd.concat([grid_geo_data_2_ut, extended_filtered], ignore_index=True)
			hull_hashes_gdf_2_ut = pd.concat([hull_hashes_gdf_2_ut, extended_filtered], ignore_index=True)

		else:
			grid_geo_data_1_ut = pd.concat([grid_geo_data_1_ut, extended_grid_1_ut_gdf], ignore_index=True)
			grid_geo_data_2_ut = pd.concat([grid_geo_data_2_ut, extended_grid_2_ut_gdf], ignore_index=True)

			hull_hashes_gdf_1_ut = pd.concat([hull_hashes_gdf_1_ut, extended_grid_1_ut_gdf], ignore_index=True)
			hull_hashes_gdf_2_ut = pd.concat([hull_hashes_gdf_2_ut, extended_grid_2_ut_gdf], ignore_index=True)
		
	except Exception as e:
		print(f"Concat uptilt dataframes with extended coverage grids failed: {e}")

	#######################################################################################################################################################
	### Predict RSRP for 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' based on surrounding grids for cell ###
	#######################################################################################################################################################
	try:
		print("\tPredict RSRP for 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' based on surrounding grids for cell")
		#grid_geo_data_1_ut = grid_geo_data_1_ut.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		grid_geo_data_1_ut = predict_grid_rsrp_wgs84_same_cell_only(grid_geo_data_1_ut)
		
		#grid_geo_data_2_ut = grid_geo_data_2_ut.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		grid_geo_data_2_ut = predict_grid_rsrp_wgs84_same_cell_only(grid_geo_data_2_ut)

		#hull_hashes_gdf_1_ut = hull_hashes_gdf_1_ut.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		hull_hashes_gdf_1_ut = predict_grid_rsrp_wgs84_same_cell_only(hull_hashes_gdf_1_ut)
		
		#hull_hashes_gdf_2_ut = hull_hashes_gdf_2_ut.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		hull_hashes_gdf_2_ut = predict_grid_rsrp_wgs84_same_cell_only(hull_hashes_gdf_2_ut)
	except Exception as e:
		print(f"Predict RSRP for 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' based on surrounding grids for cell failed: {e}")

	######################################################################
	### Add columns 'distance_to_cell' and 'cell_max_distance_to_cell' ###
	######################################################################
	try:
		print("\tAdd columns 'distance_to_cell' and 'cell_max_distance_to_cell'..")
		grid_geo_data_1_ut = calculate_distances(grid_geo_data_1_ut)
		grid_geo_data_2_ut = calculate_distances(grid_geo_data_2_ut)

		hull_hashes_gdf_1_ut = calculate_distances(hull_hashes_gdf_1_ut)
		hull_hashes_gdf_2_ut = calculate_distances(hull_hashes_gdf_2_ut)
	except Exception as e:
		print(f"Add columns 'distance_to_cell' and 'cell_max_distance_to_cell' failed: {e}")

	#####################################################################################################################
	### Merge 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' and 'gis_df' ###
	#####################################################################################################################
	#try:
	#	print("\tMerge 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' and 'gis_df'..")
	#	grid_geo_data_1_ut = grid_geo_data_1_ut.merge(gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', \
	#													'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
	#	grid_geo_data_1_ut.drop(columns = ['Name'], inplace = True)

	#	grid_geo_data_2_ut = grid_geo_data_2_ut.merge(gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', \
	#													'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
	#	grid_geo_data_2_ut.drop(columns = ['Name'], inplace = True)

	#	hull_hashes_gdf_1_ut = hull_hashes_gdf_1_ut.merge(gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', \
	#													'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
	#	hull_hashes_gdf_1_ut.drop(columns = ['Name'], inplace = True)

	#	hull_hashes_gdf_2_ut = hull_hashes_gdf_2_ut.merge(gis_df[['Name', 'Band', 'City', 'FreqMHz', 'HBW', 'Height', 'MaxTransPwr', 'RF_Team', \
	#													'Scr_Freq', 'SiteID', 'TAC', 'TiltE', 'TiltM', 'UARFCN', 'Vendor']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
	#	hull_hashes_gdf_2_ut.drop(columns = ['Name'], inplace = True)
	
	#except Exception as e:
	#	print(f"Merge 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' and 'gis_df' failed: {e}")

	###############################################################################################################
	### Save 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' to file ###
	###############################################################################################################
	try:
		print(f"Saving dataframes 'grid_geo_data_1_ut', 'grid_geo_data_2_ut', 'hull_hashes_gdf_1_ut', 'hull_hashes_gdf_2_ut' to file..")
		
		#print("\tAdd additional columns to 'grid_geo_data_1_dt'")
		#grid_geo_data_1_ut = add_required_columns(grid_geo_data_1_ut)
		#print("\tSave 'grid_geo_data_1_dt'")
		#save_file(grid_geo_data_1_ut, "cell_coverage_1_degree_ut", OUTPUT_PATH, "csv")
		#del grid_geo_data_1_ut
		
		#print("\tAdd additional columns to 'grid_geo_data_2_ut'")
		#grid_geo_data_2_ut = add_required_columns(grid_geo_data_2_ut)
		#print("\tSave 'grid_geo_data_2_ut'")
		#save_file(grid_geo_data_2_ut, "cell_coverage_2_degree_ut", OUTPUT_PATH, "csv")
		#del grid_geo_data_2_ut

		print("\tAdd additional columns to 'hull_hashes_gdf_1_ut'")
		hull_hashes_gdf_1_ut = add_required_columns(hull_hashes_gdf_1_ut)
		print("\tSave 'hull_hashes_gdf_1_ut'")
		save_file(hull_hashes_gdf_1_ut, "cell_coverage_complete_1_degree_ut", OUTPUT_PATH, "csv")
		del hull_hashes_gdf_1_ut

		print("\tAdd additional columns to 'hull_hashes_gdf_2_ut'")
		hull_hashes_gdf_2_ut = add_required_columns(hull_hashes_gdf_2_ut)
		print("\tSave 'hull_hashes_gdf_2_ut'")
		save_file(hull_hashes_gdf_2_ut, "cell_coverage_complete_2_degree_ut", OUTPUT_PATH, "csv")
		del hull_hashes_gdf_2_ut


		print("\tSave 'grid_geo_data'")
		save_file(grid_geo_data, "cell_coverage", OUTPUT_PATH, "csv")

	except Exception as e:
		print(f"Saving dataframes to file failed: {e}")
		

	end = time.perf_counter()
	print(f"Time takein to run: {end - start:.3f} seconds")
if __name__ == "__main__":
    main()