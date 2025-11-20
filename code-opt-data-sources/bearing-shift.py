from config import GIS_PATH, OUTPUT_PATH
from utils import rotate_hull_around_pivot, save_file
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads
from shapely import wkt
import shapely.geometry
from shapely.geometry import Polygon
import time
import ast

def main():
	'''
	This program takes 2 inputs:
		- 'cell_name' list (all need to have the same target bearing)
		- 'new_bearing'
	The program will pull in the cells location, current bearing and the convex-hull of the cell, recalculating the convex hull with the new bearing around the cell location (the fulcrum)

	Output datastes:
		- New Hull

		- Grids within the hull with associated RSRP metrics
	'''
	while True:
		cell_list_str = input("Please enter a list of cells in the below format:\n['cell_1', 'cell_2', 'cell_3']\n")
		try:
			cell_list = ast.literal_eval(cell_list_str)  # safe parse
			if not isinstance(cell_list, list) or not all(isinstance(x, str) for x in cell_list):
				raise ValueError
			print("Parsed list:", cell_list)
			break
		except (SyntaxError, ValueError):
			print("Invalid format..")

	while True:
		new_bearing = float(input("Please enter new bearing between 0 and 360:"))
		if ((new_bearing >= 0) and (new_bearing <= 360)):
			break
		else:
			print("Invalid value")

	start = time.perf_counter()
	##########################
	### Load Cell GIS data ###
	##########################
	try:
		print("Load Cell GIS data..")
		gis_df = pd.read_csv(GIS_PATH / "gis.csv", names = ['Name','CILAC', 'Latitude', 'Longitude','Bearing'])
		print("\tCell GIS data Loaded..\n\tReduce to required cells..")
		gis_df = gis_df[gis_df['Name'].isin(cell_list)].reset_index(drop=True)
		current_bearing = gis_df['Bearing'].iloc[0]
	except Exception as e:
		print(f"Processing Cell GIS failed: {e}")

	######################
	### Load Hull data ###
	######################
	try:
		print("Load Hull data..")
		hull_df = pd.read_csv(OUTPUT_PATH / "cell_hulls.csv")
		hull_df = hull_df[hull_df['cell_name'].isin(cell_list)].reset_index(drop=True)

	except Exception as e:
		print(f"Loading Hull data failed: {e}")

	
	bearing_shift = (new_bearing % 360) - (current_bearing % 360)

	rotated_hull = rotate_hull_around_pivot(hull_gdf, \
                                        gis_df[gis_df['Name'] == cell_list[0]]['Latitude'].iloc[0], \
                                        gis_df[gis_df['Name'] == cell_list[0]]['Longitude'].iloc[0], \
                                        -bearing_shift)

	######################################
	### Save 'rotated_hull' to file ###
	######################################
	try:
		print(f"Saving dataframe '{cell_list[0]}_rotated_hull' to file..")
		save_file(rotated_hull, f"{cell_list[0]}_rotated_hull", OUTPUT_PATH, "csv")
	except Exception as e:
		print(f"Saving dataframe 'rotated_hull' to file failed: {e}")
	
	end = time.perf_counter()
	print(f"Time takein to run: {end - start:.3f} seconds")
if __name__ == "__main__":
    main()