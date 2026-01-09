import geopandas as gpd
import pandas as pd
import contextily as cx
import shapely.geometry
import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import loads
from shapely.geometry import LineString, Point
from shapely.geometry import shape
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from shapely.geometry import shape
from pyproj import Transformer, transform
import sys, os
import math
import matplotlib.colors as colors
from polygon_geohasher.polygon_geohasher import geohashes_to_polygon
import geohash
from functools import reduce

def mid(s, offset, amount):
	return s[offset:offset+amount]

def loadInputData(input_path):
	for root, dir, file in os.walk(input_path):
		for filename in file:
			if ("DS_Store" not in filename) & ("SUCCESS" not in filename):
				df = pd.read_csv("{}/{}".format(root, filename))
				break
	return df

def dbm_to_mw(dbm):
	"""Convert dBm to milliwatts."""
	return 10 ** (dbm / 10)

def mw_to_db(mw):
	"""Convert milliwatts to dB."""
	return 10 * math.log10(mw)

def calculate_sinr(rsrp_dbm, rsrq_db, n_rb=50):
	"""
	Calculate SINR from RSRP (dBm) and RSRQ (dB).
	"""
	rsrp_mw = dbm_to_mw(rsrp_dbm)
	rsrq_linear = dbm_to_mw(rsrq_db)

	# Compute RSSI
	rssi_mw = (n_rb * rsrp_mw) / rsrq_linear

	# Compute Interference + Noise (I + N)
	interference_noise_mw = rssi_mw - rsrp_mw

	# Compute SINR in linear scale
	sinr_linear = rsrp_mw / interference_noise_mw

	# Convert to dB
	sinr_db = mw_to_db(sinr_linear)
	return sinr_db

def createGridTranspose(grid_transpose):
	grid_transpose_count_grid = grid_transpose[['grid', 'event_count']].groupby(['grid']).sum().reset_index()
	grid_transpose_count_grid.rename(columns={'event_count': 'grid_event_count'}, inplace=True)

	grid_transpose_count_cell = grid_transpose[['cilac', 'event_count']].groupby(['cilac']).sum().reset_index()
	grid_transpose_count_cell.rename(columns={'event_count': 'cell_event_count'}, inplace=True)

	######################
	### Get Grid means ###
	######################
	grid_transpose_mean_grid = grid_transpose[['grid', 'avg_rsrp', 'avg_rsrq', 'avg_sinr']].groupby(['grid']).mean().reset_index()

	grid_transpose_mean_grid.rename(columns={'avg_rsrp' : 'avg_rsrp_grid', 'avg_rsrq' : 'avg_rsrq_geo_hash', \
	                                         'avg_sinr' : 'avg_sinr_grid'}, inplace=True)

	######################
	### Get Cell means ###
	######################
	grid_transpose_mean_cell = grid_transpose[['cilac', 'avg_rsrp', 'avg_rsrq', 'avg_sinr']].groupby(['cilac']).mean().reset_index()

	grid_transpose_mean_cell.rename(columns={'avg_rsrp' : 'avg_rsrp_cell', 'avg_rsrq' : 'avg_rsrq_cell', \
	                                         'avg_sinr' : 'avg_sinr_cell'}, inplace=True)

	data_frames_1 = [grid_transpose, grid_transpose_count_grid, grid_transpose_mean_grid]
	grid_transpose = reduce(lambda  left,right: pd.merge(left,right,on=['grid'], how='outer'), data_frames_1)
	data_frames_2 = [grid_transpose, grid_transpose_count_cell, grid_transpose_mean_cell]
	grid_transpose = reduce(lambda  left,right: pd.merge(left,right,on=['cilac'], how='outer'), data_frames_2)

	return grid_transpose

def calcPercentage(event_count, agg_event_count):
	try:
		return event_count / agg_event_count
	except:
		return 0

def findDistance(geom, lat_cell, lon_cell):
	'''
	This function returns;
		- Distance from cell to centroid of the bin
		- Percentage of max propagation
		- Weighted distance
		- Angle from cell to centroid of the bin
	'''
	grid_lon = float(str(geom).split(" ")[1].replace("(", ""))
	grid_lat = float(str(geom).split(" ")[2].replace(")", ""))
	R = 6373.0 * 1000

	lat1 = math.radians(lat_cell)
	lon1 = math.radians(lon_cell)
	lat2 = math.radians(grid_lat)
	lon2 = math.radians(grid_lon)

	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	distance = R * c

	angle = math.degrees(math.atan2(grid_lat - lat_cell, (grid_lon) - (lon_cell)))
	if angle <= 0:
		angle = angle + 360

	angle = math.fabs(angle - 450) % 360

	return distance, angle

def getDistances(grid_agg_gdf):
	grid_max_dist = grid_agg_gdf[['grid', 'distance_to_cell']].groupby(['grid']).max().reset_index()
	grid_max_dist.rename(columns={'distance_to_cell': 'grid_max_distance_to_cell'}, inplace=True)

	grid_min_dist = grid_agg_gdf[['grid', 'distance_to_cell']].groupby(['grid']).min().reset_index()
	grid_min_dist.rename(columns={'distance_to_cell': 'grid_min_distance_to_cell'}, inplace=True)

	cell_max_dist = grid_agg_gdf[['cilac', 'distance_to_cell']].groupby(['cilac']).max().reset_index()
	cell_max_dist.rename(columns={'distance_to_cell': 'cell_max_distance_to_cell'}, inplace=True)

	data_frames_1 = [grid_agg_gdf, grid_max_dist, grid_min_dist]
	grid_agg_gdf = reduce(lambda  left,right: pd.merge(left,right,on=['grid'], how='outer'), data_frames_1)

	data_frames_2 = [grid_agg_gdf, cell_max_dist]
	grid_agg_gdf = reduce(lambda  left,right: pd.merge(left,right,on=['cilac'], how='outer'), data_frames_2)

	return grid_agg_gdf

def getBearingDiff(cell_angle_to_grid, bearing):
	grid_bearing_diff = math.fabs(cell_angle_to_grid - bearing)
	if grid_bearing_diff > 180:
		grid_bearing_diff = 360 - grid_bearing_diff
	return grid_bearing_diff

def getFreqBand(freq):
	if freq == "K":
		return 700
	elif freq == "L":
		return 800
	elif freq == "H":
		return 1800
	else:
		return 2100

def main():
	'''
	This pprogram takes standard MEA grid data as an input and creates final grid dataset for optimisation purposes
	It should be noted that the final grid dataset is exactly the same as the grid dataset produced using CM/PM inputs
	'''
	##########################################################
	##################### Data locations #####################
	input_path = "./../data/input-data/vf-ie/grid/"
	enrichment_path = "./../data/input-data/vf-ie/enrichment/"
	gis_path = "./../data/input-data/vf-ie/gis/"
	output_path = "./../data/output-data/vf-ie/grid/"
	##########################################################

	# Load data
	try:
		print("Load grid data..")
		grid_df = loadInputData(input_path)
		print("Grid file processed with {} rows".format(grid_df.shape[0]))
	except:
		print("Issue loading grid data, exiting program..")
		sys.exit(0)

	# Rename columns
	try:
		print("\nRename columns..")
		grid_df.rename(columns = {'id' : 'grid', 'global_cell_id' : 'cilac', 'eventCount' : 'event_count'}, inplace = True)
	except:
		print("\nIssue renaming columns, exiting program..")
		sys.exit(0)

	# Multiply values of RSRP/RSRQ for aggregation 
	try:
		grid_df['avg_rsrp'] = grid_df['avg_rsrp'] * grid_df['event_count']
		grid_df['avg_rsrq'] = grid_df['avg_rsrq'] * grid_df['event_count']
	except:
		print("\nIssue renaming columns, exiting program..")
		sys.exit(0)

	# Create grid-cell aggregation
	try:
		print("\nCreate grid-cell aggregation..")
		grid_agg_df = grid_df[['grid', 'cilac', 'geometry', 'avg_rsrp', 'avg_rsrq', 'event_count']].groupby(['grid', 'cilac', 'geometry']).sum().reset_index()
	except:
		print("\nIssue creating grid-cell aggregation, exiting program..")
		sys.exit(0)

	# Create rsrp/rsrq KPIs
	try:
		print("\nCreate average counters..")
		grid_agg_df['avg_rsrp'] = grid_agg_df['avg_rsrp'] / grid_agg_df['event_count']
		grid_agg_df['avg_rsrq'] = grid_agg_df['avg_rsrq'] / grid_agg_df['event_count']
	except:
		print("\nIssue creating average KPIs, exiting program..")
		sys.exit(0)

	# Create sinr KPI
	try:
		print("\nCreate sinr counter..")
		grid_agg_df['avg_sinr'] = grid_agg_df.apply(lambda x: calculate_sinr(x['avg_rsrp'], x['avg_rsrq']) + 45, axis = 1)
	except:
		print("\nIssue creating average KPIs, exiting program..")
		sys.exit(0)

	# Create grid-cell 
	try:
		print("\nCreated 'grid_cell'..")
		grid_agg_df['grid_cell'] = grid_agg_df.apply(lambda x: str(x['grid']) + "_" + str(x['cilac']), axis = 1)
	except:
		print("\nIssue creating 'grid_cell', exiting program..")
		sys.exit(0)
 
	try:
		print("\nCreate transpose dataframe..")
		grid_agg_df = createGridTranspose(grid_agg_df)
		
	except:
		print("\nIssue creating aggregated KQIs, exiting program..")
		sys.exit(0)

	try:
		print("\nCalculate percentage cell and grid data..")
		grid_agg_df['perc_cell_events'] = grid_agg_df.apply(lambda x: calcPercentage(x.event_count, x.cell_event_count), axis = 1)
		grid_agg_df['perc_grid_events'] = grid_agg_df.apply(lambda x: calcPercentage(x.event_count, x.grid_event_count), axis = 1)
	except:
		print("\nIssue creating percentage KPIs, exiting program..")
		sys.exit(0)

	try:
		print("\nJoin to GIS data..")
		gis_df = pd.read_csv("{}cell-gis.csv".format(gis_path))

		gis_df = gis_df[['Name', 'SectorID', 'Scr_Freq', 'Latitude', 'Longitude', 'Bearing', 'TiltE', 'TiltM', \
						'SiteID', 'City', 'Height', 'LAC']]

		gis_df['UARFCN'] = ""
		gis_df['Band'] = gis_df.apply(lambda x: getFreqBand(mid(x.Name, 5, 1)), axis = 1)
		gis_df['Vendor'] = "Ericsson"
		gis_df['MaxTransPwr'] = ""
		gis_df['FreqMHz'] = ""
		gis_df['HBW'] = 60

		gis_df.rename(columns = {'LAC' : 'TAC', 'SectorID' : 'cilac'}, inplace = True)

		print("\tRows pre GIS merge = {}".format(grid_agg_df.shape[0]))

		gis_df['cilac'] = gis_df['cilac'].astype(int).astype(str)
		grid_agg_df['cilac'] = grid_agg_df['cilac'].astype(int).astype(str)

		grid_agg_df = pd.merge(grid_agg_df,gis_df,how='inner',on = 'cilac')
		print("\tRows post GIS merge = {}".format(grid_agg_df.shape[0]))

	except:
		print("\nIssue importing GIS data, exiting program..")
		sys.exit(0)

	# Create GeoDataFrame
	try:
		print("\nCreate GeoDataFrame..")
		#grid_transpose["geometry"] = grid_transpose.apply(lambda x: "POINT (" + str(grid.decode(x.grid)).split(", ")[1].replace(")", "") + " " + str(grid.decode(x.grid)).split(", ")[0].replace("(", "") + ")", axis = 1)
		grid_agg_df['geometry'] = grid_agg_df['geometry'].apply(lambda x: shapely.wkt.loads(x))
		grid_agg_gdf = gpd.GeoDataFrame(grid_agg_df, geometry='geometry', crs='EPSG:4326')
		del grid_agg_df
	except:
		print("\nIssue creating GeoDataFrame, exiting program..")
		sys.exit(0)

	try:
		print("Get centroid of geometry..")
		grid_agg_gdf = grid_agg_gdf.to_crs('epsg:3785')
		grid_agg_gdf['geometry'] = grid_agg_gdf['geometry'].centroid
		grid_agg_gdf = grid_agg_gdf.to_crs('epsg:4326')
	except:
		print("\nIssue creating centroid of geometry, exiting program..")
		sys.exit(0)

	# Create distance and angle from cell to grid
	try:
		print("\nCreate distance and angle from cell to grid..")
		grid_agg_gdf['distance_to_cell'], grid_agg_gdf['cell_angle_to_grid'] = zip(*grid_agg_gdf.apply(lambda x: findDistance(x.geometry, x.Latitude, x.Longitude), axis = 1))
	except:
		print("\nIssue creating distance and angle from cell to grid, exiting program..")
		sys.exit(0)

	try:
		print("\nCreate distance metics..")
		grid_agg_gdf = getDistances(grid_agg_gdf)
	except:
		print("\nIssue creating distance to cells, exiting program..")
		sys.exit(0)

	# Calculate the percentage of the max distance from the cell the current cells is
	try:
		print("\nCalculate the percentage of the max distance from the cell the current cells is..")
		grid_agg_gdf['perc_cell_max_dist'] = grid_agg_gdf.apply(lambda x: calcPercentage(x.distance_to_cell, x.cell_max_distance_to_cell), axis = 1)
	except:
		print("\nIssue calculating the percentage of the max distance from the cell the current cells is, exiting program..")
		sys.exit(0)

	try:
		print("\nCalculate grid-bearing difference..")
		grid_agg_gdf['grid_bearing_diff'] = grid_agg_gdf.apply(lambda x: getBearingDiff(x['cell_angle_to_grid'], x['Bearing']), axis = 1)
	except:
		print("\nIssue calculating grid-bearing difference, exiting program..")
		sys.exit(0)

	try:
		print("\nReduce dataframe columns..")
		grid_agg_gdf = grid_agg_gdf[['grid_cell', 'grid', 'Name', 'cilac', 'avg_rsrp', 'avg_rsrq', 'avg_sinr', 'event_count', \
												'grid_event_count', 'cell_event_count', 'avg_rsrp_grid', 'avg_rsrq_geo_hash', 'avg_sinr_grid', \
												'avg_rsrp_cell', 'avg_rsrq_cell', 'avg_sinr_cell', 'perc_cell_events', 'perc_grid_events',  'distance_to_cell', \
												'grid_max_distance_to_cell', 'grid_min_distance_to_cell', 'cell_max_distance_to_cell', 'perc_cell_max_dist', \
												'cell_angle_to_grid', 'grid_bearing_diff', 'Scr_Freq', 'UARFCN', 'Bearing', 'TiltE', 'TiltM', 'SiteID', 'City', \
												'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', 'geometry']]
	except:
		print("\nIssue reducing columns in dataframe 'grid_agg_gdf', exiting program..")
		sys.exit(0)

	try:
		print("\nFilter dataframe by max distance..")
		max_dist = 32000
		print("Pre distance filtering rows = {}".format(grid_agg_gdf.shape[0]))
		grid_agg_gdf = grid_agg_gdf[grid_agg_gdf.distance_to_cell <= max_dist]
		print("Post distance filtering rows = {}".format(grid_agg_gdf.shape[0]))
	except:
		print("\nIssue filtering 'grid_agg_gdf' by max distance, exiting program..")
		sys.exit(0)

	try:
		print("\nChange column names..")
		grid_agg_gdf.rename(columns={'grid': 'grid', 'Name' : 'cell_name', \
											'avg_rsrp_grid' : 'avg_rsrp_grid',	'avg_rsrq_geo_hash' : 'avg_rsrq_grid', \
											'avg_sinr_grid' : 'avg_sinr_grid', }, inplace=True)
	except:
		print("\nIssue changing column names in 'grid_agg_gdf', exiting program..")
		sys.exit(0)

	# Save to file
	try:
		print("\nSave to file..")
		grid_agg_gdf.to_csv("{}grid-cell-data.csv".format(output_path), index = False)
	except:
		print("\nIssue saving 'grid_agg_gdf' to output_path, exiting program..")
		sys.exit(0)

if __name__ == "__main__":
	main()