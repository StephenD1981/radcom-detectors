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


def loadInputData(input_path):
	i = 0
	for root, dir, file in os.walk(input_path):
		for filename in file:
			if ("DS_Store" not in filename) & ("SUCCESS" not in filename):
				#print("\tLoading file: {}/{}".format(root, filename))
				if i == 0:
					df = pd.read_csv("{}/{}".format(root, filename), names = ['timestamp', 'geohash', 'cell', 'name', 'valueType', \
																				'group', 'mobility', 'slowMobility', 'stationay',    'unknown'])
					i += 1
				else:
					temp_df = pd.read_csv("{}/{}".format(root, filename), names = ['timestamp', 'geohash', 'cell', 'name', 'valueType', \
																					'group', 'mobility', 'slowMobility', 'stationay',    'unknown'])
					df = pd.concat([df, temp_df])
					i += 1
	return df, i

def kqiLookup(enrichment_path, grid_df):
	kqi_lookup = pd.read_csv("{}kqi-lookup.csv".format(enrichment_path))
	grid_df = pd.merge(grid_df,kqi_lookup,how='left',left_on='name', right_on = 'kqi_id')
	return grid_df

def extractKPIs(grid_df):
	kqi_list = ['RRC_Interval_Sum_RSRP', 'RRC_Interval_Sum_RSRQ', 'RRC_Interval_Sum_SINR', \
				'RRC_Interval_Coverage_Samples', 'RRC_Interval_Quality_Samples', 'RRC_Interval_SINR_Samples' , \
				'RRC_Sum_RSRP', 'RRC_Sum_RSRQ', 'RRC_Sum_SINR', \
				'RRC_Coverage_Samples', 'RRC_Quality_Samples', 'RRC_SINR_Samples']

	print("Number of rows pre-filtering KPIs = {}".format(grid_df.shape[0]))
	grid_df = grid_df[grid_df['kqi_name'].isin(kqi_list)].copy()
	print("Number of rows post-filtering KPIs = {}".format(grid_df.shape[0]))

	# Check if all required KPIs are available
	if (('RRC_Interval_Coverage_Samples' in grid_df['kqi_name'].to_list()) | ('RRC_Coverage_Samples' in grid_df['kqi_name'].to_list())) & \
		(('RRC_Interval_Quality_Samples' in grid_df['kqi_name'].to_list()) | ('RRC_Quality_Samples' in grid_df['kqi_name'].to_list())) & \
		(('RRC_Interval_SINR_Samples' in grid_df['kqi_name'].to_list()) | ('RRC_SINR_Samples' in grid_df['kqi_name'].to_list())) & \
		('RRC_Sum_RSRP' in grid_df['kqi_name'].to_list()) & ('RRC_Sum_RSRQ' in grid_df['kqi_name'].to_list()) & ('RRC_Sum_SINR' in grid_df['kqi_name'].to_list()):	
		return grid_df
	else:
		print("Issues, not all required KPIs are available, exiting program..")
		sys.exit(0)

def createKQIs(y):
	try:
		avg_int_rsrp = y['RRC_Interval_Sum_RSRP'] / y['RRC_Interval_Coverage_Samples']
	except:
		avg_int_rsrp = 'NaN'
	try:
		avg_int_rsrq = y['RRC_Interval_Sum_RSRQ'] / y['RRC_Interval_Quality_Samples']
	except:
		avg_int_rsrq = 'NaN'
	try:
		avg_int_sinr = y['RRC_Interval_Sum_SINR'] / y['RRC_Interval_SINR_Samples']
	except:
		avg_int_sinr = 'NaN'
	try:
		avg_rsrp = y['RRC_Sum_RSRP'] / y['RRC_Coverage_Samples']
	except:
		avg_rsrp = 'NaN'
	try:
		avg_rsrq = y['RRC_Sum_RSRQ'] / y['RRC_Quality_Samples']
	except:
		avg_rsrq = 'NaN'
	try:
		avg_sinr = y['RRC_Sum_SINR'] / y['RRC_SINR_Samples']
	except:
		avg_sinr =  'NaN'
	
	try:
		if math.isnan(y['RRC_Interval_Coverage_Samples']):
			int_samples_rsrp = 0
		else:
			int_samples_rsrp = y['RRC_Interval_Coverage_Samples']
	except:
		int_samples_rsrp = 0
	try:
		if math.isnan(y['RRC_Interval_Quality_Samples']):
			int_samples_rsrq = 0
		else:
			int_samples_rsrq = y['RRC_Interval_Quality_Samples']
	except:
		int_samples_rsrq = 0
	try:
		if math.isnan(y['RRC_Interval_SINR_Samples']):
			int_samples_sinr = 0
		else:
			int_samples_sinr = y['RRC_Interval_SINR_Samples']
	except:
		int_samples_sinr = 0
	try:
		if math.isnan(y['RRC_Coverage_Samples']):
			samples_rsrp = 0
		else:
			samples_rsrp = y['RRC_Coverage_Samples']
	except:
		samples_rsrp = 0
	try:
		if math.isnan(y['RRC_Quality_Samples']):
			samples_rsrq = 0
		else:
			samples_rsrq = y['RRC_Quality_Samples']
	except:
		samples_rsrq = 0
	try:
		if math.isnan(y['RRC_SINR_Samples']):
			samples_sinr = 0
		else:
			samples_sinr = y['RRC_SINR_Samples']
	except:
		samples_sinr = 0

	event_count = int_samples_rsrp + int_samples_rsrq + int_samples_sinr + samples_rsrp + samples_rsrq + samples_sinr

	return avg_int_rsrp, avg_int_rsrq, avg_int_sinr, avg_rsrp, avg_rsrq, avg_sinr, event_count

def SelectValues(avg_interval_rsrp, avg_interval_rsrq, avg_interval_sinr, avg_rsrp, avg_rsrq, avg_sinr):
	# Get RSRP
	if math.isnan(avg_interval_rsrp):
		rsrp = avg_rsrp
	else:
		rsrp = avg_interval_rsrp
	# Get RSRQ
	if math.isnan(avg_interval_rsrq):
		rsrq = avg_rsrq
	else:
		rsrq = avg_interval_rsrq
	# Get SINR
	if math.isnan(avg_interval_sinr):
		sinr = avg_sinr
	else:
		sinr = avg_interval_sinr
	
	return rsrp, rsrq, sinr

def createGridTranspose(grid_transpose):
	grid_transpose_count_grid = grid_transpose[['geohash', 'event_count']].groupby(['geohash']).sum().reset_index()
	grid_transpose_count_grid.rename(columns={'event_count': 'grid_event_count'}, inplace=True)

	grid_transpose_count_cell = grid_transpose[['cell', 'event_count']].groupby(['cell']).sum().reset_index()
	grid_transpose_count_cell.rename(columns={'event_count': 'cell_event_count'}, inplace=True)

	######################
	### Get Grid means ###
	######################
	grid_transpose_mean_grid = grid_transpose[['geohash', 'avg_rsrp', 'avg_rsrq', 'avg_sinr']].groupby(['geohash']).mean().reset_index()

	grid_transpose_mean_grid.rename(columns={'avg_rsrp' : 'avg_rsrp_geohash', 'avg_rsrq' : 'avg_rsrq_geo_hash', \
	                                         'avg_sinr' : 'avg_sinr_geohash'}, inplace=True)

	######################
	### Get Cell means ###
	######################
	grid_transpose_mean_cell = grid_transpose[['cell', 'avg_rsrp', 'avg_rsrq', 'avg_sinr']].groupby(['cell']).mean().reset_index()

	grid_transpose_mean_cell.rename(columns={'avg_rsrp' : 'avg_rsrp_cell', 'avg_rsrq' : 'avg_rsrq_cell', \
	                                         'avg_sinr' : 'avg_sinr_cell'}, inplace=True)

	data_frames_1 = [grid_transpose, grid_transpose_count_grid, grid_transpose_mean_grid]
	grid_transpose = reduce(lambda  left,right: pd.merge(left,right,on=['geohash'], how='outer'), data_frames_1)
	data_frames_2 = [grid_transpose, grid_transpose_count_cell, grid_transpose_mean_cell]
	grid_transpose = reduce(lambda  left,right: pd.merge(left,right,on=['cell'], how='outer'), data_frames_2)

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

def getDistances(grid_transpose_gdf):
	grid_max_dist = grid_transpose_gdf[['geohash', 'distance_to_cell']].groupby(['geohash']).max().reset_index()
	grid_max_dist.rename(columns={'distance_to_cell': 'grid_max_distance_to_cell'}, inplace=True)

	grid_min_dist = grid_transpose_gdf[['geohash', 'distance_to_cell']].groupby(['geohash']).min().reset_index()
	grid_min_dist.rename(columns={'distance_to_cell': 'grid_min_distance_to_cell'}, inplace=True)

	cell_max_dist = grid_transpose_gdf[['cell', 'distance_to_cell']].groupby(['cell']).max().reset_index()
	cell_max_dist.rename(columns={'distance_to_cell': 'cell_max_distance_to_cell'}, inplace=True)

	data_frames_1 = [grid_transpose_gdf, grid_max_dist, grid_min_dist]
	grid_transpose_gdf = reduce(lambda  left,right: pd.merge(left,right,on=['geohash'], how='outer'), data_frames_1)

	data_frames_2 = [grid_transpose_gdf, cell_max_dist]
	grid_transpose_gdf = reduce(lambda  left,right: pd.merge(left,right,on=['cell'], how='outer'), data_frames_2)

	return grid_transpose_gdf

def getBearingDiff(cell_angle_to_grid, bearing):
	grid_bearing_diff = math.fabs(cell_angle_to_grid - bearing)
	if grid_bearing_diff > 180:
		grid_bearing_diff = 360 - grid_bearing_diff
	return grid_bearing_diff

def cellCountPerGrid(grid_transpose_gdf):
	cells_per_grid = grid_transpose_gdf[['geohash', 'Name']].groupby('geohash').count().reset_index()
	cells_per_grid.rename(columns={'Name': 'cell_count'}, inplace=True)
	grid_transpose_gdf = pd.merge(grid_transpose_gdf, cells_per_grid, on = 'geohash', how = 'left')
	return grid_transpose_gdf

def cellPciCountPerGrid(grid_transpose_gdf):
	pci_grid = grid_transpose_gdf[['geohash', 'Scr_Freq', 'Name']].groupby(['geohash', 'Scr_Freq']).count().reset_index()
	pci_grid.rename(columns={'Name': 'same_pci_cell_count'}, inplace=True)
	grid_transpose_gdf = pd.merge(grid_transpose_gdf, pci_grid, on = ['geohash', 'Scr_Freq'], how = 'left')
	return grid_transpose_gdf

def dominanceClassify(perc_grid_events, perc_cell_max_dist, dominant , interferer, tier_2, tier_3, tier_4):
	# Classify cell as 'dominant', 'interferer', 'contibutor'
	if perc_grid_events >= dominant:
		dominance = "dominant"
	elif perc_grid_events <= interferer:
		dominance = "interferer"
	else:
		dominance = "contibutor"

	# Classify cell to grid distance 
	if perc_cell_max_dist >= tier_4:
		tier = 'tier_4'
	elif perc_cell_max_dist >= tier_3:
		tier = "tier_3"
	elif perc_cell_max_dist >= tier_2:
		tier = "tier_2"
	else:
		tier = "tier_1"
	return dominance, tier


	

def main():
	'''
	This pprogram takes standard MEA grid data as an input and creates final grid dataset for optimisation purposes
	It should be noted that the final grid dataset is exactly the same as the grid dataset produced using CM/PM inputs
	'''
	##########################################################
	##################### Data locations #####################
	input_path = "./../data/input-data/dish/grid/"
	intermediate_path = "./../data/intermediate-data/dish/"
	enrichment_path = "./../data/input-data/dish/enrichment/"
	gis_path = "./../data/input-data/dish/gis/"
	output_path = "./../data/output-data/dish/grid/"
	##########################################################

	#################### Define Variables ####################
	dominant = 0.3
	interferer = 0.02
	tier_4 = 0.95
	tier_3 = 0.7
	tier_2 = 0.35
	##########################################################


	# Load data
	try:
		print("Load grid data..")
		grid_df, file_count = loadInputData(input_path)
		print("{} files processed with {} rows".format(file_count, grid_df.shape[0]))
	except:
		print("Issue loading grid data, exiting program..")
		sys.exit(0)
	
	# Enrich data with KPI names
	try:
		print("\nEnrich data with KQI names..")
		grid_df = kqiLookup(enrichment_path, grid_df)
	except:
		print("\nIssue enriching data with KQI names, exiting program..")
		sys.exit(0)

	# Extract KPIs
	try:
		print("\nExtract required KPIs..")
		grid_df = extractKPIs(grid_df)
	except:
		print("\nIssue extracting required KPIs, exiting program..")
		sys.exit(0)

	# Create 'value' column
	try:
		print("\nCreate 'value' column..")
		grid_df["value"] = grid_df['mobility'] + grid_df['slowMobility'] + grid_df['stationay'] + grid_df['unknown']
		grid_df = grid_df[['cell', 'geohash','kqi_name', 'value']]
	except:
		print("\nIssue creating 'value' KPI, exiting program..")
		sys.exit(0)

	# Create Grid transpose
	try:
		print("\nCreate Transpose dataframe..")
		grid_transpose = grid_df.pivot(index = ['cell', 'geohash'], columns = 'kqi_name', values = 'value').rename_axis(columns=None).reset_index()
		grid_transpose["grid_cell"] = grid_transpose.apply(lambda x: str(x.geohash) + "_" + str(x.cell), axis = 1)
		print("Number of rows = {}, number of distinct grid-cells = {}".format(grid_transpose.shape[0], len(grid_transpose['grid_cell'].unique().tolist())))
	except:
		print("\nIssue creating transpose dataframe, exiting program..")
		sys.exit(0)

	# Create rsrp/rsrq/sinr KPIs
	try:
		print("\nCreate average counters..")
		grid_transpose['avg_interval_rsrp'], grid_transpose['avg_interval_rsrq'], grid_transpose['avg_interval_sinr'], grid_transpose['avg_rsrp'], grid_transpose['avg_rsrq'], grid_transpose['avg_sinr'], grid_transpose['event_count']= zip(*grid_transpose.apply(lambda x: createKQIs(x), axis = 1))#['RRC_Interval_Sum_RSRP'], x['RRC_Interval_Coverage_Samples'], x['RRC_Interval_Sum_RSRQ'], x['RRC_Interval_Quality_Samples'], x['RRC_Interval_Sum_SINR'], x['RRC_Interval_SINR_Samples'], x['RRC_Sum_RSRP'], x['RRC_Coverage_Samples'], x['RRC_Sum_RSRQ'], x['RRC_Quality_Samples'], x['RRC_Sum_SINR'], x['RRC_SINR_Samples']), axis = 1))
	except:
		print("\nIssue creating average KPIs, exiting program..")
		sys.exit(0)

	# Remove unrequired KPIs
	try:
		print("\nRemove unrequired KQI..")
		grid_transpose = grid_transpose[['cell', 'geohash', 'grid_cell', 'avg_interval_rsrp', 'avg_interval_rsrq', \
										'avg_interval_sinr', 'avg_rsrp', 'avg_rsrq', 'avg_sinr', 'event_count']]
	except:
		print("\nIssue reducing KPIs, exiting program..")
		sys.exit(0)

	try:
		print("\nCreate average RSRP/RSRQ/SINR KQIs..")
		grid_transpose['avg_rsrp'], grid_transpose['avg_rsrq'], grid_transpose['avg_sinr'] = zip(*grid_transpose.apply(lambda x: SelectValues(x['avg_interval_rsrp'], x['avg_interval_rsrq'], x['avg_interval_sinr'], x['avg_rsrp'], x['avg_rsrq'], x['avg_sinr']), axis = 1))
		grid_transpose = grid_transpose[['cell', 'geohash', 'grid_cell', 'avg_rsrp', 'avg_rsrq', 'avg_sinr', 'event_count']]
	except:
		print("\nIssue RSRP/RSRQ/SINR KPIs, exiting program..")
		sys.exit(0)

	try:
		print("\nCreate transpose dataframe..")
		grid_transpose = createGridTranspose(grid_transpose)
		
	except:
		print("\nIssue creating aggregated KIs, exiting program..")
		sys.exit(0)

	try:
		print("\nCalculate percentage cell and grid data..")
		grid_transpose['perc_cell_events'] = grid_transpose.apply(lambda x: calcPercentage(x.event_count, x.cell_event_count), axis = 1)
		grid_transpose['perc_grid_events'] = grid_transpose.apply(lambda x: calcPercentage(x.event_count, x.grid_event_count), axis = 1)
	except:
		print("\nIssue creating percentage KPIs, exiting program..")
		sys.exit(0)

	try:
		print("\nJoin to GIS data..")
		gis_df = pd.read_csv("{}gis.csv".format(gis_path), names = ['Name','CILAC', 'SectorID', 'RNC_BSC', 'LAC', 'SectorType', 'Scr_Freq', \
																	'UARFCN', 'BSIC', 'Tech', 'Latitude', 'Longitude','Bearing','AvgNeighborDist', \
																	'MaxNeighborDist', 'NeighborsCount', 'Eng', 'TiltE','TiltM', 'SiteID', \
																	'AdminCellState', 'Asset', 'Asset_Configuration', 'Cell_Type', 'Cell_Name', \
																	'City', 'Height', 'RF_Team', 'Asset_Calc', 'Sector_uniq', 'FreqType', 'TAC', \
																	'RAC', 'Band', 'Vendor', 'CPICHPwr', 'MaxTransPwr', 'FreqMHz', 'HBW', \
																	'VBW', 'Antenna'])

		gis_df = gis_df[['Name', 'CILAC', 'Scr_Freq', 'UARFCN', 'Latitude', 'Longitude', 'Bearing', 'TiltE', 'TiltM', \
						'SiteID', 'City', 'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW']]

		print("\tRows pre GIS merge = {}".format(grid_transpose.shape[0]))

		gis_df['CILAC'] = gis_df['CILAC'].astype(int).astype(str)
		grid_transpose['cell'] = grid_transpose['cell'].astype(int).astype(str)

		grid_transpose = pd.merge(grid_transpose,gis_df,how='inner',left_on = 'cell', right_on = 'CILAC')
		print("\tRows post GIS merge = {}".format(grid_transpose.shape[0]))

	except:
		print("\nIssue importing GIS data, exiting program..")
		sys.exit(0)

	# Create GeoDataFrame
	try:
		print("\nCreate GeoDataFrame..")
		grid_transpose["geometry"] = grid_transpose.apply(lambda x: "POINT (" + str(geohash.decode(x.geohash)).split(", ")[1].replace(")", "") + " " + str(geohash.decode(x.geohash)).split(", ")[0].replace("(", "") + ")", axis = 1)
		grid_transpose['geometry'] = grid_transpose['geometry'].apply(lambda x: shapely.wkt.loads(x))
		grid_transpose_gdf = gpd.GeoDataFrame(grid_transpose, geometry='geometry', crs='EPSG:4326')
		del grid_transpose
	except:
		print("\nIssue creating GeoDataFrame, exiting program..")
		sys.exit(0)

	# Create distance and angle from cell to grid
	try:
		print("\nCreate distance and angle from cell to grid..")
		grid_transpose_gdf['distance_to_cell'], grid_transpose_gdf['cell_angle_to_grid'] = zip(*grid_transpose_gdf.apply(lambda x: findDistance(x.geometry, x.Latitude, x.Longitude), axis = 1))
	except:
		print("\nIssue creating distance and angle from cell to grid, exiting program..")
		sys.exit(0)

	try:
		print("\nCreate distance metics..")
		grid_transpose_gdf = getDistances(grid_transpose_gdf)
	except:
		print("\nIssue creating distance to cells, exiting program..")
		sys.exit(0)

	# Calculate the percentage of the max distance from the cell the current cells is
	try:
		print("\nCalculate the percentage of the max distance from the cell the current cells is..")
		grid_transpose_gdf['perc_cell_max_dist'] = grid_transpose_gdf.apply(lambda x: calcPercentage(x.distance_to_cell, x.cell_max_distance_to_cell), axis = 1)
	except:
		print("\nIssue calculating the percentage of the max distance from the cell the current cells is, exiting program..")
		sys.exit(0)

	try:
		print("\nCalculate grid-bearing difference..")
		grid_transpose_gdf['grid_bearing_diff'] = grid_transpose_gdf.apply(lambda x: getBearingDiff(x['cell_angle_to_grid'], x['Bearing']), axis = 1)
	except:
		print("\nIssue calculating grid-bearing difference, exiting program..")
		sys.exit(0)

	try:
		print("\nReduce dataframe columns..")
		grid_transpose_gdf = grid_transpose_gdf[['grid_cell', 'geohash', 'Name', 'CILAC', 'avg_rsrp', 'avg_rsrq', 'avg_sinr', 'event_count', \
												'grid_event_count', 'cell_event_count', 'avg_rsrp_geohash', 'avg_rsrq_geo_hash', 'avg_sinr_geohash', \
												'avg_rsrp_cell', 'avg_rsrq_cell', 'avg_sinr_cell', 'perc_cell_events', 'perc_grid_events',  'distance_to_cell', \
												'grid_max_distance_to_cell', 'grid_min_distance_to_cell', 'cell_max_distance_to_cell', 'perc_cell_max_dist', \
												'cell_angle_to_grid', 'grid_bearing_diff', 'Scr_Freq', 'UARFCN', 'Bearing', 'TiltE', 'TiltM', 'SiteID', 'City', \
												'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', 'geometry']]
	except:
		print("\nIssue reducing columns in dataframe 'grid_transpose_gdf', exiting program..")
		sys.exit(0)

	try:
		print("\nFilter dataframe by max distance..")
		max_dist = 32000
		print("Pre distance filtering rows = {}".format(grid_transpose_gdf.shape[0]))
		grid_transpose_gdf = grid_transpose_gdf[grid_transpose_gdf.distance_to_cell <= max_dist]
		print("Post distance filtering rows = {}".format(grid_transpose_gdf.shape[0]))
	except:
		print("\nIssue filtering 'grid_transpose_gdf' by max distance, exiting program..")
		sys.exit(0)

	try:
		print("\nAdd cell count per grid..")
		grid_transpose_gdf = cellCountPerGrid(grid_transpose_gdf)
	except:
		print("\nIssue adding cell count per grid, exiting program..")
		sys.exit(0)

	try:
		print("\nAdd PCI cell count per grid..")
		grid_transpose_gdf = cellPciCountPerGrid(grid_transpose_gdf)
	except:
		print("\nIssue adding cell PCI count per grid, exiting program..")
		sys.exit(0)

	try:
		print("\nAdd dominance and distance tier clasifications..")
		grid_transpose_gdf['dominance'], grid_transpose_gdf['dist_band'] = zip(*grid_transpose_gdf.apply(lambda x: dominanceClassify(x.perc_grid_events, x.perc_cell_max_dist, dominant , interferer, tier_2, tier_3, tier_4), axis = 1))
	except:
	print("\nIssue adding dominance ad distance tier clasifications, exiting program..")
	sys.exit(0)

	try:
		print("\nChange column names..")
		grid_transpose_gdf.rename(columns={'geohash': 'grid', 'Name' : 'cell_name', 'CILAC' : 'cilac', \
											'avg_rsrp_geohash' : 'avg_rsrp_grid',	'avg_rsrq_geo_hash' : 'avg_rsrq_grid', \
											'avg_sinr_geohash' : 'avg_sinr_grid', }, inplace=True)
	except:
		print("\nIssue changing column names in 'grid_transpose_gdf', exiting program..")
		sys.exit(0)
	

	# Save to file
	try:
		print("\nSave to file..")
		grid_transpose_gdf.to_csv("{}grid-cell-data.csv".format(output_path), index = False)
	except:
		print("\nIssue saving 'grid_transpose_gdf' to output_path, exiting program..")
		sys.exit(0)

if __name__ == "__main__":
	main()