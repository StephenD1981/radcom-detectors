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
import random

def mid(s, offset, amount):
	return s[offset:offset+amount]

def getMaxTa(x):
	percentage = 0
	for key, value in x.items():
		if ("perc" in key) and (percentage < 98):
			percentage += value
			if percentage >= 98:
				return_ta =  int(key.split("_")[2])
				break
	return return_ta

def createDF(eci, count, min_ta, max_ta):
	ta_list = [random.uniform(min_ta, max_ta) for _ in range(int(count))]
	eci_list = [eci for _ in range(int(count))]
	max_ta_list = [max_ta for _ in range(int(count))]
	return_df = pd.DataFrame(
		{'eci': eci_list,
		'ta': ta_list,
		'max_ta' : max_ta_list
		})
	return return_df

def explodeData(ta_density_df):
	for index, row in ta_density_df.iterrows():
		ta_0_200 = createDF(row['eci'], row['ta_0_200'], 0, 200)
		ta_200_400 = createDF(row['eci'], row['ta_200_400'], 200, 400)
		ta_400_600 = createDF(row['eci'], row['ta_400_600'], 400, 600)
		ta_600_800 = createDF(row['eci'], row['ta_600_800'], 600, 800)
		ta_800_1000 = createDF(row['eci'], row['ta_800_1000'], 800, 1000)
		ta_1000_1200 = createDF(row['eci'], row['ta_1000_1200'], 1000, 1200)
		ta_1200_1400 = createDF(row['eci'], row['ta_1200_1400'], 1200, 1400)
		ta_1400_1600 = createDF(row['eci'], row['ta_1400_1600'], 1400, 1600)
		ta_1600_1800 = createDF(row['eci'], row['ta_1600_1800'], 1600, 1800)
		ta_1800_2000 = createDF(row['eci'], row['ta_1800_2000'], 1800, 2000)
		ta_2000_2200 = createDF(row['eci'], row['ta_2000_2200'], 2000, 2200)
		ta_2200_2400 = createDF(row['eci'], row['ta_2200_2400'], 2200, 2400)
		ta_2400_2600 = createDF(row['eci'], row['ta_2400_2600'], 2400, 2600)
		ta_2600_2800 = createDF(row['eci'], row['ta_2600_2800'], 2600, 2800)
		ta_2800_3000 = createDF(row['eci'], row['ta_2800_3000'], 2800, 3000)
		ta_3000_3200 = createDF(row['eci'], row['ta_3000_3200'], 3000, 3200)
		ta_3200_3400 = createDF(row['eci'], row['ta_3200_3400'], 3200, 3400)
		ta_3400_3600 = createDF(row['eci'], row['ta_3400_3600'], 3400, 3600)
		ta_3600_3800 = createDF(row['eci'], row['ta_3600_3800'], 3600, 3800)
		ta_3800_4000 = createDF(row['eci'], row['ta_3800_4000'], 3800, 4000)
		ta_4000_4200 = createDF(row['eci'], row['ta_4000_4200'], 4000, 4200)
		ta_4200_4400 = createDF(row['eci'], row['ta_4200_4400'], 4200, 4400)
		ta_4400_4600 = createDF(row['eci'], row['ta_4400_4600'], 4400, 4600)
		ta_4600_4800 = createDF(row['eci'], row['ta_4600_4800'], 4600, 4800)
		ta_4800_5000 = createDF(row['eci'], row['ta_4800_5000'], 4800, 5000)
		ta_5000_5200 = createDF(row['eci'], row['ta_5000_5200'], 5000, 5200)
		ta_5200_5400 = createDF(row['eci'], row['ta_5200_5400'], 5200, 5400)
		ta_5400_5600 = createDF(row['eci'], row['ta_5400_5600'], 5400, 5600)
		ta_5600_5800 = createDF(row['eci'], row['ta_5600_5800'], 5600, 5800)
		ta_5800_6000 = createDF(row['eci'], row['ta_5800_6000'], 5800, 6000)
		ta_6000_6200 = createDF(row['eci'], row['ta_6000_6200'], 6000, 6200)
		ta_6200_6400 = createDF(row['eci'], row['ta_6200_6400'], 6200, 6400)
		ta_6400_6600 = createDF(row['eci'], row['ta_6400_6600'], 6400, 6600)
		ta_6600_6800 = createDF(row['eci'], row['ta_6600_6800'], 6600, 6800)
		ta_6800_7000 = createDF(row['eci'], row['ta_6800_7000'], 6800, 7000)
		ta_7000_7200 = createDF(row['eci'], row['ta_7000_7200'], 7000, 7200)
		ta_7200_7400 = createDF(row['eci'], row['ta_7200_7400'], 7200, 7400)
		ta_7400_7600 = createDF(row['eci'], row['ta_7400_7600'], 7400, 7600)
		ta_7600_7800 = createDF(row['eci'], row['ta_7600_7800'], 7600, 7800)
		ta_7800_8000 = createDF(row['eci'], row['ta_7800_8000'], 7800, 8000)
		ta_8000_8200 = createDF(row['eci'], row['ta_8000_8200'], 8000, 8200)
		ta_8200_8400 = createDF(row['eci'], row['ta_8200_8400'], 8200, 8400)
		ta_8400_8600 = createDF(row['eci'], row['ta_8400_8600'], 8400, 8600)
		ta_8600_8800 = createDF(row['eci'], row['ta_8600_8800'], 8600, 8800)
		ta_8800_9000 = createDF(row['eci'], row['ta_8800_9000'], 8800, 9000)
		ta_9000_9200 = createDF(row['eci'], row['ta_9000_9200'], 9000, 9200)
		ta_9200_9400 = createDF(row['eci'], row['ta_9200_9400'], 9200, 9400)
		ta_9400_9600 = createDF(row['eci'], row['ta_9400_9600'], 9400, 9600)
		ta_9600_9800 = createDF(row['eci'], row['ta_9600_9800'], 9600, 9800)
		ta_9800_10000 = createDF(row['eci'], row['ta_9800_10000'], 9800, 10000)
		ta_10000_10200 = createDF(row['eci'], row['ta_10000_10200'], 10000, 10200)
		ta_10200_10400 = createDF(row['eci'], row['ta_10200_10400'], 10200, 10400)
		ta_10400_10600 = createDF(row['eci'], row['ta_10400_10600'], 10400, 10600)
		ta_10600_10800 = createDF(row['eci'], row['ta_10600_10800'], 10600, 10800)
		ta_10800_11000 = createDF(row['eci'], row['ta_10800_11000'], 10800, 11000)
		ta_11000_11200 = createDF(row['eci'], row['ta_11000_11200'], 11000, 11200)
		ta_11200_11400 = createDF(row['eci'], row['ta_11200_11400'], 11200, 11400)
		ta_11400_11600 = createDF(row['eci'], row['ta_11400_11600'], 11400, 11600)
		ta_11600_11800 = createDF(row['eci'], row['ta_11600_11800'], 11600, 11800)
		ta_11800_12000 = createDF(row['eci'], row['ta_11800_12000'], 11800, 12000)
		ta_12000_12200 = createDF(row['eci'], row['ta_12000_12200'], 12000, 12200)
		ta_12200_12400 = createDF(row['eci'], row['ta_12200_12400'], 12200, 12400)
		ta_12400_12600 = createDF(row['eci'], row['ta_12400_12600'], 12400, 12600)
		ta_12600_12800 = createDF(row['eci'], row['ta_12600_12800'], 12600, 12800)
		ta_12800_13000 = createDF(row['eci'], row['ta_12800_13000'], 12800, 13000)
		ta_13000_13200 = createDF(row['eci'], row['ta_13000_13200'], 13000, 13200)
		ta_13200_13400 = createDF(row['eci'], row['ta_13200_13400'], 13200, 13400)
		ta_13400_13600 = createDF(row['eci'], row['ta_13400_13600'], 13400, 13600)
		ta_13600_13800 = createDF(row['eci'], row['ta_13600_13800'], 13600, 13800)
		ta_13800_14000 = createDF(row['eci'], row['ta_13800_14000'], 13800, 14000)
		ta_14000_14200 = createDF(row['eci'], row['ta_14000_14200'], 14000, 14200)
		ta_14200_14400 = createDF(row['eci'], row['ta_14200_14400'], 14200, 14400)
		ta_14400_14600 = createDF(row['eci'], row['ta_14400_14600'], 14400, 14600)
		ta_14600_14800 = createDF(row['eci'], row['ta_14600_14800'], 14600, 14800)
		ta_14800_15000 = createDF(row['eci'], row['ta_14800_15000'], 14800, 15000)
		ta_15000_15200 = createDF(row['eci'], row['ta_15000_15200'], 15000, 15200)
		ta_15200_15400 = createDF(row['eci'], row['ta_15200_15400'], 15200, 15400)
		ta_15400_15600 = createDF(row['eci'], row['ta_15400_15600'], 15400, 15600)
		ta_15600_15800 = createDF(row['eci'], row['ta_15600_15800'], 15600, 15800)
		ta_15800_16000 = createDF(row['eci'], row['ta_15800_16000'], 15800, 16000)
		ta_16000_16200 = createDF(row['eci'], row['ta_16000_16200'], 16000, 16200)
		ta_16200_16400 = createDF(row['eci'], row['ta_16200_16400'], 16200, 16400)
		ta_16400_16600 = createDF(row['eci'], row['ta_16400_16600'], 16400, 16600)
		ta_16600_16800 = createDF(row['eci'], row['ta_16600_16800'], 16600, 16800)
		ta_16800_17000 = createDF(row['eci'], row['ta_16800_17000'], 16800, 17000)
		ta_17000_17200 = createDF(row['eci'], row['ta_17000_17200'], 17000, 17200)
		ta_17200_17400 = createDF(row['eci'], row['ta_17200_17400'], 17200, 17400)
		ta_17400_17600 = createDF(row['eci'], row['ta_17400_17600'], 17400, 17600)
		ta_17600_17800 = createDF(row['eci'], row['ta_17600_17800'], 17600, 17800)
		ta_17800_18000 = createDF(row['eci'], row['ta_17800_18000'], 17800, 18000)
		ta_18000_18200 = createDF(row['eci'], row['ta_18000_18200'], 18000, 18200)
		ta_18200_18400 = createDF(row['eci'], row['ta_18200_18400'], 18200, 18400)
		ta_18400_18600 = createDF(row['eci'], row['ta_18400_18600'], 18400, 18600)
		ta_18600_18800 = createDF(row['eci'], row['ta_18600_18800'], 18600, 18800)
		ta_18800_19000 = createDF(row['eci'], row['ta_18800_19000'], 18800, 19000)
		ta_19000_19200 = createDF(row['eci'], row['ta_19000_19200'], 19000, 19200)
		ta_19200_19400 = createDF(row['eci'], row['ta_19200_19400'], 19200, 19400)
		ta_19400_19600 = createDF(row['eci'], row['ta_19400_19600'], 19400, 19600)
		ta_19600_19800 = createDF(row['eci'], row['ta_19600_19800'], 19600, 19800)
		ta_19800_20000 = createDF(row['eci'], row['ta_19800_20000'], 19800, 20000)
		ta_20000_20200 = createDF(row['eci'], row['ta_20000_20200'], 20000, 20200)
		ta_20200_20400 = createDF(row['eci'], row['ta_20200_20400'], 20200, 20400)
		ta_20400_20600 = createDF(row['eci'], row['ta_20400_20600'], 20400, 20600)
		ta_20600_20800 = createDF(row['eci'], row['ta_20600_20800'], 20600, 20800)
		ta_20800_21000 = createDF(row['eci'], row['ta_20800_21000'], 20800, 21000)
		ta_21000_21200 = createDF(row['eci'], row['ta_21000_21200'], 21000, 21200)
		ta_21200_21400 = createDF(row['eci'], row['ta_21200_21400'], 21200, 21400)
		ta_21400_21600 = createDF(row['eci'], row['ta_21400_21600'], 21400, 21600)
		ta_21600_21800 = createDF(row['eci'], row['ta_21600_21800'], 21600, 21800)
		ta_21800_22000 = createDF(row['eci'], row['ta_21800_22000'], 21800, 22000)
		ta_22000_22200 = createDF(row['eci'], row['ta_22000_22200'], 22000, 22200)
		ta_22200_22400 = createDF(row['eci'], row['ta_22200_22400'], 22200, 22400)
		ta_22400_22600 = createDF(row['eci'], row['ta_22400_22600'], 22400, 22600)
		ta_22600_22800 = createDF(row['eci'], row['ta_22600_22800'], 22600, 22800)
		ta_22800_23000 = createDF(row['eci'], row['ta_22800_23000'], 22800, 23000)
		ta_23000_23200 = createDF(row['eci'], row['ta_23000_23200'], 23000, 23200)
		ta_23200_23400 = createDF(row['eci'], row['ta_23200_23400'], 23200, 23400)
		ta_23400_23600 = createDF(row['eci'], row['ta_23400_23600'], 23400, 23600)
		ta_23600_23800 = createDF(row['eci'], row['ta_23600_23800'], 23600, 23800)
		ta_23800_24000 = createDF(row['eci'], row['ta_23800_24000'], 23800, 24000)
		ta_24000_24200 = createDF(row['eci'], row['ta_24000_24200'], 24000, 24200)
		ta_24200_24400 = createDF(row['eci'], row['ta_24200_24400'], 24200, 24400)
		ta_24400_24600 = createDF(row['eci'], row['ta_24400_24600'], 24400, 24600)
		ta_24600_24800 = createDF(row['eci'], row['ta_24600_24800'], 24600, 24800)
		ta_24800_25000 = createDF(row['eci'], row['ta_24800_25000'], 24800, 25000)
		ta_25000_25200 = createDF(row['eci'], row['ta_25000_25200'], 25000, 25200)
		ta_25200_25400 = createDF(row['eci'], row['ta_25200_25400'], 25200, 25400)
		ta_25400_25600 = createDF(row['eci'], row['ta_25400_25600'], 25400, 25600)
		ta_25600_25800 = createDF(row['eci'], row['ta_25600_25800'], 25600, 25800)
		ta_25800_26000 = createDF(row['eci'], row['ta_25800_26000'], 25800, 26000)
		ta_26000_26200 = createDF(row['eci'], row['ta_26000_26200'], 26000, 26200)
		ta_26200_26400 = createDF(row['eci'], row['ta_26200_26400'], 26200, 26400)
		ta_26400_26600 = createDF(row['eci'], row['ta_26400_26600'], 26400, 26600)
		ta_26600_26800 = createDF(row['eci'], row['ta_26600_26800'], 26600, 26800)
		ta_26800_27000 = createDF(row['eci'], row['ta_26800_27000'], 26800, 27000)
		ta_27000_27200 = createDF(row['eci'], row['ta_27000_27200'], 27000, 27200)
		ta_27200_27400 = createDF(row['eci'], row['ta_27200_27400'], 27200, 27400)
		ta_27400_27600 = createDF(row['eci'], row['ta_27400_27600'], 27400, 27600)
		ta_27600_27800 = createDF(row['eci'], row['ta_27600_27800'], 27600, 27800)
		ta_27800_28000 = createDF(row['eci'], row['ta_27800_28000'], 27800, 28000)
		ta_28000_28200 = createDF(row['eci'], row['ta_28000_28200'], 28000, 28200)
		ta_28200_28400 = createDF(row['eci'], row['ta_28200_28400'], 28200, 28400)
		ta_28400_28600 = createDF(row['eci'], row['ta_28400_28600'], 28400, 28600)
		ta_28600_28800 = createDF(row['eci'], row['ta_28600_28800'], 28600, 28800)
		ta_28800_29000 = createDF(row['eci'], row['ta_28800_29000'], 28800, 29000)
		ta_29000_29200 = createDF(row['eci'], row['ta_29000_29200'], 29000, 29200)
		ta_29200_29400 = createDF(row['eci'], row['ta_29200_29400'], 29200, 29400)
		ta_29400_29600 = createDF(row['eci'], row['ta_29400_29600'], 29400, 29600)
		ta_29600_29800 = createDF(row['eci'], row['ta_29600_29800'], 29600, 29800)
		ta_29800_30000 = createDF(row['eci'], row['ta_29800_30000'], 29800, 30000)
		ta_30000_30200 = createDF(row['eci'], row['ta_30000_30200'], 30000, 30200)
		ta_30200_30400 = createDF(row['eci'], row['ta_30200_30400'], 30200, 30400)
		ta_30400_30600 = createDF(row['eci'], row['ta_30400_30600'], 30400, 30600)
		ta_30600_30800 = createDF(row['eci'], row['ta_30600_30800'], 30600, 30800)
		ta_30800_31000 = createDF(row['eci'], row['ta_30800_31000'], 30800, 31000)
		ta_31000_31200 = createDF(row['eci'], row['ta_31000_31200'], 31000, 31200)
		ta_31200_31400 = createDF(row['eci'], row['ta_31200_31400'], 31200, 31400)
		ta_31400_31600 = createDF(row['eci'], row['ta_31400_31600'], 31400, 31600)
		ta_31600_31800 = createDF(row['eci'], row['ta_31600_31800'], 31600, 31800)
		ta_31800_32000 = createDF(row['eci'], row['ta_31800_32000'], 31800, 32000)

		df_list = [ta_0_200, ta_200_400, ta_400_600, ta_600_800, ta_800_1000, ta_1000_1200, ta_1200_1400, ta_1400_1600, ta_1600_1800, ta_1800_2000, ta_2000_2200, ta_2200_2400, \
					ta_2400_2600, ta_2600_2800, ta_2800_3000, ta_3000_3200, ta_3200_3400, ta_3400_3600, ta_3600_3800, ta_3800_4000, ta_4000_4200, ta_4200_4400, ta_4400_4600, ta_4600_4800, \
					ta_4800_5000, ta_5000_5200, ta_5200_5400, ta_5400_5600, ta_5600_5800, ta_5800_6000, ta_6000_6200, ta_6200_6400, ta_6400_6600, ta_6600_6800, ta_6800_7000, ta_7000_7200, \
					ta_7200_7400, ta_7400_7600, ta_7600_7800, ta_7800_8000, ta_8000_8200, ta_8200_8400, ta_8400_8600, ta_8600_8800, ta_8800_9000, ta_9000_9200, ta_9200_9400, ta_9400_9600, \
					ta_9600_9800, ta_9800_10000, ta_10000_10200, ta_10200_10400, ta_10400_10600, ta_10600_10800, ta_10800_11000, ta_11000_11200, ta_11200_11400, ta_11400_11600, ta_11600_11800, \
					ta_11800_12000, ta_12000_12200, ta_12200_12400, ta_12400_12600, ta_12600_12800, ta_12800_13000, ta_13000_13200, ta_13200_13400, ta_13400_13600, ta_13600_13800, ta_13800_14000, \
					ta_14000_14200, ta_14200_14400, ta_14400_14600, ta_14600_14800, ta_14800_15000, ta_15000_15200, ta_15200_15400, ta_15400_15600, ta_15600_15800, ta_15800_16000, ta_16000_16200, \
					ta_16200_16400, ta_16400_16600, ta_16600_16800, ta_16800_17000, ta_17000_17200, ta_17200_17400, ta_17400_17600, ta_17600_17800, ta_17800_18000, ta_18000_18200, ta_18200_18400, \
					ta_18400_18600, ta_18600_18800, ta_18800_19000, ta_19000_19200, ta_19200_19400, ta_19400_19600, ta_19600_19800, ta_19800_20000, ta_20000_20200, ta_20200_20400, ta_20400_20600, \
					ta_20600_20800, ta_20800_21000, ta_21000_21200, ta_21200_21400, ta_21400_21600, ta_21600_21800, ta_21800_22000, ta_22000_22200, ta_22200_22400, ta_22400_22600, ta_22600_22800, \
					ta_22800_23000, ta_23000_23200, ta_23200_23400, ta_23400_23600, ta_23600_23800, ta_23800_24000, ta_24000_24200, ta_24200_24400, ta_24400_24600, ta_24600_24800, ta_24800_25000, \
					ta_25000_25200, ta_25200_25400, ta_25400_25600, ta_25600_25800, ta_25800_26000, ta_26000_26200, ta_26200_26400, ta_26400_26600, ta_26600_26800, ta_26800_27000, ta_27000_27200, \
					ta_27200_27400, ta_27400_27600, ta_27600_27800, ta_27800_28000, ta_28000_28200, ta_28200_28400, ta_28400_28600, ta_28600_28800, ta_28800_29000, ta_29000_29200, ta_29200_29400, \
					ta_29400_29600, ta_29600_29800, ta_29800_30000, ta_30000_30200, ta_30200_30400, ta_30400_30600, ta_30600_30800, ta_30800_31000, ta_31000_31200, ta_31200_31400, ta_31400_31600, \
					ta_31600_31800, ta_31800_32000]

		if index == 0:
			return_df = pd.concat(df_list)
		else:
			temp_df = pd.concat(df_list)
			return_df = pd.concat([return_df, temp_df])

	return return_df

def calculateLatLon(cell_lat, cell_lon, ta, max_ta, multiple, hbeam, azimuth):
	"""
	This function calculates the lat, lon position of an event base on;
		1. Timing Advance + cell model
		2. Timing Advance (based on regression model) + cell model 
	"""
	# Lobe factors
	back_lobe_factor = 0.1
	side_lobe_factor = 0.5

	# Earth Radius
	R = 6378.1 * 1000

	# Offset TA by +/- 5% to spread data
	ta = ta * random.uniform(.95, 1.05)

	# Oval beam shape
	#beam = ((hbeam * 3)/2) - (((hbeam * 3)/2) * multiple)
	lat = False

	# If propagation < "back_lobe_factor" of max propagation then allow 360 degree (with bias)
	if int(ta) <= int(max_ta * back_lobe_factor):
		# allow 30% of the points to exsits in 360 degree space
		if random.random() < 0.3:
			uniform = random.uniform(1, 359)
			lat = math.asin(math.sin(cell_lat)*math.cos(ta/R) + \
					math.cos(cell_lat)*math.sin(ta/R) * \
					math.cos(math.radians(uniform)))

			lon = cell_lon + math.atan2(math.sin(math.radians(uniform)) * \
					math.sin(ta/R)*math.cos(cell_lat), math.cos(ta/R) - math.sin(cell_lat) * \
					math.sin(lat))

		# 70% of data in max +/- 1.5 antenna horizontal beamwidth space
		# (reducing as points move further from the cell)
		else:
			multiple = ta / (max_ta * back_lobe_factor)
			# Oval beam shape
			beam = ((hbeam * 3)/2) - (hbeam * multiple)

			uniform = random.uniform(azimuth - beam, azimuth + beam)
			lat = math.asin(math.sin(cell_lat) * math.cos(ta/R) + \
					math.cos(cell_lat) * math.sin(ta/R) * \
					math.cos(math.radians(uniform)))

			lon = cell_lon + math.atan2(math.sin(math.radians(uniform)) * \
					math.sin(ta/R) * math.cos(cell_lat), math.cos(ta/R) - math.sin(cell_lat) * \
					math.sin(lat))

	# Side lobes if ta <= "side_lobe_factor" of "max_ta"
	elif int(ta) <= int(max_ta * side_lobe_factor):
		# Oval beam shape
		multiple = ta / (max_ta * side_lobe_factor)
		beam = ((hbeam * 3)/2) - (((hbeam * 3)/2) * multiple)
		# Assuption is that ration will be 25% traffic per side lobe, and 50% traffic in main lobe for 
		# data points in "back_lobe_factor" to "side_lobe_factor range"
		if random.random() <= 0.5:
			if random.random() <= 0.5:
				uniform = random.uniform((azimuth - hbeam) - beam, (azimuth - hbeam) + beam)
				lat = math.asin(math.sin(cell_lat) * math.cos(ta/R) + \
						math.cos(cell_lat) * math.sin(ta/R) * \
						math.cos(math.radians(uniform)))

				lon = cell_lon + math.atan2(math.sin(math.radians(uniform)) * \
						math.sin(ta/R) * math.cos(cell_lat), math.cos(ta/R) - math.sin(cell_lat) * \
						math.sin(lat))
			else:
				uniform = random.uniform((azimuth + hbeam) - beam, (azimuth + hbeam) + beam)
				lat = math.asin(math.sin(cell_lat) * math.cos(ta/R) + \
						math.cos(cell_lat) * math.sin(ta/R) * \
						math.cos(math.radians(uniform)))

				lon = cell_lon + math.atan2(math.sin(math.radians(uniform)) * \
						math.sin(ta/R) * math.cos(cell_lat), math.cos(ta/R) - math.sin(cell_lat) * \
						math.sin(lat))

	# Main lobe
	if lat == False:
		multiple = min(random.uniform(0.9, 1), ta / max_ta)
		# Oval beam shape
		beam = ((hbeam * 3)/2) - (((hbeam * 3)/2) * multiple)
		uniform = random.uniform(azimuth - beam, azimuth + beam)
		lat = math.asin(math.sin(cell_lat)*math.cos(ta/R) + \
					math.cos(cell_lat)*math.sin(ta/R) * \
					math.cos(math.radians(uniform)))

		lon = cell_lon + math.atan2(math.sin(math.radians(uniform)) * \
					math.sin(ta/R) * math.cos(cell_lat), math.cos(ta/R) - math.sin(cell_lat) * \
					math.sin(lat))

	return pd.Series([math.degrees(lat), math.degrees(lon)])

def findBearing(pos_lat, pos_lon, lat_cell, lon_cell, propagation):
	'''
	This function returns;
		- Angle from cell to centroid of the bin
	'''
	lat1 = math.radians(lat_cell)
	lon1 = math.radians(lon_cell)
	lat2 = math.radians(pos_lat)
	lon2 = math.radians(pos_lon)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	angle = math.degrees(math.atan2(dlat, dlon))
	if angle <= 0:
		angle = math.fabs(angle) + 90
	elif (angle > 0) & (angle <= 90):
		angle = 90 - angle
	else:
		angle = 450 - angle
	return angle


def calculate_rsrp():

	rsrp = - 40 - (100 * ((x.distance_to_cell * (100 / x.bearing_weight)) / x.max_ta))

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

	return angle

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
	pm_path = "./../data/input-data/vf-ie/pm/"
	gis_path = "./../data/input-data/vf-ie/gis/"
	output_path = "./../data/output-data/vf-ie/grid/"
	##########################################################

	# Load pm (TA)
	try:
		print("Load TA pm data..")
		ta_density_df = pd.read_csv("{}ta-density.csv".format(pm_path), nrows=50)
	except:
		print("Issue loading loading TA pm data, exiting program..")
		sys.exit(0)

	# Get max TA
	try:
		print("Get max TA..")
		ta_density_df['max_ta'] = ta_density_df.apply(lambda x: getMaxTa(x), axis = 1)
	except:
		print("Issue getting max TA, exiting program..")
		sys.exit(0)

	# Explode data
	try:
		print("Explode data..")
		ta_density_df = explodeData(ta_density_df)
	except:
		print("Issue exploding data, exiting program..")
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

		print("\tRows pre GIS merge = {}".format(ta_density_df.shape[0]))

		gis_df['cilac'] = gis_df['cilac'].astype(int).astype(str)
		ta_density_df['eci'] = ta_density_df['eci'].astype(int).astype(str)

		ta_density_df = pd.merge(ta_density_df,gis_df,how='inner',left_on = 'eci', right_on = 'cilac')
		print("\tRows post GIS merge = {}".format(ta_density_df.shape[0]))

	except:
		print("\nIssue importing GIS data, exiting program..")
		sys.exit(0)

	# Calculate LAT/LON position
	try:
		print("\nCalculate LAT/LON position..")
		ta_density_df[['lat', 'lon']] = ta_density_df.apply(lambda x: calculateLatLon(math.radians(x.Latitude), math.radians(x.Longitude), x.ta, x.max_ta, 1, x.HBW, x.Bearing), axis = 1)

	except:
		print("\nIssue calculating LAT/LON position, exiting program..")
		sys.exit(0)

	
	
	
	ta_density_df['bearing_to_centroid'] = ta_density_df.apply(lambda x: findBearing(x['lat'], x['lon'], x['Latitude'], x['Longitude'], x['max_ta']), axis = 1)

	ta_density_df['bearing_weight'] = ta_density_df.apply(lambda x: 100 - 0.5 * (math.fabs(x.Bearing - x.bearing_to_centroid) % 180), axis = 1)

	ta_density_df['rsrp'] = ta_density_df.apply(lambda x: - 40 - (100 * ((x.ta * (100 / x.bearing_weight)) / x.max_ta) * math.exp(-0.0004316 * x.ta)), axis = 1)

	print(ta_density_df.head())
	sys.exit(0)


	# Create geodataframe
	#try:
	print("\nCreate GeoDataFrame..")
	ta_density_gdf = gpd.GeoDataFrame(ta_density_df, geometry=gpd.points_from_xy(ta_density_df.lon, ta_density_df.lat))
	ta_density_gdf = ta_density_gdf.set_crs(4326)
	#except:
	#	print("\nIssue creating GeoDataFrame, exiting program..")
	#	sys.exit(0)

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