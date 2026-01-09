import math

def polysTriangle(lat, lon, bearing, hbeam):
	"""
	Function creates polygons for all cells, if "H_BEAM" = 360 then create circle
	"""
	R = 6378.1
	lat1 = math.radians(lat)
	lon1 = math.radians(lon)

	# Size cell based on 'tech'
	d = 0.5

	# Create LAT/LON points
	lat2 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing - (hbeam / 2))))

	lon2 = lon1 + math.atan2(math.sin(math.radians(bearing - (hbeam / 2)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

	lat3 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing - (3 * hbeam / 8))))

	lon3 = lon1 + math.atan2(math.sin(math.radians(bearing - (3 * hbeam / 8)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat3))

	lat4 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing - (hbeam / 4))))

	lon4 = lon1 + math.atan2(math.sin(math.radians(bearing - (hbeam / 4)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat4))

	lat5 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing)))

	lon5 = lon1 + math.atan2(math.sin(math.radians(bearing))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat5))

	lat6 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing + (hbeam / 4))))

	lon6 = lon1 + math.atan2(math.sin(math.radians(bearing + (hbeam / 4)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat6))

	lat7 = math.asin(math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing + (3 * hbeam / 8))))

	lon7 = lon1 + math.atan2(math.sin(math.radians(bearing + (3 * hbeam / 8)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat7))

	lat8 = math.asin( math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing + (hbeam / 2))))

	lon8 = lon1 + math.atan2(math.sin(math.radians(bearing + (hbeam / 2)))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat8))

	lat9 = math.asin( math.sin(lat1)*math.cos(d/R) +
					math.cos(lat1)*math.sin(d/R)*math.cos(math.radians(bearing + 180)))

	lon9 = lon1 + math.atan2(math.sin(math.radians(bearing + 180))*math.sin(d/R)*math.cos(lat1),
					math.cos(d/R)-math.sin(lat1)*math.sin(lat9))

	# RADIANS to DEGREES
	lat1 = math.degrees(lat1)
	lon1 = math.degrees(lon1)
	lat2 = math.degrees(lat2)
	lon2 = math.degrees(lon2)
	lat3 = math.degrees(lat3)
	lon3 = math.degrees(lon3)
	lat4 = math.degrees(lat4)
	lon4 = math.degrees(lon4)
	lat5 = math.degrees(lat5)
	lon5 = math.degrees(lon5)
	lat6 = math.degrees(lat6)
	lon6 = math.degrees(lon6)
	lat7 = math.degrees(lat7)
	lon7 = math.degrees(lon7)
	lat8 = math.degrees(lat8)
	lon8 = math.degrees(lon8)
	lat9 = math.degrees(lat9)
	lon9 = math.degrees(lon9)

	# If not an omni antenna
	if hbeam < 360:
		return "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}))".format(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5, lon6, lat6, lon7, lat7, lon8, lat8, lon1, lat1)
	else:
		return "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}, {} {}))".format(lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5, lon6, lat6, lon7, lat7, lon8, lat8, lon9, lat9, lon2, lat2)
