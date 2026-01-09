import pandas as pd
import numpy as np
import geopandas as gpd
import math
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing, GeometryCollection
from shapely.affinity import scale as shp_scale
from shapely.ops import unary_union
from shapely import prepared
from shapely.geometry import box
from shapely.validation import make_valid
import geohash
from scipy.spatial import cKDTree
import pyproj
from pyproj import CRS
import sys

import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in intersects", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", message="invalid value encountered in simplify_preserve_topology", category=RuntimeWarning, module=r"shapely\..*",)

def clean_clamp_metrics(df):
	"""
	Clean metrics
		- Ensure all RSRP/RSRQ metrics are negative 
	Clamp metrics 
		- Ensure RSRP, RSRQ, SINR adhere to valid theoretical ranges
		- If a value falls outside, it’s replaced by the nearest valid boundary
	Bound Cells
		- Bound cells to 35km range
	"""

	# Make all RSRP/RSRQ values negative
	try:
		cols = ['avg_rsrp', 'avg_rsrq', 'avg_rsrp_grid', 'avg_rsrq_grid']
		df[cols] = -df[cols].abs()

		# Define valid ranges
		ranges = {
			'avg_rsrp':  (-144, -44),
			'avg_rsrq':  (-24, 0),
			'avg_sinr':  (-20, 30),
			'avg_rsrp_grid':  (-144, -44),
			'avg_rsrq_grid':  (-24, 0),
			'avg_sinr_grid':  (-20, 30)
		}

		for col, (min_val, max_val) in ranges.items():
			if col in df.columns:
				df[col] = df[col].clip(lower=min_val, upper=max_val)

		# Remove points further than 35km away
		df = df[df.distance_to_cell <= 35000]
		# Recalculate 'cell_max_distance_to_cell'
		df["cell_max_distance_to_cell"] = (
			df.groupby("cell_name")["distance_to_cell"].transform("max")
		)

		df[["cell_max_distance_to_cell", "avg_rsrp", "avg_rsrq", "avg_sinr", \
			"avg_rsrp_grid", "avg_rsrq_grid", "avg_sinr_grid"]].hist(figsize=(15, 10));

		return df

	except Exception as e:
		print(f"clean_clamp_metrics function failed: {e}")

def create_grid_geo_df(df):
	"""
	Get associated geohash centroid geometry and create a geodataframe
	"""
	try:
		# Decode all geohashes at once using list comprehension (much faster than apply)
		if 'grid' in df.columns:
			decoded = [geohash.decode(h) for h in df["grid"]]
		else:
			decoded = [geohash.decode(h) for h in df["geohash"]]
		# geohash.decode returns (lat, lon)

		# Extract lats/lons into NumPy arrays (optional for clarity)
		lats = [lat for lat, lon in decoded]
		lons = [lon for lat, lon in decoded]

		# Build geometry directly — no string conversions
		df["geometry"] = [Point(lon, lat) for lat, lon in decoded]

		# Convert to GeoDataFrame
		gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
		return gdf

	except Exception as e:
		print(f"create_grid_geo_df function failed: {e}")

def create_convex_hulls(gdf, gis_df):
	"""
	Create convex-hulls of potential cell coverage
	"""
	try:
		gdf_m = gdf[["cell_name", "distance_to_cell", "geometry"]].copy()
		proj_crs = gdf.estimate_utm_crs()
		gdf_m = gdf_m.to_crs(proj_crs)

		# Add cell origin so that convex-hull in worst case will start at the cell
		gis_df_add = gis_df[gis_df.Name.isin(gdf_m.cell_name.to_list())][["Name", "geometry"]].copy()
		gis_df_add.rename(columns = {"Name" : "cell_name"}, inplace = True)
		gis_df_add["distance_to_cell"] = 0.0
		gis_df_add = gis_df_add.to_crs(proj_crs)

		gdf_m["pct_rank"] = gdf_m.groupby("cell_name")["distance_to_cell"].rank(method="first", pct=True)

		# keep the top 98% *closest* samples per cell (i.e., smallest distances)
		gdf_m = gdf_m[gdf_m["pct_rank"] <= 0.98].drop(columns="pct_rank")

		gdf_m = pd.concat([gdf_m, gis_df_add], ignore_index=True)
		# 2) Make hulls per cell
		hulls_m = (
			gdf_m.groupby('cell_name')['geometry']
				.apply(lambda s: s.unary_union.convex_hull)
				.reset_index()
		)

		hulls_m = gpd.GeoDataFrame(hulls_m, geometry='geometry', crs=proj_crs)

		# 3) (Optional) compute area in km²
		hulls_m['area_km2'] = hulls_m.area / 1e6

		# 4) Reproject back to WGS84 for web maps
		hulls = hulls_m.to_crs(4326)

		return hulls_m

	except Exception as e:
		print(f"create_convex_hulls function failed: {e}")

def clean_hull_hashes_gdf(hull_hashes_gdf, grid_geo_data, gis_df):
	try:
		# CHANGE BACK, QUICK FIX FOR VF
		gis_df["CILAC"] = gis_df["CILAC"].astype(str).str[1:].astype(int)
		hull_hashes_gdf.rename(columns = {'geohash': 'grid'}, inplace = True)
		hull_hashes_gdf = hull_hashes_gdf.merge(gis_df[['CILAC', 'Name', 'Bearing', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'left')
		hull_hashes_gdf['grid_cell'] = hull_hashes_gdf['grid'].astype(str) + "_" + hull_hashes_gdf['CILAC'].astype(str).str.replace('.0', '')
		hull_hashes_gdf = hull_hashes_gdf.merge(grid_geo_data[['grid_cell', 'avg_rsrp']], on = 'grid_cell', how = 'left')
		hull_hashes_gdf.drop(columns = ['Name', 'CILAC'], inplace = True)
		return hull_hashes_gdf

	except Exception as e:
		print(f"clean_hull_hashes_gdf function failed: {e}")


def calculate_distances(df):
	try:
		mask = df["geometry"].notna() & df["Latitude"].notna() & df["Longitude"].notna()

		lon1 = df.loc[mask, "geometry"].x.to_numpy()
		lat1 = df.loc[mask, "geometry"].y.to_numpy()
		lon2 = df.loc[mask, "Longitude"].to_numpy()
		lat2 = df.loc[mask, "Latitude"].to_numpy()

		# Geodesic (WGS84 ellipsoid)
		geod = pyproj.Geod(ellps="WGS84")
		_, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)  # returns fwd_az, back_az, distance (m)

		# Write result
		df["distance_to_cell"] = np.nan
		df.loc[mask, "distance_to_cell"] = dist_m

		df['cell_max_distance_to_cell'] = (
			df.groupby('cell_name')['distance_to_cell'].transform('max')
		)
		return df

	except Exception as e:
		print(f"calculate_distances function failed: {e}")

def save_file(df, file_name, OUTPUT_PATH, type):
	try:
		print(f"\tSave dataframe to file {file_name}.{type}..")
		print("\tEnsure grid geometry is polygon..")
		arr = df["grid"].to_numpy()
		geoms = []
		for gh in arr:
			b = geohash.bbox(gh)          # {'n','s','e','w'}
			geoms.append(box(b['w'], b['s'], b['e'], b['n']))  # lon/lat order

		df["geometry"] = geoms
		
		if type == 'csv':
			df.to_csv(OUTPUT_PATH / f"{file_name}.{type}", index = False)
			print("\tSave complete..")
		elif type == 'shp':
			df.to_file(OUTPUT_PATH / f"{file_name}.{type}", index = False)
			print("\tSave complete..")
		else:
			print(f"\tUnknown file type {type}, file not saved..")
	except Exception as e:
		print(f"save_file_to_csv function failed: {e}")	

def create_tilt_files(df, max_dist_col, avg_rsrp_col):
	try:
		base_cols = ['grid_cell', 'grid', 'cell_name', 'distance_to_cell', 'Scr_Freq', 'UARFCN', 'Bearing', 'Latitude', 'Longitude', \
			'TiltE', 'TiltM', 'SiteID', 'City', 'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', 'RF_Team', 'geometry']

		required = base_cols + [max_dist_col, avg_rsrp_col]
		df = df[required].copy()
		df.rename(columns = {max_dist_col : 'cell_max_distance_to_cell', avg_rsrp_col : 'avg_rsrp'}, inplace = True)
		df = df[~df.avg_rsrp.isna()].reset_index()
		df = df.set_crs("EPSG:4326")
		return df
	except Exception as e:
		print(f"create_downtilt_files function failed: {e}")

def build_extended_grid(cell_dist_metrics, gis_df):
	try:
		print("\tBuild extended grids..")
		cell_dist_metrics.rename(columns = {'cell_max_distance_to_cell' : 'max_ta'}, inplace = True)
		extended_grid_df = build_new_grids(cell_dist_metrics, gis_df, n_samples=1000, seed=42)
		extended_grid_df['grid'] = extended_grid_df.apply(lambda r: geohash.encode(r['lat'], r['lon'], precision=7), axis=1)
		# Remove duplicate cell-grid pairs
		extended_grid_df = extended_grid_df.drop_duplicates(subset=['cell_name', 'grid'], keep='first')
		extended_grid_gdf = create_grid_geo_df(extended_grid_df)

		extended_grid_gdf.rename(columns = {'ta' :'distance_to_cell'}, inplace = True)
		extended_grid_gdf["avg_rsrp"] = np.nan

		extended_grid_gdf = extended_grid_gdf.merge(gis_df[['Name', 'CILAC', 'Scr_Freq', 'UARFCN', 'Bearing', 'TiltE', 'TiltM', 'SiteID', 'City', \
															'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', \
															'RF_Team']], left_on = 'cell_name', right_on = 'Name', how = 'left')

		extended_grid_gdf['grid_cell'] = extended_grid_gdf['grid'].astype(str) + "_" + extended_grid_gdf['CILAC'].astype(str).str.replace('.0', '')

		extended_grid_gdf.drop(columns = ['lat', 'lon', 'Name', 'CILAC'], inplace = True)

		# Split into 2 datframes 'extended_grid_1_ut_gdf' and 'extended_grid_2_ut_gdf'
		extended_grid_1_ut_gdf = extended_grid_gdf.merge(cell_dist_metrics[['cell_name', 'max_dist_1_ut']], on = 'cell_name', how = 'inner')
		extended_grid_1_ut_gdf = extended_grid_1_ut_gdf[extended_grid_1_ut_gdf['distance_to_cell'] <= extended_grid_1_ut_gdf['max_dist_1_ut']].copy()
		extended_grid_1_ut_gdf.rename(columns = {'max_dist_1_ut' : 'cell_max_distance_to_cell'}, inplace = True)
		
		extended_grid_2_ut_gdf = extended_grid_gdf.merge(cell_dist_metrics[['cell_name', 'max_dist_2_ut']], on = 'cell_name', how = 'inner')
		extended_grid_2_ut_gdf = extended_grid_2_ut_gdf[extended_grid_2_ut_gdf['distance_to_cell'] <= extended_grid_2_ut_gdf['max_dist_2_ut']].copy()
		extended_grid_2_ut_gdf.rename(columns = {'max_dist_2_ut' : 'cell_max_distance_to_cell'}, inplace = True)
		print("\tExtended grids built..")
		return extended_grid_1_ut_gdf, extended_grid_2_ut_gdf
	except Exception as e:
		print(f"build_extended_grid function failed: {e}")

def build_extended_hulls_grid(cell_dist_metrics, gis_df, hulls):
	try:
		print("\tBuild extended hull grids..")
		print("\tGet hull extension +1 degree uptilt..")
		cell_dist_metrics['dist_increase_ut'] = cell_dist_metrics['max_dist_1_ut'] - cell_dist_metrics['cell_max_distance_to_cell']
		hulls = hulls.merge(gis_df[['Name', 'Latitude', 'Longitude']], left_on = 'cell_name', right_on = 'Name', how = 'inner')
		hulls = hulls.merge(cell_dist_metrics[['cell_name', 'dist_increase_ut']], on = 'cell_name', how = 'inner')
		hulls_1_ut = expand_hulls_weighted_by_pivot_rowwise(hulls)
		print("\tGet geohashes under extended hulls..")
		hull_hashes_1_ut = geohash7_inside_hulls_fast(hulls_1_ut, precision=7, geometry_mode="cell", simplify_tolerance_m=10)
		hull_hashes_1_ut = hull_hashes_1_ut[['cell_name', 'geohash']]
		print("\tGet geohashes geometries..")
		hull_hashes_1_ut = create_grid_geo_df(hull_hashes_1_ut)
		
		
		print("\tGet hull extension +2 degree uptilt..")
		hulls.drop(columns = ['Name', 'dist_increase_ut'], inplace = True)
		cell_dist_metrics['dist_increase_ut'] = cell_dist_metrics['max_dist_2_ut'] - cell_dist_metrics['cell_max_distance_to_cell']
		hulls = hulls.merge(cell_dist_metrics[['cell_name', 'dist_increase_ut']], on = 'cell_name', how = 'inner')
		hulls_2_ut = expand_hulls_weighted_by_pivot_rowwise(hulls)	
		print("\tGet geohashes under extended hulls..")
		hull_hashes_2_ut = geohash7_inside_hulls_fast(hulls_2_ut, precision=7, geometry_mode="cell", simplify_tolerance_m=10)
		hull_hashes_2_ut = hull_hashes_2_ut[['cell_name', 'geohash']]
		print("\tGet geohashes geometries..")
		hull_hashes_2_ut = create_grid_geo_df(hull_hashes_2_ut)
		

		############################
		### Add required columns ###
		############################
		# Add 'distance_to_cell'
		#calculate_distances

		# Add 'cell_max_distance_to_cell'
		hull_hashes_1_ut = hull_hashes_1_ut.merge(cell_dist_metrics[['cell_name', 'max_dist_1_ut']], on = 'cell_name', how = 'inner')
		hull_hashes_1_ut.rename(columns = {'max_dist_1_ut' : 'cell_max_distance_to_cell'}, inplace = True)
		hull_hashes_2_ut = hull_hashes_2_ut.merge(cell_dist_metrics[['cell_name', 'max_dist_2_ut']], on = 'cell_name', how = 'inner')		
		hull_hashes_2_ut.rename(columns = {'max_dist_2_ut' : 'cell_max_distance_to_cell'}, inplace = True)
		
		hull_hashes_1_ut["avg_rsrp"] = np.nan
		hull_hashes_2_ut["avg_rsrp"] = np.nan

		hull_hashes_1_ut = hull_hashes_1_ut.merge(gis_df[['Name', 'CILAC', 'Scr_Freq', 'UARFCN', 'Bearing', 'TiltE', 'TiltM', 'SiteID', 'City', \
															'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', \
															'RF_Team']], left_on = 'cell_name', right_on = 'Name', how = 'left')

		hull_hashes_2_ut = hull_hashes_2_ut.merge(gis_df[['Name', 'CILAC', 'Scr_Freq', 'UARFCN', 'Bearing', 'TiltE', 'TiltM', 'SiteID', 'City', \
															'Height', 'TAC', 'Band', 'Vendor', 'MaxTransPwr', 'FreqMHz', 'HBW', \
															'RF_Team']], left_on = 'cell_name', right_on = 'Name', how = 'left')

		hull_hashes_1_ut.rename(columns = {'geohash' : 'grid'}, inplace = True)
		hull_hashes_2_ut.rename(columns = {'geohash' : 'grid'}, inplace = True)
		
		hull_hashes_1_ut['grid_cell'] = hull_hashes_1_ut['grid'].astype(str) + "_" + hull_hashes_1_ut['CILAC'].astype(str).str.replace('.0', '')
		hull_hashes_2_ut['grid_cell'] = hull_hashes_2_ut['grid'].astype(str) + "_" + hull_hashes_2_ut['CILAC'].astype(str).str.replace('.0', '')

		hull_hashes_1_ut.drop(columns = ['Name', 'CILAC'], inplace = True) # Removed 'lat', 'lon'
		hull_hashes_2_ut.drop(columns = ['Name', 'CILAC'], inplace = True) # Removed 'lat', 'lon'

		print("\tExtended grids built..")
		return hull_hashes_1_ut, hull_hashes_2_ut
	except Exception as e:
		print(f"build_extended_grid function failed: {e}")

######################################################################
### Functions Related to finding geohashes within the convex hulls ###
######################################################################

def _gh_cell_size(lat: float, lon: float, precision: int):
	b = geohash.bbox(geohash.encode(lat, lon, precision))
	return (b["n"] - b["s"], b["e"] - b["w"])  # (dlat, dlon in degrees)

def _aligned_centers_1d(min_v: float, max_v: float, center0: float, step: float, up: bool):
	"""
	Build a 1D array of geohash cell centers covering [min_v, max_v],
	aligned to the first cell center 'center0' and spaced by 'step'.
	If up=True: produce decreasing sequence from top (for latitude).
	"""
	if up:
		# go up to the top then step downward
		offs_up = int(np.ceil((max_v - center0) / step))
		v_top = center0 + offs_up * step
		offs_dn = int(np.ceil((v_top - min_v) / step))
		n = offs_dn + 1
		return v_top - np.arange(n) * step
	else:
		# go down to the left then step rightward
		offs_left = int(np.ceil((center0 - min_v) / step))
		v_left = center0 - offs_left * step
		offs_rt = int(np.ceil((max_v - v_left) / step))
		n = offs_rt + 1
		return v_left + np.arange(n) * step

def _cover_hull_geohash_fast(hull, precision: int = 7):
	"""
	Return a set of geohash ids (precision) whose cells intersect the hull.
	"""
	minx, miny, maxx, maxy = hull.bounds
	# approximate cell size at hull center
	c_lat = (miny + maxy) / 2.0
	c_lon = (minx + maxx) / 2.0
	dlat, dlon = _gh_cell_size(c_lat, c_lon, precision)

	# align to geohash grid: get cell covering (miny, minx) and (maxy, minx)
	b_ll = geohash.bbox(geohash.encode(miny, minx, precision))
	b_ul = geohash.bbox(geohash.encode(maxy, minx, precision))
	lon_center0 = (b_ll["w"] + b_ll["e"]) / 2.0
	lat_center0 = (b_ul["s"] + b_ul["n"]) / 2.0  # start near the top row

	lon_centers = _aligned_centers_1d(minx, maxx, lon_center0, dlon, up=False)
	lat_centers = _aligned_centers_1d(miny, maxy, lat_center0, dlat, up=True)

	# prepared hull for fast intersects
	prep = prepared.prep(hull)
	out = set()
    
	# Sweep rows (lat) × cols (lon)
	for lat in lat_centers:
		for lon in lon_centers:
			gh = geohash.encode(float(lat), float(lon), precision)
			b = geohash.bbox(gh)
			# quick bbox reject vs hull bbox
			if b["e"] < minx or b["w"] > maxx or b["n"] < miny or b["s"] > maxy:
				continue
			cell = box(b["w"], b["s"], b["e"], b["n"])
			if prep.intersects(cell):
				out.add(gh)
	return out

def geohash7_inside_hulls_fast(
								hulls_gdf: gpd.GeoDataFrame,
								precision: int = 7,
								geometry_mode: str = "cell",      # "cell" -> geohash cell polygon, "hull" -> repeat hull
								simplify_tolerance_m: float | None = 10.0  # simplify in meters for speed; set None to disable
								) -> gpd.GeoDataFrame:
	
	"""
	Input: hulls_gdf with columns ['cell_name','geometry'] in EPSG:4326.
	Output: GeoDataFrame with rows per (cell_name, geohash{precision}),
		columns: ['cell_name','geohash', 'geometry'].
	"""
	assert "cell_name" in hulls_gdf and "geometry" in hulls_gdf, "Expect columns: cell_name, geometry"

	gdf = hulls_gdf
	# ensure WGS84
	if gdf.crs is None or gdf.crs.to_epsg() != 4326:
		gdf = gdf.to_crs(4326)

	# optional simplify in meters to accelerate intersects
	if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
		gdf_simpl = gdf.to_crs(3857).copy()
		gdf_simpl["geometry"] = gdf_simpl.geometry.simplify(simplify_tolerance_m, preserve_topology=True)
		gdf = gdf_simpl.to_crs(4326)

	rows = []
	for cell_name, hull in zip(gdf["cell_name"], gdf.geometry):
		if hull.is_empty:
			continue
		gh_set = _cover_hull_geohash_fast(hull, precision=precision)
		for gh in gh_set:
			if geometry_mode == "cell":
				b = geohash.bbox(gh)
				geom = box(b["w"], b["s"], b["e"], b["n"])
			else:
				geom = hull
			rows.append({"cell_name": cell_name, "geohash": gh, "geometry": geom})

	return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)

######################################################################

######################################################################################
### Functions Related to Calculating RSRP from surrounding bins and decay function ###
######################################################################################


##########################
# ---- Tunables ----
FALLBACK_ALPHA = 35.0          # dB/decade (~ n≈3.5)
CLAMP_RSRP     = (-144.0, -44.0)
IDW_POWER      = 2
K_SAME_CELL    = 8
K_GLOBAL       = 24
BORE_SIGMA_DEG = 35.0
DIR_EXP        = 1.0
EARTH_R        = 6371000.0     # meters

# ---- Helpers (WGS84) ----
def _to_rad(x): 
	return np.deg2rad(x.astype(float))

def _haversine_m(lat1, lon1, lat2, lon2):
	"""Vectorized haversine distance in meters."""
	lat1 = np.asarray(lat1, dtype=float); lon1 = np.asarray(lon1, dtype=float)
	lat2 = np.asarray(lat2, dtype=float); lon2 = np.asarray(lon2, dtype=float)
	dlat = np.radians(lat2 - lat1)
	dlon = np.radians(lon2 - lon1)
	a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2.0)**2
	c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
	return EARTH_R * c

def _bearing_deg(lat1, lon1, lat2, lon2):
	"""Initial bearing (0°=north, clockwise) from (lat1,lon1) to (lat2,lon2)."""
	φ1, λ1, φ2, λ2 = map(np.radians, [lat1, lon1, lat2, lon2])
	dλ = λ2 - λ1
	y = np.sin(dλ) * np.cos(φ2)
	x = np.cos(φ1)*np.sin(φ2) - np.sin(φ1)*np.cos(φ2)*np.cos(dλ)
	θ = np.degrees(np.arctan2(y, x))
	return (θ + 360.0) % 360.0

def _ang_diff_deg(a, b):  # minimal signed diff in [-180,180]
	return (a - b + 180.0) % 360.0 - 180.0

def _dir_weight(boresight_deg, bearing_deg, sigma_deg=BORE_SIGMA_DEG, exp=DIR_EXP):
	if boresight_deg is None or not np.isfinite(boresight_deg): return 1.0
	d = _ang_diff_deg(boresight_deg, bearing_deg)
	w = np.exp(-0.5 * (d / max(1e-3, sigma_deg))**2)
	return w**exp

def _log10_safe(x, eps=1.0):
	return np.log10(np.maximum(np.asarray(x, dtype=float), eps))

def _ols_alpha(dist_m, rsrp, fallback=FALLBACK_ALPHA):
	dist_m = np.asarray(dist_m, float); rsrp = np.asarray(rsrp, float)
	m = np.isfinite(dist_m) & np.isfinite(rsrp) & (dist_m > 0)
	if m.sum() < 3: return float(fallback)
	x = _log10_safe(dist_m[m]); y = rsrp[m]
	if x.max() - x.min() < 1e-6: return float(fallback)
	xm, ym = x.mean(), y.mean()
	b = ((x - xm)*(y - ym)).sum() / ((x - xm)**2).sum()
	alpha = -b
	return float(np.clip(alpha, 20.0, 50.0))

def _project_rsrp(rsrp_src, alpha, d_src, d_tgt):
	d_src = np.maximum(np.asarray(d_src, float), 1.0)
	d_tgt = np.maximum(np.asarray(d_tgt, float), 1.0)
	return rsrp_src - alpha * (_log10_safe(d_tgt) - _log10_safe(d_src))

def _safe_normed(w):
	w = np.asarray(w, float); w[~np.isfinite(w)] = 0.0
	s = w.sum()
	return w / s if s > 0 else np.zeros_like(w)

def predict_grid_rsrp_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""
	Requires columns:
		'cell_name','grid','geometry'(Point WGS84 grid centroid),
		'grid_cell','Latitude','Bearing','Longitude','avg_rsrp'
	Keeps known avg_rsrp; predicts only where avg_rsrp is NaN.
	Returns original gdf with 'pred_rsrp' (filled) and 'rsrp_final'.
	"""
	try:
		needed = {'cell_name','grid','geometry','grid_cell','Latitude','Bearing','Longitude','avg_rsrp'}
		if not needed.issubset(gdf.columns):
			raise ValueError(f"Missing required columns: {needed - set(gdf.columns)}")
		if gdf.crs is None:
			gdf = gdf.set_crs("EPSG:4326")

		# Cells table (unique per cell)
		cells = (
			gdf[['cell_name','Latitude','Longitude','Bearing']]
			.dropna(subset=['Latitude','Longitude'])
			.drop_duplicates(subset=['cell_name'])
			.copy()
		)
		cell_lat = dict(zip(cells['cell_name'], cells['Latitude'].astype(float)))
		cell_lon = dict(zip(cells['cell_name'], cells['Longitude'].astype(float)))
		cell_brg = dict(zip(cells['cell_name'], cells['Bearing'].astype(float)))

		# Grids (unique by grid)
		grids = gdf[['grid','grid_cell','cell_name','geometry']].drop_duplicates(subset=['grid']).copy()
		# ensure Points
		if not grids.geometry.geom_type.isin(['Point']).all():
			grids['geometry'] = grids.geometry.centroid
		grids['grid_lat'] = grids.geometry.y.astype(float)
		grids['grid_lon'] = grids.geometry.x.astype(float)
		grids['cell_lat'] = grids['cell_name'].map(cell_lat)
		grids['cell_lon'] = grids['cell_name'].map(cell_lon)

		# Distances grid->its cell
		grids['dist_to_cell_m'] = _haversine_m(grids['grid_lat'], grids['grid_lon'],
		                                       grids['cell_lat'], grids['cell_lon'])
		grids['dist_to_cell_m'] = np.maximum(grids['dist_to_cell_m'], 1.0)

		# Bins = rows with observed avg_rsrp
		bins = gdf.dropna(subset=['avg_rsrp']).copy()
		if not bins.geometry.geom_type.isin(['Point']).all():
			bins['geometry'] = bins.geometry.centroid
		bins['bin_lat'] = bins.geometry.y.astype(float)
		bins['bin_lon'] = bins.geometry.x.astype(float)
		bins['cell_lat'] = bins['cell_name'].map(cell_lat)
		bins['cell_lon'] = bins['cell_name'].map(cell_lon)

		# Distances bin->its cell
		bins['dist_to_cell_m'] = _haversine_m(bins['bin_lat'], bins['bin_lon'],
		                                      bins['cell_lat'], bins['cell_lon'])
		bins['dist_to_cell_m'] = np.maximum(bins['dist_to_cell_m'], 1.0)

		# Per-cell slopes from bins
		alphas = {}
		for cid, dfc in bins.groupby('cell_name'):
			alphas[cid] = _ols_alpha(dfc['dist_to_cell_m'].values, dfc['avg_rsrp'].values, fallback=FALLBACK_ALPHA)
		default_alpha = float(FALLBACK_ALPHA)

		# KDTree on (lat, lon) for neighbor lookup (approx for selection only)
		# Use radians to reduce anisotropy
		b_lat_rad = _to_rad(bins['bin_lat'])
		b_lon_rad = _to_rad(bins['bin_lon'])
		tree = cKDTree(np.c_[b_lat_rad, b_lon_rad])

		b_vals   = bins['avg_rsrp'].values
		b_dcell  = bins['dist_to_cell_m'].values
		b_cellid = bins['cell_name'].values
		b_lat    = bins['bin_lat'].values
		b_lon    = bins['bin_lon'].values

		# Predict per unique grid
		g_lat = grids['grid_lat'].values
		g_lon = grids['grid_lon'].values
		g_cellid = grids['cell_name'].values
		g_dcell  = grids['dist_to_cell_m'].values

		preds = np.full(grids.shape[0], np.nan, float)

		for i in range(grids.shape[0]):
			if b_vals.size == 0:
				break

			# neighbor query
			kq = min(K_GLOBAL, b_vals.size)
			dists, nn_idx = tree.query([np.deg2rad(g_lat[i]), np.deg2rad(g_lon[i])], k=kq)
			if np.isscalar(dists):
				dists = np.array([dists]); nn_idx = np.array([nn_idx])

			same_mask = (b_cellid[nn_idx] == g_cellid[i])
			same_idx = nn_idx[same_mask]
			other_idx = nn_idx[~same_mask]

			take_same = same_idx[:K_SAME_CELL]
			need = max(0, K_SAME_CELL - take_same.size)
			take_other = other_idx[:need] if need > 0 else np.array([], dtype=int)
			chosen = np.concatenate([take_same, take_other]) if take_other.size else take_same
			if chosen.size == 0:
				continue

			nb_idx = chosen
			nb_rsrp = b_vals[nb_idx]
			nb_bcell = b_cellid[nb_idx]
			nb_dcell = b_dcell[nb_idx]

			# true great-circle distances grid <-> bin (for IDW weights)
			nb_dxy = _haversine_m(g_lat[i], g_lon[i], b_lat[nb_idx], b_lon[nb_idx])

			# distance grid -> neighbor's cell, plus azimuth weighting
			d_grid_to_nbcell = np.empty(nb_idx.size, float)
			dir_w = np.ones(nb_idx.size, float)
			alpha_arr = np.empty(nb_idx.size, float)

			for j, cnb in enumerate(nb_bcell):
				alpha_arr[j] = alphas.get(cnb, default_alpha)
				clat = cell_lat.get(cnb, np.nan)
				clon = cell_lon.get(cnb, np.nan)
				if not np.isfinite(clat) or not np.isfinite(clon):
					d_grid_to_nbcell[j] = np.nan
					dir_w[j] = 1.0
				else:
					d_grid_to_nbcell[j] = _haversine_m(g_lat[i], g_lon[i], clat, clon)
					# azimuth weight (cell->grid bearing vs Bearing)
					brg = _bearing_deg(clat, clon, g_lat[i], g_lon[i])
					az = cell_brg.get(cnb, None)
					dir_w[j] = _dir_weight(az, brg, sigma_deg=BORE_SIGMA_DEG, exp=DIR_EXP)

			d_grid_to_nbcell = np.maximum(d_grid_to_nbcell, 1.0)
			same_nb = (nb_bcell == g_cellid[i])
			d_tgt = np.where(same_nb, g_dcell[i], d_grid_to_nbcell)

			rsrp_proj = _project_rsrp(nb_rsrp, alpha_arr, nb_dcell, d_tgt)

			w_dist = 1.0 / np.maximum(nb_dxy, 1.0)**IDW_POWER
			w = _safe_normed(w_dist * dir_w)
			preds[i] = np.sum(w * rsrp_proj)

		# Build per-grid prediction and clamp
		grid_pred = grids[['grid']].copy()
		grid_pred['pred_rsrp'] = np.clip(preds, CLAMP_RSRP[0], CLAMP_RSRP[1])

		# Merge back; keep known avg_rsrp, fill NaNs with prediction
		out = gdf.merge(grid_pred, on='grid', how='left')
		out['rsrp_final'] = out['avg_rsrp'].where(out['avg_rsrp'].notna(), out['pred_rsrp'])
		out['rsrp_final'] = out['rsrp_final'].clip(lower=CLAMP_RSRP[0], upper=CLAMP_RSRP[1])
		out = out[['cell_name', 'grid', 'grid_cell', 'Latitude', 'Longitude', 'Bearing', 'rsrp_final', 'geometry']]
		out.rename(columns = {'rsrp_final' : 'avg_rsrp'}, inplace = True)
		return out
	
	except Exception as e:
		print(f"predict_grid_rsrp_wgs84 function failed: {e}")

### Testing new function to replace predict_grid_rsrp_wgs84 - cellwise function
# ------------ helpers ------------
def _hav1toN(glat, glon, lats, lons):
	R = 6371000.0
	glat, glon = np.deg2rad(glat), np.deg2rad(glon)
	lats, lons = np.deg2rad(lats), np.deg2rad(lons)
	dlat = lats - glat
	dlon = lons - glon
	a = np.sin(dlat/2)**2 + np.cos(glat)*np.cos(lats)*np.sin(dlon/2)**2
	return 2*R*np.arcsin(np.sqrt(a))

def _fit_ab(dist_m, rsrp):
	x = np.log10(np.asarray(dist_m, float))
	y = np.asarray(rsrp, float)
	m = np.isfinite(x) & np.isfinite(y)
	x = x[m]; y = y[m]
	if x.size < 2:
		return np.nan, np.nan
	X = np.c_[x, np.ones_like(x)]
	a, b = np.linalg.lstsq(X, y, rcond=None)[0]
	return float(a), float(b)

def predict_grid_rsrp_wgs84_same_cell_only(
	gdf: gpd.GeoDataFrame,
	k_min: int = 3,
	idw_power: float = 2.0,
	clamp: tuple[float, float] | None = None
	) -> gpd.GeoDataFrame:

	needed = {'cell_name','grid','geometry','grid_cell','Latitude','Longitude','Bearing','avg_rsrp'}
	if not needed.issubset(gdf.columns):
		raise ValueError(f"Missing required columns: {sorted(needed - set(gdf.columns))}")

	df = gdf.copy()
	if df.crs is None:
		df = df.set_crs("EPSG:4326")

	# ---- per-cell metadata ----
	cells = (df[['cell_name','Latitude','Longitude']]
				.dropna(subset=['Latitude','Longitude'])
				.drop_duplicates('cell_name'))
	cell_lat = dict(zip(cells['cell_name'], cells['Latitude'].astype(float)))
	cell_lon = dict(zip(cells['cell_name'], cells['Longitude'].astype(float)))

	# ---- targets: one per (grid, cell_name) ----
	targets = (df[['grid','grid_cell','cell_name','geometry']]
				.drop_duplicates(['grid','cell_name'])
				.copy())
	if not targets.geometry.geom_type.isin(['Point']).all():
		targets['geometry'] = targets.geometry.centroid
	targets['grid_lat'] = targets.geometry.y.astype(float)
	targets['grid_lon'] = targets.geometry.x.astype(float)
	targets = targets.reset_index(drop=True)

	# Pre-extract arrays for speed/robust indexing
	glats  = targets['grid_lat'].to_numpy()
	glons  = targets['grid_lon'].to_numpy()
	cellsA = targets['cell_name'].to_numpy()
	gridsA = targets['grid'].to_numpy()

	# ---- observed bins ----
	bins = df.dropna(subset=['avg_rsrp']).copy()
	if not bins.geometry.geom_type.isin(['Point']).all():
		bins['geometry'] = bins.geometry.centroid
	bins['bin_lat'] = bins.geometry.y.astype(float)
	bins['bin_lon'] = bins.geometry.x.astype(float)
	bins['cell_lat'] = bins['cell_name'].map(cell_lat)
	bins['cell_lon'] = bins['cell_name'].map(cell_lon)

	dist = _hav1toN(bins['bin_lat'], bins['bin_lon'], bins['cell_lat'], bins['cell_lon'])
	bins['dist_to_cell_m'] = pd.Series(dist, index=bins.index).clip(lower=1.0)

	bins_by_cell = {cid: dfc for cid, dfc in bins.groupby('cell_name')}

	# per-cell model + medians
	cell_ab = {}
	cell_median = {}
	for cid, dfc in bins_by_cell.items():
		a, b = _fit_ab(dfc['dist_to_cell_m'], dfc['avg_rsrp'])
		cell_ab[cid] = (a, b)
		cell_median[cid] = float(dfc['avg_rsrp'].median())

	# global model
	a_g, b_g = _fit_ab(bins['dist_to_cell_m'], bins['avg_rsrp'])
	global_median = float(bins['avg_rsrp'].median())

	preds  = np.full(len(targets), np.nan, float)
	src    = np.full(len(targets), "", dtype=object)
	n_used = np.zeros(len(targets), dtype=int)

	# ---- prediction loop (use arrays, not .at/.loc) ----
	for i in range(len(targets)):
		grid = gridsA[i]
		cell = cellsA[i]
		glat = glats[i]
		glon = glons[i]

		clat = cell_lat.get(cell, np.nan)
		clon = cell_lon.get(cell, np.nan)
		if not (np.isfinite(clat) and np.isfinite(clon) and np.isfinite(glat) and np.isfinite(glon)):
			d_gc = np.nan
		else:
			d_gc = float(_hav1toN(glat, glon, np.array([clat]), np.array([clon])).item())
			d_gc = max(d_gc, 1.0)

		dfc = bins_by_cell.get(cell)
		if dfc is not None and not dfc.empty:
			blats = dfc['bin_lat'].to_numpy()
			blons = dfc['bin_lon'].to_numpy()
			brsrp = dfc['avg_rsrp'].to_numpy()

			if blats.size >= k_min and np.isfinite(glat) and np.isfinite(glon):
				dxy = _hav1toN(glat, glon, blats, blons)
				dxy = np.maximum(dxy, 1.0)
				w = 1.0 / (dxy ** idw_power)
				s = w.sum()
				if s > 0:
					w /= s
					preds[i]  = float(np.sum(w * brsrp))
					src[i]    = "same_cell_idw"
					n_used[i] = int(blats.size)
					continue

		# fallback 1: per-cell model
		a, b = cell_ab.get(cell, (np.nan, np.nan))
		if np.isfinite(a) and np.isfinite(b) and np.isfinite(d_gc):
			preds[i]  = a * np.log10(d_gc) + b
			src[i]    = "cell_model"
			n_used[i] = int(len(bins_by_cell.get(cell, [])))
			continue

		# fallback 2: global model
		if np.isfinite(a_g) and np.isfinite(b_g) and np.isfinite(d_gc):
			preds[i]  = a_g * np.log10(d_gc) + b_g
			src[i]    = "global_model"
			n_used[i] = 0
			continue

		# fallback 3: per-cell median, else global median
		m = cell_median.get(cell, np.nan)
		if np.isfinite(m):
			preds[i]  = m
			src[i]    = "cell_median"
			n_used[i] = int(len(bins_by_cell.get(cell, [])))
		else:
			preds[i]  = global_median
			src[i]    = "global_median"
			n_used[i] = 0

	# ---- assemble & fill ----
	target_pred = targets[['grid','cell_name']].copy()
	target_pred['pred_rsrp']    = preds
	target_pred['pred_source']  = src
	target_pred['n_bins_used']  = n_used

	out = df.merge(target_pred, on=['grid','cell_name'], how='left')
	out['avg_rsrp'] = out['avg_rsrp'].where(out['avg_rsrp'].notna(), out['pred_rsrp'])
	if clamp is not None:
		out['avg_rsrp'] = out['avg_rsrp'].clip(lower=clamp[0], upper=clamp[1])

	return out.drop(columns=['pred_rsrp'])

######################################################################################

##################################################################
### Functions Related to Calculating distance metrics per cell ###  
##################################################################
def _vertical_attenuation(theta_deg: float, alpha_deg: float, hpbw_v_deg: float, sla_v_db: float) -> float:
	"""3GPP-style parabolic attenuation in vertical plane (dB)."""
	return min(12.0 * (((theta_deg - alpha_deg) / hpbw_v_deg) ** 2), sla_v_db)

def estimate_distance_after_tilt(
	d_max_m: float,
	alpha_deg: float,
	h_m: float,
	hpbw_v_deg: float = 6.5,
	sla_v_db: float = 30.0,
	n: float = 3.5,
	delta_tilt_deg: float = 1.0,
	tilt_direction: str = "downtilt"
	):# -> Dict[str, Any]:
	"""
	Estimate the new max coverage distance after changing electrical tilt by delta_tilt_deg.
	Returns a dict with distances, percent change, and useful intermediates.
	"""
	if (alpha_deg == 0) and (tilt_direction == "uptilt"):
		return d_max_m, 0
	if d_max_m <= 0 or h_m < 0:
		raise ValueError("d_max_m must be > 0 and h_m must be >= 0")
	if hpbw_v_deg <= 0 or n <= 0:
		raise ValueError("hpbw_v_deg and n must be > 0")

	# Elevation angle from site to current edge user (deg)
	theta_e_deg = math.degrees(math.atan2(h_m, d_max_m))

	# Attenuation before/after at that direction
	A_before = _vertical_attenuation(theta_e_deg, alpha_deg, hpbw_v_deg, sla_v_db)
	A_after  = _vertical_attenuation(theta_e_deg, alpha_deg + delta_tilt_deg, hpbw_v_deg, sla_v_db)

	# Gain change at the edge direction (dB); typically negative when increasing downtilt
	deltaG_dB = -(A_after - A_before)

	# Translate to distance using log-distance model
	d_new_m = d_max_m * (10.0 ** (deltaG_dB / (10.0 * n)))

	# Geometry-only boresight intercepts (for context)
	def _bore_intercept(alpha_deg_val: float) -> float:
		if abs(alpha_deg_val) < 1e-6:
			return float('inf')
		return h_m / math.tan(math.radians(alpha_deg_val))

	bore_before = _bore_intercept(alpha_deg)
	bore_after  = _bore_intercept(alpha_deg + delta_tilt_deg)

	if tilt_direction == "downtilt":
		reduction_m = d_max_m - d_new_m
		reduction_pct = (reduction_m / d_max_m) * 100.0
		if reduction_pct < 0:
			reduction_pct = 0
	else:
		reduction_m = d_new_m - d_max_m
		reduction_pct = (reduction_m / d_new_m) * 100.0
		if reduction_pct < 0:
			reduction_pct = 0
	return d_new_m, reduction_pct


def calc_cell_dist_metrics(df):
	cell_dist_metrics = (
		df
		.groupby('cell_name', as_index=False)
		.agg(
			#cell_max_distance_to_cell=('cell_max_distance_to_cell', 'max'),
			cell_max_distance_to_cell_all_data=('distance_to_cell', 'max'), # This includes outliers, use this when constructing up-tilt grids so that they spread using this as the max TA
			cell_max_distance_to_cell=('distance_to_cell', lambda s: s.quantile(0.98)),
			TiltE=('TiltE', 'max'),
			TiltM=('TiltM', 'max'),
			Height=('Height', 'max')
			)
		.reset_index(drop=True)
		)

	cell_dist_metrics[['max_dist_1_dt', 'perc_dist_reduct_1_dt']] = cell_dist_metrics.apply(
	lambda x: pd.Series(estimate_distance_after_tilt(
									d_max_m= x.cell_max_distance_to_cell,      # current max distance from TA (meters)
									alpha_deg= x.TiltE + x.TiltM,      # current electrical downtilt (deg)
									h_m= x.Height,             # antenna height above UE height (m)
									hpbw_v_deg= 6.5,     # vertical HPBW (deg) from datasheet
									sla_v_db= 30,        # vertical side-lobe cap (dB)
									n= 3.5,              # path-loss exponent
									delta_tilt_deg= 1.0,  # tilt change (+1°)
									tilt_direction= "downtilt"
		)), axis=1)

	# 2 degrees down-tilt
	cell_dist_metrics[['max_dist_2_dt', 'perc_dist_reduct_2_dt']] = cell_dist_metrics.apply(
		lambda x: pd.Series(estimate_distance_after_tilt(
									d_max_m= x.cell_max_distance_to_cell,      # current max distance from TA (meters)
									alpha_deg= x.TiltE + x.TiltM,      # current electrical downtilt (deg)
									h_m= x.Height,             # antenna height above UE height (m)
									hpbw_v_deg= 6.5,     # vertical HPBW (deg) from datasheet
									sla_v_db= 30,        # vertical side-lobe cap (dB)
									n= 3.5,              # path-loss exponent
									delta_tilt_deg= 2.0,  # tilt change (+1°)
									tilt_direction= "downtilt"
		)), axis=1)

	# 1 degrees up-tilt
	cell_dist_metrics[['max_dist_1_ut', 'perc_dist_inc_1_ut']] = cell_dist_metrics.apply(
		lambda x: pd.Series(estimate_distance_after_tilt(
									d_max_m= x.cell_max_distance_to_cell,      # current max distance from TA (meters)
									alpha_deg= x.TiltE + x.TiltM,      # current electrical downtilt (deg)
									h_m= x.Height,             # antenna height above UE height (m)
									hpbw_v_deg= 6.5,     # vertical HPBW (deg) from datasheet
									sla_v_db= 30,        # vertical side-lobe cap (dB)
									n= 3.5,              # path-loss exponent
									delta_tilt_deg= -1.0,  # tilt change (-1°)
									tilt_direction= "uptilt"
		)), axis=1)

	# 2 degrees up-tilt
	cell_dist_metrics[['max_dist_2_ut', 'perc_dist_inc_2_ut']] = cell_dist_metrics.apply(
		lambda x: pd.Series(estimate_distance_after_tilt(
									d_max_m= x.cell_max_distance_to_cell,      # current max distance from TA (meters)
									alpha_deg= x.TiltE + x.TiltM,      # current electrical downtilt (deg)
									h_m= x.Height,             # antenna height above UE height (m)
									hpbw_v_deg= 6.5,     # vertical HPBW (deg) from datasheet
									sla_v_db= 30,        # vertical side-lobe cap (dB)
									n= 3.5,              # path-loss exponent
									delta_tilt_deg= -2.0,  # tilt change (-1°)
									tilt_direction= "uptilt"
		)),axis=1)
	return cell_dist_metrics

#####################################################################################
### Functions Related to Calculating RSRP values for existing bins with new tilts ###
#####################################################################################

def _vertical_attenuation_vec(theta_deg, alpha_deg, hpbw_v_deg=6.5, sla_v_db=30.0):
    """
    Vectorized vertical attenuation:
      A_v(θ) = min( 12 * ((θ - α)/HPBW_v)^2 , SLA_v )
    """
    x = (np.asarray(theta_deg, float) - np.asarray(alpha_deg, float)) / float(hpbw_v_deg)
    A = 12.0 * np.square(x)
    return np.minimum(A, float(sla_v_db))

def _predict_tilt_scenario(
    df, *,
    rsrp_col, r_col, h_col, tilt_e_col, tilt_m_col,
    dmax_col, pchg_col, out_col,
    delta_tilt_deg, hpbw_v_deg=6.5, sla_v_db=30.0
):
    # Pull arrays
    rsrp  = df[rsrp_col].to_numpy(dtype=float)
    r_m   = df[r_col].to_numpy(dtype=float)
    h_m   = df[h_col].to_numpy(dtype=float)
    alpha = (df[tilt_e_col].to_numpy(dtype=float) + df[tilt_m_col].to_numpy(dtype=float))
    dmax  = df[dmax_col].to_numpy(dtype=float)
    pchg  = df[pchg_col].to_numpy(dtype=float)

    # Elevation angle θ_e = atan2(h, r) [deg]
    theta = np.degrees(np.arctan2(h_m, r_m))

    # Attenuation before/after tilt
    A_before = _vertical_attenuation_vec(theta, alpha, hpbw_v_deg, sla_v_db)
    A_after  = _vertical_attenuation_vec(theta, alpha + float(delta_tilt_deg), hpbw_v_deg, sla_v_db)

    # Gain change and raw new RSRP
    deltaG = A_before - A_after              # same as -(A_after - A_before)
    new_rsrp = rsrp + deltaG

    # Masks per your rules
    invalid = (r_m <= 0) | (h_m < 0) | ~np.isfinite(r_m) | ~np.isfinite(h_m)
    beyond  = (r_m > dmax)
    nochg   = (pchg == 0.0)

    # Compose result
    out = np.where(nochg, rsrp, new_rsrp)    # keep original if percent_dist_change == 0
    out = np.where(beyond, np.nan, out)      # NaN if r_m > d_max
    out = np.where(invalid, np.nan, out)     # NaN for invalid inputs

    df[out_col] = out

def predict_rsrp_existing_bins_vec(df):
    """
    Vectorised replacement for your apply-based function.
    Adds:
      - 'avg_rsrp_1_degree_downtilt'
      - 'avg_rsrp_2_degree_downtilt'
      - 'avg_rsrp_1_degree_uptilt'
      - 'avg_rsrp_2_degree_uptilt'
    """
    common_kwargs = dict(
        df=df,
        rsrp_col="avg_rsrp",
        r_col="distance_to_cell",
        h_col="Height",
        tilt_e_col="TiltE",
        tilt_m_col="TiltM",
        hpbw_v_deg=6.5,
        sla_v_db=30.0,
    )

    _predict_tilt_scenario(
        dmax_col="max_dist_1_dt", pchg_col="perc_dist_reduct_1_dt",
        out_col="avg_rsrp_1_degree_downtilt", delta_tilt_deg=+1.0, **common_kwargs
    )
    _predict_tilt_scenario(
        dmax_col="max_dist_2_dt", pchg_col="perc_dist_reduct_2_dt",
        out_col="avg_rsrp_2_degree_downtilt", delta_tilt_deg=+2.0, **common_kwargs
    )
    _predict_tilt_scenario(
        dmax_col="max_dist_1_ut", pchg_col="perc_dist_inc_1_ut",
        out_col="avg_rsrp_1_degree_uptilt", delta_tilt_deg=-1.0, **common_kwargs
    )
    _predict_tilt_scenario(
        dmax_col="max_dist_2_ut", pchg_col="perc_dist_inc_2_ut",
        out_col="avg_rsrp_2_degree_uptilt", delta_tilt_deg=-2.0, **common_kwargs
    )

    # Optional: clamp to LTE RSRP bounds
    for c in [
        "avg_rsrp_1_degree_downtilt",
        "avg_rsrp_2_degree_downtilt",
        "avg_rsrp_1_degree_uptilt",
        "avg_rsrp_2_degree_uptilt",
    ]:
        df[c] = df[c].clip(-140, -44)

    return df

###########################################################
### Functions Related to Finding new bins with up-tilts ###
###########################################################

# ==============================
# 1) Vectorized geometry helpers
# ==============================

R_EARTH = 6_371_008.8  # meters

def _destination_from(lat0_deg, lon0_deg, dist_m, bearing_deg):
	"""
	Vectorized great-circle forward solution.
	lat0_deg, lon0_deg: scalars (deg)
	dist_m, bearing_deg: arrays/scalars (meters, deg)
	Returns arrays lat2_deg, lon2_deg
	"""
	lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
	brng = np.deg2rad(bearing_deg)
	ang  = np.asarray(dist_m, dtype=float) / R_EARTH

	sin_lat0, cos_lat0 = np.sin(lat0), np.cos(lat0)
	sin_ang,  cos_ang  = np.sin(ang),  np.cos(ang)

	lat2 = np.arcsin(sin_lat0 * cos_ang + cos_lat0 * sin_ang * np.cos(brng))
	lon2 = lon0 + np.arctan2(np.sin(brng) * sin_ang * cos_lat0, cos_ang - sin_lat0 * np.sin(lat2))
	lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi  # normalize to [-180, 180)
	return np.rad2deg(lat2), np.rad2deg(lon2)

def latlon1cell_vectorized(cell_lat_deg, cell_lon_deg, ta_m, max_ta_m, hbeam_deg, azimuth_deg, rng=None):
	"""
	Vectorized rewrite of your latLon1Cell (same lobe logic, ±5% TA jitter, oval beam).
	Inputs:
		- cell_lat_deg, cell_lon_deg, hbeam_deg, azimuth_deg: scalars
		- ta_m: array of distances (meters)
		- max_ta_m: scalar (meters)
	Returns:
		(lat_deg_array, lon_deg_array)
	"""
	rng = np.random.default_rng() if rng is None else rng
	ta = np.asarray(ta_m, dtype=float)
	N = ta.size

	# +/- 5% TA jitter
	ta = ta * rng.uniform(0.95, 1.05, size=N)

	back_factor = 0.10
	side_factor = 0.50
	base_beam   = 0.25 * float(hbeam_deg)

	back = ta <= (max_ta_m * back_factor)
	side = (~back) & (ta <= (max_ta_m * side_factor))

	bearings = np.empty(N, dtype=float)
	set_mask = np.zeros(N, dtype=bool)

	# Back-lobe: 30% anywhere 1..359°, else in shrinking beam
	'''if back.any():
		choose360 = rng.random(N) < 0.30
		m360   = back & choose360
		mbeamB = back & (~choose360)

		if m360.any():
			bearings[m360] = rng.uniform(1.0, 359.0, size=m360.sum()); set_mask[m360] = True

		if mbeamB.any():
			multiple = ta[mbeamB] / (max_ta_m * back_factor)
			beam     = base_beam - (hbeam_deg * multiple)
			u        = rng.random(mbeamB.sum())
			bearings[mbeamB] = (azimuth_deg - beam) + u * (2 * beam); set_mask[mbeamB] = True

	# Side-lobes: 50% go to side lobes (split left/right), others fall through to main
	if side.any():
		go_side = rng.random(N) <= 0.5
		m_go    = side & go_side
		if m_go.any():
			multiple = ta[m_go] / (max_ta_m * side_factor)
			beam     = base_beam - (base_beam * multiple)
			left     = rng.random(m_go.sum()) <= 0.5
			centers  = np.where(left, azimuth_deg - hbeam_deg, azimuth_deg + hbeam_deg)
			u        = rng.random(m_go.sum())
			bearings[m_go] = (centers - beam) + u * (2 * beam); set_mask[m_go] = True'''

	# Main lobe: everything not set yet
	m_main = ~set_mask
	if m_main.any():
		ru       = rng.uniform(0.9, 1.0, size=m_main.sum())
		ratio    = ta[m_main] / max_ta_m
		multiple = np.minimum(ru, ratio)
		beam     = base_beam - (base_beam * multiple)
		u        = rng.random(m_main.sum())
		#bearings[m_main] = (azimuth_deg - beam) + u * (2 * beam)
		bearings[m_main] = (azimuth_deg - base_beam) + u * (2 * base_beam)

		return _destination_from(cell_lat_deg, cell_lon_deg, ta, bearings)

# ===========================
# 2) Build undershooter grid
# ===========================

def build_new_grids(undershooting_final_candidates: pd.DataFrame,
						gis_df: pd.DataFrame,
						n_samples: int = 1000,
						seed: int | None = None) -> gpd.GeoDataFrame:
	"""
	Parameters
	----------
	undershooting_final_candidates : DataFrame
		Columns required: ['cell_name','max_ta','max_distance_1_degree_uptilt','max_distance_2_degree_uptilt']
	gis_df : DataFrame
		Columns required: ['Name','Latitude','Longitude','HBW','Bearing']
	n_samples : int
		Samples to generate per cell.
	seed : int | None
		RNG seed for reproducibility.

	Returns
	-------
	DataFrame
		Columns: ['cell_name','ta','lat','lon']
	"""
	rng = np.random.default_rng(seed)

	# Per-cell metadata (dedupe and merge once)
	cand = (undershooting_final_candidates
			#[['cell_name','max_ta','max_dist_1_ut','max_dist_2_ut']]cell_max_distance_to_cell_all_data
			[['cell_name','max_ta','max_dist_1_ut','max_dist_2_ut', 'cell_max_distance_to_cell_all_data']]
			.drop_duplicates('cell_name'))

	gis_cols = (gis_df[['Name','Latitude','Longitude','HBW','Bearing']]
				.rename(columns={'Name':'cell_name',
					'Latitude':'lat0','Longitude':'lon0',
					'HBW':'hbw','Bearing':'bearing'}))

	meta = cand.merge(gis_cols, on='cell_name', how='inner').copy()
	meta['max_dist'] = np.maximum(meta['max_dist_1_ut'], meta['max_dist_2_ut'])

	# ensure numeric dtypes
	#num_cols = ['max_ta','max_dist_1_ut','max_dist_2_ut', 'max_dist','lat0','lon0','hbw','bearing']
	num_cols = ['max_ta','max_dist_1_ut','max_dist_2_ut', 'max_dist','lat0','lon0','hbw','bearing', 'cell_max_distance_to_cell_all_data']
	meta[num_cols] = meta[num_cols].apply(pd.to_numeric, errors='coerce')
	#meta = meta.dropna(subset=['max_ta','max_dist','lat0','lon0','hbw','bearing'])
	meta = meta.dropna(subset=['max_ta','max_dist','lat0','lon0','hbw','bearing', 'cell_max_distance_to_cell_all_data'])

	rows = []
	for row in meta.itertuples(index=False):
		low  = float(min(row.max_ta, row.max_dist))
		high = float(max(row.max_ta, row.max_dist))
		if not np.isfinite(low) or not np.isfinite(high) or high <= 0:
			continue

		# Sample TA uniformly between the bounds
		ta = rng.uniform(low, high, size=n_samples)

		# Generate lat/lon using your (vectorized) lobe model
		lats, lons = latlon1cell_vectorized(
			cell_lat_deg=float(row.lat0),
			cell_lon_deg=float(row.lon0),
			ta_m=ta,
			#max_ta_m=float(row.max_dist),
			max_ta_m=float(row.cell_max_distance_to_cell_all_data),
			hbeam_deg=float(row.hbw),
			azimuth_deg=float(row.bearing),
			rng=rng
		)

		rows.append(pd.DataFrame({
			'cell_name': row.cell_name,
			'ta': ta,
			'lat': lats,
			'lon': lons
		}))

	if not rows:
		# Nothing generated
		return gpd.GeoDataFrame(columns=['cell_name','ta','lat','lon','geometry'], crs='EPSG:4326')

	new_grids = pd.concat(rows, ignore_index=True)

	return new_grids

####################################################
### Functions Related to Building extended grids ###
####################################################

def _utm_from_latlon(lat, lon) -> CRS:
	"""Pick a local UTM CRS from WGS84 lat/lon scalars."""
	lat = float(lat); lon = float(lon)
	if lon > 180: lon -= 360
	if lon < -180: lon += 360
	zone = int(np.floor((lon + 180) / 6) + 1)
	epsg = 32600 + zone if lat >= 0 else 32700 + zone
	return CRS.from_epsg(epsg)


def _shift_ring_coords(coords, px, py, max_offset_m):
	"""
	Shift each vertex away from pivot (px,py) by an offset proportional to its
	distance relative to the farthest vertex distance in the ring.
	"""
	arr = np.asarray(coords, float)
	if arr.size == 0:
		return coords

	x = arr[:, 0]; y = arr[:, 1]
	dx = x - px; dy = y - py
	d = np.hypot(dx, dy)
	if d.size == 0:  # empty
		return coords
	dmax = d.max()
	if not np.isfinite(dmax) or dmax <= 0:
		return coords

	w = d / dmax
	# unit vectors from pivot (safe divide)
	ux = np.divide(dx, d, out=np.zeros_like(dx), where=d > 0)
	uy = np.divide(dy, d, out=np.zeros_like(dy), where=d > 0)
	off = float(max_offset_m) * w

	x_new = x + ux * off
	y_new = y + uy * off

	# guard against NaNs/Infs from numeric issues
	if not (np.isfinite(x_new).all() and np.isfinite(y_new).all()):
		return coords

	out = np.column_stack([x_new, y_new])
	# ensure ring closure
	if not np.allclose(out[0], out[-1]):
		out = np.vstack([out, out[0]])
	return [tuple(pt) for pt in out]

def _uniform_expand_about_pivot(geom_m, px, py, max_offset_m):
	# compute dmax (farthest vertex distance from pivot)
	def _dmax(poly):
		c = np.asarray(poly.exterior.coords)
		if c.size == 0:
			return 0.0
		dx, dy = c[:,0] - px, c[:,1] - py
		return float(np.hypot(dx, dy).max())

	if isinstance(geom_m, Polygon):
		dmax = _dmax(geom_m)
		parts = [geom_m]
	elif isinstance(geom_m, MultiPolygon):
		dmax = max((_dmax(p) for p in geom_m.geoms), default=0.0)
		parts = list(geom_m.geoms)
	else:
		return GeometryCollection()

	if not np.isfinite(dmax) or dmax <= 0:
		return GeometryCollection()

	f = 1.0 + float(max_offset_m) / dmax  # scale so farthest point moves by offset
	scaled = [shp_scale(p, xfact=f, yfact=f, origin=(px, py)) for p in parts]
	out = unary_union(scaled)
	return make_valid(out)

def expand_hulls_weighted_by_pivot_rowwise(
	hull_gdf: gpd.GeoDataFrame,
	lat_col: str = "Latitude",
	lon_col: str = "Longitude",
	offset_col: str = "dist_increase_ut",
	) -> gpd.GeoDataFrame:
	"""
	For each row, create the *difference area* (expanded - original) using a per-row
	WGS84 pivot (lat_col, lon_col) and a per-row max offset (offset_col, meters).
	If there is no difference (expanded ~= original), return the expanded geometry.
	Output CRS: EPSG:4326.
	"""
	required = {lat_col, lon_col, offset_col, "geometry"}
	missing = required - set(hull_gdf.columns)
	if missing:
		raise KeyError(f"Missing required columns: {sorted(missing)}")

	# Ensure WGS84 tagging
	wgs = hull_gdf.set_crs("EPSG:4326") if hull_gdf.crs is None else hull_gdf.to_crs("EPSG:4326")

	# Drop obvious bad rows early
	wgs = wgs[wgs.geometry.notna() & ~wgs.geometry.is_empty].copy()
	bounds = wgs.geometry.bounds
	wgs = wgs[~bounds.isna().any(axis=1)].copy()

	new_geoms = []

	for _, row in wgs.iterrows():
		geom = row.geometry

		# Per-row scalars
		try:
			max_offset_m = float(row[offset_col])
			lat = float(row[lat_col])
			lon = float(row[lon_col])
		except Exception:
			new_geoms.append(GeometryCollection())
			continue

		if not np.isfinite(max_offset_m) or max_offset_m == 0.0:
			new_geoms.append(GeometryCollection())
			continue
		if not (np.isfinite(lat) and np.isfinite(lon)):
			new_geoms.append(GeometryCollection())
			continue

		# Per-row local CRS in meters
		proj = _utm_from_latlon(lat, lon)

		# Project original geometry and pivot
		try:
			geom_m = gpd.GeoSeries([geom], index=[0], crs="EPSG:4326").to_crs(proj).iloc[0]
			pivot_m = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(proj).iloc[0]
		except Exception:
			new_geoms.append(GeometryCollection())
			continue

		px, py = pivot_m.x, pivot_m.y

		# Build expanded geometry (in meters)
		if isinstance(geom_m, Polygon):
			ext   = _shift_ring_coords(list(geom_m.exterior.coords), px, py, max_offset_m)
			holes = [list(r.coords) for r in geom_m.interiors]  # keep holes unchanged
			#expanded_m = make_valid(Polygon(ext, holes=holes))
			expanded_m = _uniform_expand_about_pivot(geom_m, px, py, max_offset_m)
		elif isinstance(geom_m, MultiPolygon):
			parts = []
			for p in geom_m.geoms:
				ext   = _shift_ring_coords(list(p.exterior.coords), px, py, max_offset_m)
				holes = [list(r.coords) for r in p.interiors]
				parts.append(Polygon(ext, holes=holes))
			#expanded_m = make_valid(MultiPolygon(parts))
			expanded_m = _uniform_expand_about_pivot(geom_m, px, py, max_offset_m)
		else:
			new_geoms.append(GeometryCollection())
			continue

		# Skip corrupted shapes
		bx = expanded_m.bounds
		by = geom_m.bounds
		if any(np.isnan(bx)) or any(np.isnan(by)) or expanded_m.is_empty or geom_m.is_empty:
			d = GeometryCollection()
		else:
			# Pre-repair
			em = make_valid(expanded_m)
			gm = make_valid(geom_m)

			# Boolean op (mute that specific warning locally)
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore",
					message="invalid value encountered in difference",
					category=RuntimeWarning,
					module=r"shapely\..*",
				)
				try:
					d = em.difference(gm)
				except Exception:
					d = GeometryCollection()

		# Keep only polygonal parts
		if isinstance(d, (Polygon, MultiPolygon)):
			diff_m = d
		elif isinstance(d, GeometryCollection):
			polys = [g for g in d.geoms if isinstance(g, (Polygon, MultiPolygon))]
			diff_m = unary_union(polys) if polys else GeometryCollection()
		else:
			diff_m = GeometryCollection()

		# Fallback: if no difference, return expanded instead
		eq_exact = False
		try:
			# tolerance in meters (projected CRS)
			eq_exact = expanded_m.equals_exact(geom_m, tolerance=0.01)
		except Exception:
			eq_exact = False

		no_diff = (
			(isinstance(diff_m, GeometryCollection) and diff_m.is_empty) or
			(hasattr(diff_m, "area") and diff_m.area == 0) or
			eq_exact
		)
		if no_diff:
			cand = expanded_m
			if isinstance(cand, GeometryCollection):
				polys = [g for g in cand.geoms if isinstance(g, (Polygon, MultiPolygon))]
				cand = unary_union(polys) if polys else GeometryCollection()
			diff_m = cand

		# Back to WGS84 and append
		try:
			diff_wgs = gpd.GeoSeries([diff_m], crs=proj).to_crs("EPSG:4326").iloc[0]
		except Exception:
			diff_wgs = GeometryCollection()

		new_geoms.append(diff_wgs)

	out = wgs.copy()
	out.geometry = new_geoms
	return out

########################################################
### Add additional required columns to the dataframe ###
######################################################## 

def add_required_columns(df):
	'''
	Add additional columns;
		- 'perc_grid_events'
		- 'avg_rsrp_grid'
		- 'avg_rsrp_cell'
		- 'grid_max_distance_to_cell'
		- 'grid_min_distance_to_cell'
		- 'perc_cell_max_dist'
		- 'cell_angle_to_grid'
		- 'grid_bearing_diff'
		- 'cell_count'
		- 'same_pci_cell_count'
	'''
	####################################
	### Calculate 'perc_grid_events' ###
	####################################
	df['rsrp_weight'] = 150 + df['avg_rsrp']
	df['rsrp_weight'] = pd.to_numeric(df['rsrp_weight'], errors='coerce').fillna(0)

	# sum of rsrp_weight per grid
	grid_tot = df.groupby('grid')['rsrp_weight'].transform('sum')

	# percentage of each row within its grid (0–100)
	df['perc_grid_events'] = np.where(grid_tot > 0,
		df['rsrp_weight'] / grid_tot, 0.0)
	# Round
	df['perc_grid_events'] = df['perc_grid_events'].round(4)

	#################################
	### Calculate 'avg_rsrp_grid' ###
	#################################
	df['rsrp_weighted'] = df['perc_grid_events'] * df['avg_rsrp']
	df['rsrp_weighted'] = pd.to_numeric(df['rsrp_weighted'], errors='coerce').fillna(0)
	grid_avg_rsrp = df.groupby('grid')['rsrp_weighted'].transform('sum')

	df['avg_rsrp_grid'] = grid_avg_rsrp

	#################################
	### Calculate 'avg_rsrp_grid' ###
	#################################
	cell_avg_rsrp = df.groupby('cell_name')['avg_rsrp'].transform('mean')
	df['avg_rsrp_cell'] = cell_avg_rsrp

	#############################################
	### Calculate 'grid_max_distance_to_cell' ###
	#############################################
	grid_dist_to_cell_max = df.groupby('grid')['distance_to_cell'].transform('max')
	df['grid_max_distance_to_cell'] = grid_dist_to_cell_max

	#############################################
	### Calculate 'grid_min_distance_to_cell' ###
	#############################################
	grid_dist_to_cell_min = df.groupby('grid')['distance_to_cell'].transform('min')
	df['grid_min_distance_to_cell'] = grid_dist_to_cell_min

	######################################
	### Calculate 'perc_cell_max_dist' ###
	######################################
	df['perc_cell_max_dist'] = df['distance_to_cell'] / df['grid_max_distance_to_cell']

	######################################
	### Calculate 'cell_angle_to_grid' ###
	######################################
	lon2 = df.geometry.x.to_numpy()   # grid lon
	lat2 = df.geometry.y.to_numpy()   # grid lat

	# Cell coordinates (columns 'Latitude', 'Longitude')
	lat1 = np.radians(df['Latitude'].to_numpy())
	lon1 = np.radians(df['Longitude'].to_numpy())
	lat2 = np.radians(lat2)
	lon2 = np.radians(lon2)

	# Initial bearing (geodesic) formula
	dlon = lon2 - lon1
	y = np.sin(dlon) * np.cos(lat2)
	x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
	bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0  # 0–360, 0=north

	df['cell_angle_to_grid'] = bearing

	#####################################
	### Calculate 'grid_bearing_diff' ###
	#####################################
	df['grid_bearing_diff'] = np.abs(((df['cell_angle_to_grid'] - df['Bearing'] + 180) % 360) - 180)

	##############################
	### Calculate 'cell_count' ###
	##############################
	cell_count = df.groupby('grid')['cell_name'].transform('count')
	df['cell_count'] = cell_count

	#######################################
	### Calculate 'same_pci_cell_count' ###
	#######################################
	pci_grid = df[['grid', 'Scr_Freq', 'cell_name']].groupby(['grid', 'Scr_Freq']).count().reset_index()
	pci_grid.rename(columns={'cell_name': 'same_pci_cell_count'}, inplace=True)
	df = pd.merge(df, pci_grid, on = ['grid', 'Scr_Freq'], how = 'left')
	
	################
	### Clean up ###
	################
	df.drop(columns = ['rsrp_weight', 'rsrp_weighted'], inplace = True)
	del grid_tot
	del grid_avg_rsrp
	del cell_avg_rsrp
	del bearing
	del cell_count

	return df



