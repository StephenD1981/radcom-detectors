import geopandas as gpd
from shapely.geometry import Point
from shapely.affinity import rotate
from pyproj import CRS

def rotate_hull_around_pivot(hull_gdf, lat, lon, angle_deg):
	# ensure WGS84
	hull_wgs = hull_gdf.to_crs("EPSG:4326") if hull_gdf.crs else hull_gdf.set_crs("EPSG:4326")
	# pick local projected CRS near the hull
	proj_crs = hull_wgs.estimate_utm_crs()
	# build pivot
	pivot_wgs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
	# project both
	hull_p = hull_wgs.to_crs(proj_crs)
	pivot_p = pivot_wgs.to_crs(proj_crs).iloc[0]
	# rotate around pivot
	rotated = hull_p.geometry.apply(lambda g: rotate(g, angle_deg, origin=(pivot_p.x, pivot_p.y), use_radians=False))
	out = hull_p.copy()
	out.geometry = rotated
	return out.to_crs("EPSG:4326")