import numpy as np
import pandas as pd
import geohash
from functools import lru_cache
import math
from sklearn.cluster import AgglomerativeClustering


def neigh8(gh: str) -> list[str]:
	return geohash.neighbors(gh)

@lru_cache(maxsize=None)
def kring(gh: str, k: int) -> frozenset[str]:
	seen = {gh}
	frontier = {gh}
	for _ in range(k):
		nxt = set()
		for g in frontier:
			nxt.update(neigh8(g))
		nxt -= seen
		seen |= nxt
		frontier = nxt
	return frozenset(seen)

def count_present_in_kring(gh: str, k: int, present: set[str]) -> int:
	K = kring(gh, k)
	return len(K & present) - 1  # minus 1 to exclude gh itself

def cluster_by_anchor_within5(s: pd.Series, width: float = 5.0) -> pd.Series:
	"""
	s: sorted Series of rsrp_dist_max for a single grid.
	width: max allowed difference from the cluster's anchor (default 5).
	Returns 1-based cluster ids aligned to s.index.
	"""
	vals = s.to_numpy()
	if len(vals) == 0:
		return pd.Series(dtype="int64", index=s.index)

	grp = np.zeros(len(vals), dtype=int)  # 0-based ids
	anchor = vals[0]
	gid = 0
	for i in range(1, len(vals)):
		if vals[i] - anchor <= width:
			grp[i] = gid
		else:
			gid += 1
			grp[i] = gid
			anchor = vals[i]
	return pd.Series(grp + 1, index=s.index)  # 1-based

def cluster_with_complete_linkage(s: pd.Series, width: float = 5.0) -> pd.Series:
	"""
	Per-grid clustering of a 1D series so each cluster's max pairwise distance <= 2*width.
	Robust to NaNs and groups with <2 items.
	Returns 1-based labels aligned to s.index.
	"""
	# work on a clean, sorted copy
	x = pd.to_numeric(s, errors='coerce').dropna().sort_values()
	if x.size == 0:
		# nothing to label -> all missing
		return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)
	if x.size == 1:
		# single item -> label 1 (or pd.NA if you prefer)
		out = pd.Series(pd.array([1], dtype="Int64"), index=x.index)
		return out.reindex(s.index)

	X = x.to_numpy().reshape(-1, 1)
	# sklearn API compatibility: some versions want 'affinity', newer prefer 'metric'
	try:
		model = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=2 * width,
			linkage='complete',
			metric='euclidean',
		)
	except TypeError:
		model = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=2 * width,
			linkage='complete',
			affinity='euclidean',
		)

	labels = model.fit_predict(X) + 1  # 1-based labels
	out = pd.Series(pd.array(labels, dtype="Int64"), index=x.index)
	return out.reindex(s.index)

def find_interference_cells(df, min_filtered_cells_per_grid, min_cell_event_count, perc_grid_events, dominant_perc_grid_events, dominance_diff, max_rsrp_diff, k, perc_interference, data_type, clustering_algo):
	
	# min_filtered_cells_per_grid 	4
	# min_cell_event_count			2
	# perc_grid_events				0.05
	# dominant_perc_grid_events		0.3
	# max_rsrp_diff					5
	# k 							3
	# perc_interference 			0.33

	#max_number_of_interference_cells = math.ceil(len(list(set(df.cell_name.to_list()))) / 10)
	
	interference_grids_per_cell_complete = pd.DataFrame()
	grid_geo_data_geo_filitered_complete = pd.DataFrame()

	print(f"Working on '{data_type} data..")
	for band in list(set(df.Band.to_list())):
		print(f"\tWorking on Band '{band}'")

		grid_geo_data_diff = df[df.Band == band].copy()

		# Recalculate 'perc_grid_events'
		den = grid_geo_data_diff.groupby('grid')['perc_grid_events'].transform('sum')
		grid_geo_data_diff['perc_grid_events'] = (grid_geo_data_diff['perc_grid_events'].div(den).fillna(0))            # if a group's sum is 0 or NaN, set result to 0

		# Find 80th/95th Percentile 'cell_count' for 'grid_geo_data'
		# -> We want to tackle interference gradually, always targeting the worst 20%/5% of interference grids
		#if data_type == 'perceived':
		#	min_cell_grid_count = grid_geo_data_diff[grid_geo_data_diff.Band == band].drop_duplicates(['grid', 'cell_count']).groupby('grid')['cell_count'].max().quantile(0.95)
		#	print(f"\t\t95th percentile 'cell_count' = {min_cell_grid_count} cells per grid")
		#else:
		#	min_cell_grid_count = grid_geo_data_diff[grid_geo_data_diff.Band == band].drop_duplicates(['grid', 'cell_count']).groupby('grid')['cell_count'].max().quantile(0.80)
		#	print(f"\t\t80th percentile 'cell_count' = {min_cell_grid_count} cells per grid")

		print("\t\tInitial cell-grid rows = {}".format(grid_geo_data_diff.shape[0]))

		# Apply initial interference filters to Grid-Cell dataset
		if data_type == 'perceived':
			# Ensure we have grid cell_count >= min_cell_grid_count
			#grid_geo_data_diff = grid_geo_data_diff[(grid_geo_data_diff.cell_count >= min_cell_grid_count)]
			pass

		else:
			grid_geo_data_diff = grid_geo_data_diff[#(grid_geo_data_diff.cell_count >= min_cell_grid_count) & \
													(grid_geo_data_diff.event_count >= min_cell_event_count) & \
													(grid_geo_data_diff.perc_grid_events >= perc_grid_events)]
			print("\t\tFilter by 'event_count' & 'perc_grid_events'; cell-grid rows = {}".format(grid_geo_data_diff.shape[0]))

		# Now calculate 'rsrp_dist_max' and create clustering
		grid_rsrp_max = (grid_geo_data_diff.groupby(['grid'], as_index=False).agg({'avg_rsrp': 'max'}))
		grid_rsrp_max.rename(columns = {'avg_rsrp' : 'max_rsrp'}, inplace = True)
		grid_geo_data_diff = grid_geo_data_diff.merge(grid_rsrp_max, on = 'grid', how = 'inner')
		#grid_geo_data_diff['avg_rsrp_diff'] = grid_geo_data_diff['avg_rsrp'] - grid_geo_data_diff['avg_rsrp_grid']
		
		grid_geo_data_diff['rsrp_dist_max'] = grid_geo_data_diff['max_rsrp'] - grid_geo_data_diff['avg_rsrp']

		grid_geo_data_diff['rsrp_dist_max'] = pd.to_numeric(grid_geo_data_diff['rsrp_dist_max'], errors='coerce')

		grid_geo_data_diff = grid_geo_data_diff.sort_values(['grid', 'rsrp_dist_max'])
		if clustering_algo == 'fixed':
			grid_geo_data_diff['grid_cluster_no'] = (grid_geo_data_diff.groupby('grid', group_keys=False)['rsrp_dist_max'].apply(lambda s: cluster_by_anchor_within5(s, width=5.0)))
		elif clustering_algo == 'dynamic-sklearn':
			grid_geo_data_diff['grid_cluster_no'] = (grid_geo_data_diff.groupby('grid', group_keys=False)['rsrp_dist_max'].apply(lambda s: cluster_with_complete_linkage(s, width=2.5)))
		
		# give each cluster a readable name unique per grid
		labels_num = grid_geo_data_diff['grid_cluster_no'].astype('Int64').fillna(0)

		grid_geo_data_diff['grid_cluster_name'] = (grid_geo_data_diff['grid'].astype(str) + '_grp' + labels_num.astype(int).astype(str).str.zfill(2)) # 0 -> '00'

		#grid_geo_data_diff = grid_geo_data_diff[(grid_geo_data_diff['avg_rsrp_diff'].abs() <= max_rsrp_diff)]

		#print(grid_geo_data_diff[['cell_name', 'grid', 'avg_rsrp', 'max_rsrp' , 'rsrp_dist_max', 'grid_cluster_no', 'grid_cluster_name']].head(20))
		# Count number of cells in each grid meeting the criteria, filter to where grid has more than 'min_filtered_cells_per_grid'
		#print("\t\tInitial filtering (grid-cell count & avg_rsrp_diff) reduces to = {}".format(grid_geo_data_diff.shape[0]))

		#######################################
		### Remove grids with dominant cell ###
		#######################################
		grid_geo_data_dominant = df[df.Band == band].copy()
		grid_geo_data_dominant = grid_geo_data_dominant.merge(grid_rsrp_max[['grid', 'max_rsrp']], on = 'grid', how = 'inner')

		second = (df.groupby('grid')['avg_rsrp'].nlargest(2).groupby(level=0).nth(1).reset_index(name='avg_rsrp_second'))
		grid_geo_data_dominant = grid_geo_data_dominant.merge(second, on='grid', how='inner')

		grid_geo_data_dominant['rsrp_diff_1_2'] = grid_geo_data_dominant['max_rsrp'] - grid_geo_data_dominant['avg_rsrp_second']

		# Apply initial interference filters to Grid-Cell dataset
		if data_type == 'perceived':
			grid_geo_data_dominant = grid_geo_data_dominant[(grid_geo_data_dominant['rsrp_diff_1_2'] >= dominance_diff)]
		else:
			grid_geo_data_dominant = grid_geo_data_dominant[(grid_geo_data_dominant.perc_grid_events >= dominant_perc_grid_events) & \
														(grid_geo_data_dominant['rsrp_diff_1_2'] >= dominance_diff)]

		# Find grids to be excluded
		dominant_cell_grids = list(set(grid_geo_data_dominant.grid.to_list()))
		print(f"\t\tCount of dominant grids = {len(dominant_cell_grids)}")
		grid_geo_data_diff = grid_geo_data_diff[~grid_geo_data_diff.grid.isin(dominant_cell_grids)]
		print("\t\tFiltering grids with dominant cells reduces to = {}".format(grid_geo_data_diff.shape[0]))


		print("\t\tCalculate target clusters count per grid")
		grid_counts = grid_geo_data_diff.groupby("grid_cluster_name").size().reset_index(name="count")  
		print("\t\t\tCluster counts pre filter = {}".format(grid_counts.shape[0]))
		if data_type == 'perceived':
			grid_counts = grid_counts[grid_counts['count'] >= math.ceil(min_filtered_cells_per_grid * 3)]#min_cell_grid_count * 0.75]
		else:
			grid_counts = grid_counts[grid_counts['count'] >= min_filtered_cells_per_grid]
		print("\t\t\tCluster counts post filter = {}".format(grid_counts.shape[0]))
		# Merge back to 'grid_geo_data_diff' for reduced grid-cell dataset
		grid_geo_data_diff = grid_geo_data_diff.merge(grid_counts, on="grid_cluster_name", how="inner")

		print("\t\tUpdated Cell-Grids count = {}".format(grid_geo_data_diff.shape[0]))

		###########################################################################################################
		### Let's create a filtering based on interference neighborhood, we will use 'neigh8 and kring to count ###
		###the surrounding rings containing interference with a step size of 3 = surrounding 49 grids           ###
		###########################################################################################################
		print("\t\tFilter further based on interference clustering using neigh8 algorithm")
		interference_set = set(grid_geo_data_diff['grid'].astype(str))
		records = [(gh, count_present_in_kring(gh, k, interference_set)) for gh in interference_set]
		interferer_k_df = pd.DataFrame(records, columns=['grid', f'interferers_within_{k}_steps'])

		grid_geo_data_geo_filitered = grid_geo_data_diff.copy()
		interferer_k_df = interferer_k_df[interferer_k_df.interferers_within_3_steps > (((2 * k) + 1)**2) * perc_interference]
		print("\t\t\tPre filter for surroundng interference grids = {} rows".format(grid_geo_data_geo_filitered.shape[0]))
		grid_geo_data_geo_filitered = grid_geo_data_geo_filitered.merge(interferer_k_df, on="grid", how="inner")
		print("\t\t\tPost filter for surroundng interference grids = {} rows".format(grid_geo_data_geo_filitered.shape[0]))

		print("\t\tFind RSRP range for weighting..")
		rsrp_min = grid_geo_data_geo_filitered['avg_rsrp'].quantile(0.02)
		rsrp_max = grid_geo_data_geo_filitered['avg_rsrp'].quantile(0.98)
		print(f"\t\t\tRSRP minimum = {rsrp_min}")
		print(f"\t\t\tRSRP maximum = {rsrp_max}")

		grid_geo_data_geo_filitered['weight'] = max(min((grid_geo_data_geo_filitered['avg_rsrp'] - rsrp_min) / (rsrp_max - rsrp_min), 1), 0)

		interference_grids_per_cell = (
			grid_geo_data_geo_filitered.groupby('cell_name', as_index=False).agg(
				interference_grid_count=('cell_name', 'size'),
				med_distance_to_cell=('distance_to_cell', 'median'),
				max_distance_to_cell=('grid_max_distance_to_cell', 'max'),
				grid_count=('grid_count', 'max'),
				avg_weight=('weight', 'mean')
			).sort_values('interference_grid_count', ascending=False).reset_index(drop=True)
		)

		interference_grids_per_cell['total_weight'] = interference_grids_per_cell['avg_weight'] * interference_grids_per_cell['grid_count']

		interference_grids_per_cell['perc_interference_grid'] = interference_grids_per_cell['interference_grid_count'] / interference_grids_per_cell['grid_count']
		interference_grids_per_cell['perc_distance'] = interference_grids_per_cell['med_distance_to_cell'] / interference_grids_per_cell['max_distance_to_cell']
		interference_grids_per_cell = interference_grids_per_cell[['cell_name', 'grid_count', 'interference_grid_count', 'perc_interference_grid', 'med_distance_to_cell', 'max_distance_to_cell', 'perc_distance', 'avg_weight', 'total_weight']]

		print("\t\tUpdated Cell-Grids count = {}".format(grid_geo_data_geo_filitered.shape[0]))
		print("\t\tThere are {} potential interference cells in total".format(interference_grids_per_cell.shape[0]))
		if interference_grids_per_cell_complete.empty:
			interference_grids_per_cell_complete = interference_grids_per_cell
		else:
			interference_grids_per_cell_complete = pd.concat([interference_grids_per_cell_complete, interference_grids_per_cell], ignore_index=True)

		if grid_geo_data_geo_filitered_complete.empty:
			grid_geo_data_geo_filitered_complete = grid_geo_data_geo_filitered
		else:
			grid_geo_data_geo_filitered_complete = pd.concat([grid_geo_data_geo_filitered_complete, grid_geo_data_geo_filitered], ignore_index=True)
	
	print(f"\tIn total there are;\n\t{interference_grids_per_cell_complete.shape[0]} potential interference cells\n\t{grid_geo_data_geo_filitered_complete.shape[0]} potential interference Grid-Cells\n\t{grid_geo_data_geo_filitered_complete.drop_duplicates('grid').shape[0]} potential interference Grids\n")

	# Order and reduce list to a maximum of 10% of cells
	#interference_grids_per_cell_complete['score'] = 
	#max_number_of_interference_cells interference_grids_per_cell_complete

	return interference_grids_per_cell_complete, grid_geo_data_geo_filitered_complete



