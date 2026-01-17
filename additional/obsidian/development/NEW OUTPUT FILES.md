
There are 6 recommendation files with a strict schema to be enforced that we need to produce for a production system, for now we will write these to csv but eventually they will be written to Postgres. Below provides information on the structure of the 6 files. I have introduced ‘cell_cilac’, ‘min_tilt’, ‘max_tilt’ into ‘cell_gis’ to ensure all data is available. 

I would like to these fees to sit in folder '/Users/stephendillon/Library/Mobile Documents/com~apple~CloudDocs/repos/mobile-network-ran/5-radcom-recommendations/data/vf-ie/output-data/pg_tables'

Output file: daily_overshooter_resolution_recommendations.csv

We have already defined this output (see '/Users/stephendillon/Library/Mobile Documents/com~apple~CloudDocs/repos/mobile-network-ran/5-radcom-recommendations/data/vf-ie/output-data/daily_overshooter_resolution_recommendations.csv') BUT we need to update to the below;

The file will contain all overshooter and undershooter recommendation cells

Fields:
‘analysisdate’: Date analysis was completed (format dd/mm/yyyy)
‘cell_name’: ‘cell_name’ in the same format as file ‘cell_gis’
‘cilac’: ‘cell_cilac’ in the same format as file ‘cell_gis’
‘parameter’: set to ‘Manual_tilt’ for now
‘category’: ‘OverShooter’ OR ‘UnderShooter’ depending on whether the cell was flagged in the ‘overhooter.py’ or ‘undershoot.py’ detector
‘parameter_new_value’: cells ‘tilt_elc’ from ‘cell_gis’ + recommendation (+/- 1-2 degrees tilt) in the bounds ‘min_tilt’ - ‘max_tilt’ in ‘cell_gis’
‘cycle_start_date’:  Date analysis was completed (format dd/mm/yyyy)
‘cycle_end_date’:  Date analysis was completed (format dd/mm/yyyy)
‘cycle_status’: set to ‘PENDING’ for now
‘conditions’: set to ‘AUTO_GENERATED’ for now
‘current_tilt’: ‘elec_tilt’ from ‘cell_gis’
‘min_tilt’: ‘min_tilt’ from ‘cell_gis’
‘max_tilt’: ‘max_tilt’ from ‘cell_gis’
‘tier_3_sectors_count’: Distinct count of ‘cell_name’ in ‘cell_coverage’ identified cell edge bins (highlighted in the algorithm and set by ‘edge_traffic_percent’) 
‘tier_3_cells_count’: Distinct count of ‘cell_name’ in ‘cell_coverage’ identified cell edge bins (highlighted in the algorithm and set by ‘edge_traffic_percent’) 
‘tier_3_traffic_total’: Sum of ‘event_count’ in ‘cell_coverage’ in the identified cell edge bins for the source cell and associated edge grids (highlighted in the algorithm and set by ‘edge_traffic_percent’) 
‘tier_3_drops_total’: Set to 0 for now
‘tier3_traffic_perc’: Sum of ‘perc_cell_events’ in ‘cell_coverage’ in the identified cell edge bins for the source cell and associated edge grids (highlighted in the algorithm and set by ‘edge_traffic_percent’) 
‘tier3_drops_perc’: Set to 0 for now

Output file: daily_overshooter_tier_3_recommendations.csv
Fields:
‘analysisdate’: Date analysis was completed (format dd/mm/yyyy)
‘fromname’: overshooting ‘cell_name’
‘fromcilac’: ‘cell_cilac’ of the overshooting cell, available in ‘cell_gis’, field ‘cell_cilac’
‘toname’: ‘cell_name’ of cell from high weighted neighbours (see definition below)
‘tocilac’: ‘cell_cilac’ of cell from high weighted neighbours (see definition below)
‘fromtodistance’: ‘distance’ between cells fromname and toname
‘tier_3_traffic’: sum of ‘event_count’ in edge bins for the overshooting cell, note parameter ‘edge_traffic_percent’ defines which bins are included. Note ‘event_count’ is available in ‘cell_coverage’ but we should already have this data available to us without lookup
‘tier_3_drops’: for now set to 0
‘tier3_traffic_perc’: sum of ‘perc_cell_events’ in edge bins for the overshooting cell, note parameter ‘edge_traffic_percent’ defines which bins are included. Note ‘perc_cell_events’ is available in ‘cell_coverage’ but we should already have this data available to us without lookup
‘tier3_drops_perc’: for now set to 0
‘is_neighbored’: We can find this in ‘cell_impacts’ field ‘neighbor_relation’ will be ‘Y’, ’N’, ’N/A’. ’N/A’ means neighbour relation data is not available in the data sets provided

Definition of high weighted neighbours:
1. Identify edge grids for overshooting/undershooting cells (set via parameter ‘edge_traffic_percent’)
2. From ‘cell_coverage’ find the additional cells that serve these bins
3. Include the top 5 cells overall when summing ‘event_count’ over the edge grids


Output file: daily_pci_level_1_recommendations.csv
This includes 1-hop collisions only
Fields:
‘insertdatetime’: Date analysis was completed (format dd/mm/yyyy)
‘sum_drops’: ‘drops_voice’ from ‘cell_impacts’ 
‘sum_impact’: ‘traffic_data’  from ‘cell_impacts’
‘distance’: ‘distance’  from ‘cell_impacts’
‘cilac’: cilac of the source cell - ‘cell_cilac’ in cell_impacts data 
‘nextcilac’: cilac of the impact cell - ‘cell_impact_cilac’ in cell_impacts data
‘neighbourindex’: If field ‘neighbor_relation’ in ‘cell_impacts’ = ‘Y’ THEN 1, ELSE 0
‘tosectorsc1’: ‘cell_pci’ from ‘cell_impacts’
‘tosectorsc2’: ‘cell_impact_pci’ from ‘cell_impacts’


Output file: daily_pci_level_2_neighbor_blacklisting_recommendations.csv
This includes blacklist recommendations only
Fields:
'insertdatetime': Date analysis was completed (format dd/mm/yyyy)
‘cilac’: cilac of the source cell - ‘cell_cilac’
'pci': ’cell_impact_pci’ from ‘cell_impacts’ of cell to be blacklisted
'bl': for now set to 0
'colision_type': Set to ‘Confusion’
'adj_weight': ‘traffic_data’  from ‘cell_impacts’ for impacting cell to be black listed
'other_adj_weight': ‘traffic_data’  from ‘cell_impacts’ for impacting cell not to be black listed
'adjacency': ’cell_impact_name’ from ‘cell_impacts’ of cell to be blacklisted
'parameter': set to ‘blacklist’
'value': set to 1
'sector_sc_other': ’cell_impact_pci’ from ‘cell_impacts’ of cell not to be blacklisted
'distancefrom': ‘distance’ from ‘cell_impacts’ for impacting cell not to be black listed
'distanceto': ‘distance’ from ‘cell_impacts’ for impacting cell to be black listed


Output file: daily_pci_level_2_recommendations.csv
This includes pci confusions and 2-hop pci collisions
Fields:
'insertdatetime': Date analysis was completed (format dd/mm/yyyy)
'fromcilac': cilac of the impact cell - ‘cell_cilac’
'shoa': for now set to 0
'tosectorsc1': ’cell_impact_pci’ from ‘cell_impacts’ of impact cell 1
'impact1': ‘traffic_data’  from ‘cell_impacts’ for impact cell 1
'neighbour1': 1 if ‘neighbor_relation’ = ‘Y’ for impact else 0
'shob': for now set to 0
'tosectorsc2': ’cell_impact_pci’ from ‘cell_impacts’ of impact cell 2
'impact2': ‘traffic_data’  from ‘cell_impacts’ for impact cell 2
'neighbour2': 1 if ‘neighbor_relation’ = ‘Y’ for impact else 0
'distancefrom': ‘distance’ in ‘cell_impacts’ for impact pair - source cell to cell 1 
'distanceto': ‘distance’ in ‘cell_impacts’ for impact pair - source cell to cell 2
'cilac': ‘cell_impact_cilac’ of cell 1
'nextcilac': ‘cell_impact_cilac’ of cell 2
'neibour2sector': if ‘neighbour`’ and ‘neighbour2’ = 1 THEN 1 ELSE 0
'totalsho': 0 for now
'l2_nr_bl_candidate': 1 if blacklist recommendation, 0 oherwise

  
Output file: daily_pci_recommendations.csv
Fields:
'analysisdate': Date analysis was completed (format dd/mm/yyyy)
'cilac': cilac of the source cell - ‘cell_cilac’
'pci': ’cell_impact_pci’ from ‘cell_impacts’ of cell to be blacklisted
'strongestsectorweight': If 1-hop collision THEN ‘traffic_data’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then ‘traffic_data’  from ‘cell_impacts’ for strongest impact cell.
'bl': If this is a blacklst recommendation then 1 else 0
'nearestcilac': If 1-hop collision THEN ‘cell_impact_cilac’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then ‘cell_impact_cilac’  from ‘cell_impacts’ for nearest impact cell (based on field ‘distance’).
'strongestcilac': If 1-hop collision THEN ‘cell_impact_cilac’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then ‘cell_impact_cilac’  from ‘cell_impacts’ for strongest impact cell (based on field ‘traffic_data’).
'distance_near': If 1-hop collision THEN ‘distance’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then ‘distance’  from ‘cell_impacts’ for nearest impact cell (based on field ‘distance’).
'distance_strong': If 1-hop collision THEN ‘traffic_data’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then ‘traffic_data’  from ‘cell_impacts’ for strongest impact cell (based on field ‘traffic_data’).
'totalweight': If 1-hop collision THEN ‘traffic_data’  from ‘cell_impacts’ for impact cell. If 2-hop collision, confusion, blacklist then SUM ‘traffic_data’  for impact cell (based on field ‘traffic_data’).
'is_level1': If 1-hop collision THEN ‘Yes’ ELSE ’No’.
'regionid': set to 0 for now