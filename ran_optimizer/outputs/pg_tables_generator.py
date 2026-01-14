"""
PostgreSQL-ready output file generator for RAN Optimizer.

Generates 6 output files with strict schemas for production system integration.
Initially outputs to CSV, designed for future PostgreSQL integration.

Output Files:
1. daily_overshooter_resolution_recommendations.csv - Tilt recommendations
2. daily_overshooter_tier_3_recommendations.csv - Top 5 neighbors per overshooter
3. daily_pci_level_1_recommendations.csv - 1-hop PCI collisions
4. daily_pci_level_2_neighbor_blacklisting_recommendations.csv - Blacklist recommendations
5. daily_pci_level_2_recommendations.csv - 2-hop collisions + confusions
6. daily_pci_recommendations.csv - Combined PCI summary
"""

import os
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_date_ddmmyyyy(dt: Optional[datetime] = None) -> str:
    """Format date as dd/mm/yyyy string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%d/%m/%Y')


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in meters.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def create_cilac_lookup(cell_gis_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create a lookup dictionary from cell_name to cell_cilac.

    Args:
        cell_gis_df: GIS dataframe with cell_name and cell_cilac columns

    Returns:
        Dictionary mapping cell_name -> cell_cilac
    """
    lookup = {}
    if 'cell_name' in cell_gis_df.columns and 'cell_cilac' in cell_gis_df.columns:
        for _, row in cell_gis_df.iterrows():
            if pd.notna(row['cell_name']) and pd.notna(row['cell_cilac']):
                lookup[row['cell_name']] = int(row['cell_cilac'])
    return lookup


def create_cell_coords_lookup(cell_gis_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Create a lookup dictionary from cell_name to (latitude, longitude).

    Args:
        cell_gis_df: GIS dataframe with cell_name, latitude, longitude columns

    Returns:
        Dictionary mapping cell_name -> (latitude, longitude)
    """
    lookup = {}
    if 'cell_name' in cell_gis_df.columns and 'latitude' in cell_gis_df.columns:
        for _, row in cell_gis_df.iterrows():
            if pd.notna(row['cell_name']) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
                lookup[row['cell_name']] = (float(row['latitude']), float(row['longitude']))
    return lookup


def enrich_with_cilac(
    df: pd.DataFrame,
    cell_gis_df: pd.DataFrame,
    cell_name_col: str = 'cell_name',
    cilac_col: str = 'cilac'
) -> pd.DataFrame:
    """
    Enrich a dataframe with CILAC by joining with cell_gis.

    Args:
        df: DataFrame to enrich
        cell_gis_df: GIS dataframe with cell_name and cell_cilac
        cell_name_col: Column name in df containing cell names
        cilac_col: Output column name for CILAC

    Returns:
        DataFrame with CILAC column added
    """
    if cell_name_col not in df.columns:
        logger.warning("cilac_enrichment_skipped", reason=f"column {cell_name_col} not found")
        df[cilac_col] = None
        return df

    cilac_lookup = create_cilac_lookup(cell_gis_df)
    df[cilac_col] = df[cell_name_col].map(cilac_lookup)

    return df


# =============================================================================
# PG TABLES GENERATOR CLASS
# =============================================================================

class PGTablesGenerator:
    """
    Generator for PostgreSQL-ready output files.

    Produces 6 CSV files with strict schemas matching production database tables.
    """

    def __init__(
        self,
        output_dir: str,
        cell_gis_df: pd.DataFrame,
        cell_coverage_df: Optional[pd.DataFrame] = None,
        cell_impacts_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the generator.

        Args:
            output_dir: Base output directory (pg_tables subfolder will be created)
            cell_gis_df: GIS data with cell configurations
            cell_coverage_df: Coverage/measurement data
            cell_impacts_df: Cell impact/handover relations data
        """
        self.output_dir = Path(output_dir) / 'pg_tables'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cell_gis_df = cell_gis_df
        self.cell_coverage_df = cell_coverage_df
        self.cell_impacts_df = cell_impacts_df

        # Create lookups
        self.cilac_lookup = create_cilac_lookup(cell_gis_df)
        self.coords_lookup = create_cell_coords_lookup(cell_gis_df)

        # Create GIS lookup for tilt data
        self.gis_lookup = self._create_gis_lookup()

        # Analysis date (consistent across all outputs)
        self.analysis_date = format_date_ddmmyyyy()

        logger.info(
            "pg_tables_generator_initialized",
            output_dir=str(self.output_dir),
            cilac_count=len(self.cilac_lookup),
            coords_count=len(self.coords_lookup),
        )

    def _create_gis_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Create lookup for GIS data by cell_name."""
        lookup = {}
        for _, row in self.cell_gis_df.iterrows():
            cell_name = row.get('cell_name')
            if pd.notna(cell_name):
                lookup[cell_name] = {
                    'cell_cilac': row.get('cell_cilac'),
                    'tilt_elc': row.get('tilt_elc', 0),
                    'tilt_mech': row.get('tilt_mech', 0),
                    'min_tilt': row.get('min_tilt', 0),
                    'max_tilt': row.get('max_tilt', 15),
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                }
        return lookup

    def _get_neighbor_relation(self, from_cell: str, to_cell: str) -> str:
        """
        Get neighbor relation status from cell_impacts.

        Returns: 'Y', 'N', or 'N/A'
        """
        if self.cell_impacts_df is None:
            return 'N/A'

        mask = (
            (self.cell_impacts_df['cell_name'] == from_cell) &
            (self.cell_impacts_df['cell_impact_name'] == to_cell)
        )
        matches = self.cell_impacts_df[mask]

        if len(matches) == 0:
            return 'N/A'

        relation = matches.iloc[0].get('neighbor_relation', 'N/A')
        if pd.isna(relation):
            return 'N/A'
        return str(relation)

    def _calculate_distance(self, cell1: str, cell2: str) -> float:
        """Calculate distance between two cells using haversine."""
        coords1 = self.coords_lookup.get(cell1)
        coords2 = self.coords_lookup.get(cell2)

        if coords1 is None or coords2 is None:
            return 0.0

        return haversine_distance(coords1[0], coords1[1], coords2[0], coords2[1])

    # =========================================================================
    # FILE 1: daily_overshooter_resolution_recommendations.csv
    # =========================================================================

    def generate_overshooter_resolution(
        self,
        overshooting_df: Optional[pd.DataFrame],
        undershooting_df: Optional[pd.DataFrame],
        overshooting_grids_df: Optional[pd.DataFrame] = None,
        undershooting_grids_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate File 1: daily_overshooter_resolution_recommendations.csv

        Contains all overshooter and undershooting tilt recommendations.
        """
        records = []

        # Process overshooters
        if overshooting_df is not None and len(overshooting_df) > 0:
            for _, row in overshooting_df.iterrows():
                cell_name = row.get('cell_name')
                if pd.isna(cell_name):
                    continue

                gis_data = self.gis_lookup.get(cell_name, {})

                # Get tilt recommendation
                tilt_change = row.get('recommended_downtilt_deg', row.get('recommended_tilt_change', 0))
                if pd.isna(tilt_change):
                    tilt_change = 0

                current_tilt = gis_data.get('tilt_elc', 0) or 0
                min_tilt = gis_data.get('min_tilt', 0) or 0
                max_tilt = gis_data.get('max_tilt', 15) or 15

                # Calculate new tilt (bounded)
                new_tilt = current_tilt + abs(tilt_change)  # Downtilt increases tilt
                new_tilt = max(min_tilt, min(max_tilt, new_tilt))

                # Calculate tier 3 metrics
                tier3_metrics = self._calculate_tier3_metrics(
                    cell_name, overshooting_grids_df, 'overshooting'
                )

                records.append({
                    'analysisdate': self.analysis_date,
                    'cell_name': cell_name,
                    'cilac': gis_data.get('cell_cilac'),
                    'parameter': 'Manual_tilt',
                    'category': 'OverShooter',
                    'parameter_new_value': round(new_tilt, 1),
                    'cycle_start_date': self.analysis_date,
                    'cycle_end_date': self.analysis_date,
                    'cycle_status': 'PENDING',
                    'conditions': 'AUTO_GENERATED',
                    'current_tilt': round(current_tilt, 1),
                    'min_tilt': round(min_tilt, 1),
                    'max_tilt': round(max_tilt, 1),
                    'tier_3_sectors_count': tier3_metrics['cells_count'],
                    'tier_3_cells_count': tier3_metrics['cells_count'],
                    'tier_3_traffic_total': tier3_metrics['traffic_total'],
                    'tier_3_drops_total': 0,
                    'tier3_traffic_perc': round(tier3_metrics['traffic_perc'], 2),
                    'tier3_drops_perc': 0,
                })

        # Process undershooters
        if undershooting_df is not None and len(undershooting_df) > 0:
            for _, row in undershooting_df.iterrows():
                cell_name = row.get('cell_name')
                if pd.isna(cell_name):
                    continue

                gis_data = self.gis_lookup.get(cell_name, {})

                # Get tilt recommendation
                tilt_change = row.get('recommended_uptilt_deg', row.get('recommended_tilt_change', 0))
                if pd.isna(tilt_change):
                    tilt_change = 0

                current_tilt = gis_data.get('tilt_elc', 0) or 0
                min_tilt = gis_data.get('min_tilt', 0) or 0
                max_tilt = gis_data.get('max_tilt', 15) or 15

                # Calculate new tilt (bounded) - uptilt decreases tilt
                new_tilt = current_tilt - abs(tilt_change)
                new_tilt = max(min_tilt, min(max_tilt, new_tilt))

                # Calculate tier 3 metrics
                tier3_metrics = self._calculate_tier3_metrics(
                    cell_name, undershooting_grids_df, 'undershooting'
                )

                records.append({
                    'analysisdate': self.analysis_date,
                    'cell_name': cell_name,
                    'cilac': gis_data.get('cell_cilac'),
                    'parameter': 'Manual_tilt',
                    'category': 'UnderShooter',
                    'parameter_new_value': round(new_tilt, 1),
                    'cycle_start_date': self.analysis_date,
                    'cycle_end_date': self.analysis_date,
                    'cycle_status': 'PENDING',
                    'conditions': 'AUTO_GENERATED',
                    'current_tilt': round(current_tilt, 1),
                    'min_tilt': round(min_tilt, 1),
                    'max_tilt': round(max_tilt, 1),
                    'tier_3_sectors_count': tier3_metrics['cells_count'],
                    'tier_3_cells_count': tier3_metrics['cells_count'],
                    'tier_3_traffic_total': tier3_metrics['traffic_total'],
                    'tier_3_drops_total': 0,
                    'tier3_traffic_perc': round(tier3_metrics['traffic_perc'], 2),
                    'tier3_drops_perc': 0,
                })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_overshooter_resolution_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "overshooter_resolution_generated",
            path=str(output_path),
            records=len(df),
            overshooters=len(overshooting_df) if overshooting_df is not None else 0,
            undershooters=len(undershooting_df) if undershooting_df is not None else 0,
        )

        return df

    def _calculate_tier3_metrics(
        self,
        cell_name: str,
        grids_df: Optional[pd.DataFrame],
        detection_type: str
    ) -> Dict[str, Any]:
        """Calculate tier 3 metrics for a cell."""
        result = {
            'cells_count': 0,
            'traffic_total': 0,
            'traffic_perc': 0.0,
        }

        if grids_df is None or self.cell_coverage_df is None:
            return result

        # Find cell name column
        cell_col = 'cell_name' if 'cell_name' in grids_df.columns else 'cilac'
        grid_col = 'geohash7' if 'geohash7' in grids_df.columns else 'grid'

        if cell_col not in grids_df.columns or grid_col not in grids_df.columns:
            return result

        # Get edge grids for this cell
        cell_grids = grids_df[grids_df[cell_col] == cell_name]
        if len(cell_grids) == 0:
            return result

        edge_grids = cell_grids[grid_col].unique()

        # Find coverage column names
        cov_cell_col = 'cell_name' if 'cell_name' in self.cell_coverage_df.columns else 'cilac'
        cov_grid_col = 'grid' if 'grid' in self.cell_coverage_df.columns else 'geohash7'
        event_col = 'event_count' if 'event_count' in self.cell_coverage_df.columns else 'total_traffic'
        perc_col = 'perc_cell_events' if 'perc_cell_events' in self.cell_coverage_df.columns else None

        # Filter coverage to edge grids
        edge_coverage = self.cell_coverage_df[
            self.cell_coverage_df[cov_grid_col].isin(edge_grids)
        ]

        if len(edge_coverage) == 0:
            return result

        # Count distinct cells (excluding the source cell)
        other_cells = edge_coverage[edge_coverage[cov_cell_col] != cell_name]
        result['cells_count'] = other_cells[cov_cell_col].nunique()

        # Sum traffic for the source cell in edge grids
        source_traffic = edge_coverage[edge_coverage[cov_cell_col] == cell_name]
        if event_col in source_traffic.columns:
            result['traffic_total'] = int(source_traffic[event_col].sum())

        # Calculate traffic percentage
        if perc_col and perc_col in source_traffic.columns:
            result['traffic_perc'] = float(source_traffic[perc_col].sum())

        return result

    # =========================================================================
    # FILE 2: daily_overshooter_tier_3_recommendations.csv
    # =========================================================================

    def generate_tier3_neighbors(
        self,
        overshooting_df: Optional[pd.DataFrame],
        overshooting_grids_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate File 2: daily_overshooter_tier_3_recommendations.csv

        Contains top 5 neighbor cells for each overshooter based on traffic in edge grids.
        """
        records = []

        if overshooting_df is None or len(overshooting_df) == 0:
            df = pd.DataFrame(records)
            output_path = self.output_dir / 'daily_overshooter_tier_3_recommendations.csv'
            df.to_csv(output_path, index=False)
            return df

        for _, row in overshooting_df.iterrows():
            cell_name = row.get('cell_name')
            if pd.isna(cell_name):
                continue

            # Get top 5 neighbors
            top_neighbors = self._get_top_neighbors(cell_name, overshooting_grids_df, top_n=5)

            # Get tier 3 metrics for the overshooter
            tier3_metrics = self._calculate_tier3_metrics(
                cell_name, overshooting_grids_df, 'overshooting'
            )

            from_cilac = self.cilac_lookup.get(cell_name)

            for neighbor in top_neighbors:
                to_cilac = self.cilac_lookup.get(neighbor['cell_name'])
                distance = self._calculate_distance(cell_name, neighbor['cell_name'])
                neighbor_relation = self._get_neighbor_relation(cell_name, neighbor['cell_name'])

                records.append({
                    'analysisdate': self.analysis_date,
                    'fromname': cell_name,
                    'fromcilac': from_cilac,
                    'toname': neighbor['cell_name'],
                    'tocilac': to_cilac,
                    'fromtodistance': round(distance, 2),
                    'tier_3_traffic': tier3_metrics['traffic_total'],
                    'tier_3_drops': 0,
                    'tier3_traffic_perc': round(tier3_metrics['traffic_perc'], 2),
                    'tier3_drops_perc': 0,
                    'is_neighbored': neighbor_relation,
                })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_overshooter_tier_3_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "tier3_neighbors_generated",
            path=str(output_path),
            records=len(df),
            overshooters=len(overshooting_df),
        )

        return df

    def _get_top_neighbors(
        self,
        cell_name: str,
        grids_df: Optional[pd.DataFrame],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top N neighbor cells by traffic in edge grids."""
        if grids_df is None or self.cell_coverage_df is None:
            return []

        # Find column names
        cell_col = 'cell_name' if 'cell_name' in grids_df.columns else 'cilac'
        grid_col = 'geohash7' if 'geohash7' in grids_df.columns else 'grid'

        if cell_col not in grids_df.columns or grid_col not in grids_df.columns:
            return []

        # Get edge grids for this cell
        cell_grids = grids_df[grids_df[cell_col] == cell_name]
        if len(cell_grids) == 0:
            return []

        edge_grids = cell_grids[grid_col].unique()

        # Find coverage column names
        cov_cell_col = 'cell_name' if 'cell_name' in self.cell_coverage_df.columns else 'cilac'
        cov_grid_col = 'grid' if 'grid' in self.cell_coverage_df.columns else 'geohash7'
        event_col = 'event_count' if 'event_count' in self.cell_coverage_df.columns else 'total_traffic'

        # Filter coverage to edge grids and exclude source cell
        edge_coverage = self.cell_coverage_df[
            (self.cell_coverage_df[cov_grid_col].isin(edge_grids)) &
            (self.cell_coverage_df[cov_cell_col] != cell_name)
        ]

        if len(edge_coverage) == 0:
            return []

        # Sum traffic per cell
        if event_col not in edge_coverage.columns:
            return []

        cell_traffic = edge_coverage.groupby(cov_cell_col)[event_col].sum().reset_index()
        cell_traffic = cell_traffic.sort_values(event_col, ascending=False).head(top_n)

        return [
            {'cell_name': row[cov_cell_col], 'traffic': row[event_col]}
            for _, row in cell_traffic.iterrows()
        ]

    # =========================================================================
    # FILE 3: daily_pci_level_1_recommendations.csv
    # =========================================================================

    def generate_pci_level1(
        self,
        pci_collisions_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate File 3: daily_pci_level_1_recommendations.csv

        Contains 1-hop PCI collisions only.
        """
        records = []

        if pci_collisions_df is None or len(pci_collisions_df) == 0:
            df = pd.DataFrame(records)
            output_path = self.output_dir / 'daily_pci_level_1_recommendations.csv'
            df.to_csv(output_path, index=False)
            return df

        # Filter to 1-hop collisions only
        hop_col = 'hop_type' if 'hop_type' in pci_collisions_df.columns else None
        if hop_col:
            one_hop = pci_collisions_df[pci_collisions_df[hop_col] == '1-hop']
        else:
            one_hop = pci_collisions_df

        for _, row in one_hop.iterrows():
            cell_a = row.get('cell_a')
            cell_b = row.get('cell_b')

            if pd.isna(cell_a) or pd.isna(cell_b):
                continue

            # Get CILAC values
            cilac_a = self.cilac_lookup.get(cell_a)
            cilac_b = self.cilac_lookup.get(cell_b)

            # Get cell_impacts data if available
            impact_data = self._get_impact_data(cell_a, cell_b)

            records.append({
                'insertdatetime': self.analysis_date,
                'sum_drops': impact_data.get('drops_voice', 0),
                'sum_impact': impact_data.get('traffic_data', 0),
                'distance': round(impact_data.get('distance', 0), 2),
                'cilac': cilac_a,
                'nextcilac': cilac_b,
                'neighbourindex': 1 if impact_data.get('neighbor_relation') == 'Y' else 0,
                'tosectorsc1': row.get('pci_a'),
                'tosectorsc2': row.get('pci_b'),
            })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_pci_level_1_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "pci_level1_generated",
            path=str(output_path),
            records=len(df),
        )

        return df

    def _get_impact_data(self, from_cell: str, to_cell: str) -> Dict[str, Any]:
        """Get impact data from cell_impacts for a cell pair."""
        result = {
            'traffic_data': 0,
            'drops_voice': 0,
            'distance': 0,
            'neighbor_relation': 'N/A',
        }

        if self.cell_impacts_df is None:
            # Calculate distance from coordinates
            result['distance'] = self._calculate_distance(from_cell, to_cell)
            return result

        mask = (
            (self.cell_impacts_df['cell_name'] == from_cell) &
            (self.cell_impacts_df['cell_impact_name'] == to_cell)
        )
        matches = self.cell_impacts_df[mask]

        if len(matches) == 0:
            # Try reverse direction
            mask = (
                (self.cell_impacts_df['cell_name'] == to_cell) &
                (self.cell_impacts_df['cell_impact_name'] == from_cell)
            )
            matches = self.cell_impacts_df[mask]

        if len(matches) > 0:
            row = matches.iloc[0]
            result['traffic_data'] = row.get('traffic_data', 0) or 0
            result['drops_voice'] = row.get('drops_voice', 0) or 0
            result['distance'] = row.get('distance', 0) or 0
            result['neighbor_relation'] = row.get('neighbor_relation', 'N/A')
        else:
            result['distance'] = self._calculate_distance(from_cell, to_cell)

        return result

    # =========================================================================
    # FILE 4: daily_pci_level_2_neighbor_blacklisting_recommendations.csv
    # =========================================================================

    def generate_pci_blacklisting(
        self,
        pci_blacklist_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate File 4: daily_pci_level_2_neighbor_blacklisting_recommendations.csv

        Contains PCI blacklist recommendations.
        """
        records = []

        if pci_blacklist_df is None or len(pci_blacklist_df) == 0:
            df = pd.DataFrame(records)
            output_path = self.output_dir / 'daily_pci_level_2_neighbor_blacklisting_recommendations.csv'
            df.to_csv(output_path, index=False)
            return df

        for _, row in pci_blacklist_df.iterrows():
            serving = row.get('serving')
            neighbor = row.get('neighbor')  # Cell to be blacklisted

            if pd.isna(serving) or pd.isna(neighbor):
                continue

            # Get CILAC
            serving_cilac = self.cilac_lookup.get(serving)

            # Get impact data for blacklisted cell
            bl_impact = self._get_impact_data(serving, neighbor)

            # Find the "other" cell (non-blacklisted) from confusion group if available
            other_cell = None
            other_impact = {'traffic_data': 0, 'distance': 0}
            other_pci = None

            # For now, use confusion PCI as the blacklist PCI
            confusion_pci = row.get('confusion_pci')

            records.append({
                'insertdatetime': self.analysis_date,
                'cilac': serving_cilac,
                'pci': confusion_pci,
                'bl': 0,
                'colision_type': 'Confusion',
                'adj_weight': bl_impact.get('traffic_data', 0),
                'other_adj_weight': other_impact.get('traffic_data', 0),
                'adjacency': neighbor,
                'parameter': 'blacklist',
                'value': 1,
                'sector_sc_other': other_pci,
                'distancefrom': round(other_impact.get('distance', 0), 2),
                'distanceto': round(bl_impact.get('distance', 0), 2),
            })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_pci_level_2_neighbor_blacklisting_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "pci_blacklisting_generated",
            path=str(output_path),
            records=len(df),
        )

        return df

    # =========================================================================
    # FILE 5: daily_pci_level_2_recommendations.csv
    # =========================================================================

    def generate_pci_level2(
        self,
        pci_collisions_df: Optional[pd.DataFrame],
        pci_confusions_df: Optional[pd.DataFrame],
        pci_blacklist_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate File 5: daily_pci_level_2_recommendations.csv

        Contains 2-hop collisions and confusions.
        """
        records = []

        # Process 2-hop collisions
        if pci_collisions_df is not None and len(pci_collisions_df) > 0:
            hop_col = 'hop_type' if 'hop_type' in pci_collisions_df.columns else None
            if hop_col:
                two_hop = pci_collisions_df[pci_collisions_df[hop_col] == '2-hop']
            else:
                two_hop = pd.DataFrame()

            for _, row in two_hop.iterrows():
                cell_a = row.get('cell_a')
                cell_b = row.get('cell_b')

                if pd.isna(cell_a) or pd.isna(cell_b):
                    continue

                # For 2-hop, need to identify serving cell (middle cell)
                # For now, use cell_a as serving cell
                serving_cilac = self.cilac_lookup.get(cell_a)

                impact1 = self._get_impact_data(cell_a, cell_b)
                impact2 = {'traffic_data': 0, 'distance': 0, 'neighbor_relation': 'N'}

                cilac_a = self.cilac_lookup.get(cell_a)
                cilac_b = self.cilac_lookup.get(cell_b)

                # Check if this is a blacklist candidate
                is_bl_candidate = 0
                if pci_blacklist_df is not None:
                    bl_check = pci_blacklist_df[
                        (pci_blacklist_df['serving'] == cell_a) |
                        (pci_blacklist_df['serving'] == cell_b)
                    ]
                    is_bl_candidate = 1 if len(bl_check) > 0 else 0

                records.append({
                    'insertdatetime': self.analysis_date,
                    'fromcilac': serving_cilac,
                    'shoa': 0,
                    'tosectorsc1': row.get('pci_a'),
                    'impact1': impact1.get('traffic_data', 0),
                    'neighbour1': 1 if impact1.get('neighbor_relation') == 'Y' else 0,
                    'shob': 0,
                    'tosectorsc2': row.get('pci_b'),
                    'impact2': impact2.get('traffic_data', 0),
                    'neighbour2': 1 if impact2.get('neighbor_relation') == 'Y' else 0,
                    'distancefrom': round(impact1.get('distance', 0), 2),
                    'distanceto': round(impact2.get('distance', 0), 2),
                    'cilac': cilac_a,
                    'nextcilac': cilac_b,
                    'neibour2sector': 1 if (impact1.get('neighbor_relation') == 'Y' and
                                            impact2.get('neighbor_relation') == 'Y') else 0,
                    'totalsho': 0,
                    'l2_nr_bl_candidate': is_bl_candidate,
                })

        # Process confusions (add if pci_confusions_df available)
        if pci_confusions_df is not None and len(pci_confusions_df) > 0:
            for _, row in pci_confusions_df.iterrows():
                serving = row.get('serving')
                if pd.isna(serving):
                    continue

                serving_cilac = self.cilac_lookup.get(serving)
                confusion_pci = row.get('confusion_pci')

                # Parse neighbors if available
                neighbors_str = row.get('neighbors', '')
                neighbors = [n.strip() for n in str(neighbors_str).split(',') if n.strip()]

                if len(neighbors) >= 2:
                    cell1, cell2 = neighbors[0], neighbors[1]
                    impact1 = self._get_impact_data(serving, cell1)
                    impact2 = self._get_impact_data(serving, cell2)

                    cilac_1 = self.cilac_lookup.get(cell1)
                    cilac_2 = self.cilac_lookup.get(cell2)

                    is_bl_candidate = 0
                    if pci_blacklist_df is not None:
                        bl_check = pci_blacklist_df[pci_blacklist_df['serving'] == serving]
                        is_bl_candidate = 1 if len(bl_check) > 0 else 0

                    records.append({
                        'insertdatetime': self.analysis_date,
                        'fromcilac': serving_cilac,
                        'shoa': 0,
                        'tosectorsc1': confusion_pci,
                        'impact1': impact1.get('traffic_data', 0),
                        'neighbour1': 1 if impact1.get('neighbor_relation') == 'Y' else 0,
                        'shob': 0,
                        'tosectorsc2': confusion_pci,
                        'impact2': impact2.get('traffic_data', 0),
                        'neighbour2': 1 if impact2.get('neighbor_relation') == 'Y' else 0,
                        'distancefrom': round(impact1.get('distance', 0), 2),
                        'distanceto': round(impact2.get('distance', 0), 2),
                        'cilac': cilac_1,
                        'nextcilac': cilac_2,
                        'neibour2sector': 1 if (impact1.get('neighbor_relation') == 'Y' and
                                                impact2.get('neighbor_relation') == 'Y') else 0,
                        'totalsho': 0,
                        'l2_nr_bl_candidate': is_bl_candidate,
                    })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_pci_level_2_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "pci_level2_generated",
            path=str(output_path),
            records=len(df),
        )

        return df

    # =========================================================================
    # FILE 6: daily_pci_recommendations.csv
    # =========================================================================

    def generate_pci_combined(
        self,
        pci_collisions_df: Optional[pd.DataFrame],
        pci_confusions_df: Optional[pd.DataFrame],
        pci_blacklist_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate File 6: daily_pci_recommendations.csv

        Combined PCI summary with strongest/nearest cell logic.
        """
        records = []

        # Process 1-hop collisions
        if pci_collisions_df is not None and len(pci_collisions_df) > 0:
            hop_col = 'hop_type' if 'hop_type' in pci_collisions_df.columns else None

            for _, row in pci_collisions_df.iterrows():
                cell_a = row.get('cell_a')
                cell_b = row.get('cell_b')

                if pd.isna(cell_a) or pd.isna(cell_b):
                    continue

                is_1hop = True
                if hop_col:
                    is_1hop = row.get(hop_col) == '1-hop'

                cilac_a = self.cilac_lookup.get(cell_a)
                cilac_b = self.cilac_lookup.get(cell_b)
                impact = self._get_impact_data(cell_a, cell_b)

                # Check if blacklist
                is_bl = 0
                if pci_blacklist_df is not None:
                    bl_check = pci_blacklist_df[
                        (pci_blacklist_df['serving'] == cell_a) |
                        (pci_blacklist_df['neighbor'] == cell_a)
                    ]
                    is_bl = 1 if len(bl_check) > 0 else 0

                records.append({
                    'analysisdate': self.analysis_date,
                    'cilac': cilac_a,
                    'pci': row.get('pci_b'),
                    'strongestsectorweight': impact.get('traffic_data', 0),
                    'bl': is_bl,
                    'nearestcilac': cilac_b,
                    'strongestcilac': cilac_b,
                    'distance_near': round(impact.get('distance', 0), 2),
                    'distance_strong': round(impact.get('distance', 0), 2),
                    'totalweight': impact.get('traffic_data', 0),
                    'is_level1': 'Yes' if is_1hop else 'No',
                    'regionid': 0,
                })

        # Process confusions
        if pci_confusions_df is not None and len(pci_confusions_df) > 0:
            for _, row in pci_confusions_df.iterrows():
                serving = row.get('serving')
                if pd.isna(serving):
                    continue

                serving_cilac = self.cilac_lookup.get(serving)
                confusion_pci = row.get('confusion_pci')

                # Parse neighbors
                neighbors_str = row.get('neighbors', '')
                neighbors = [n.strip() for n in str(neighbors_str).split(',') if n.strip()]

                if len(neighbors) == 0:
                    continue

                # Find strongest and nearest
                strongest_cell = None
                strongest_traffic = 0
                nearest_cell = None
                nearest_distance = float('inf')
                total_weight = 0

                for neighbor in neighbors:
                    impact = self._get_impact_data(serving, neighbor)
                    traffic = impact.get('traffic_data', 0)
                    distance = impact.get('distance', 0)
                    total_weight += traffic

                    if traffic > strongest_traffic:
                        strongest_traffic = traffic
                        strongest_cell = neighbor

                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_cell = neighbor

                strongest_cilac = self.cilac_lookup.get(strongest_cell) if strongest_cell else None
                nearest_cilac = self.cilac_lookup.get(nearest_cell) if nearest_cell else None

                # Check if blacklist
                is_bl = 0
                if pci_blacklist_df is not None:
                    bl_check = pci_blacklist_df[pci_blacklist_df['serving'] == serving]
                    is_bl = 1 if len(bl_check) > 0 else 0

                records.append({
                    'analysisdate': self.analysis_date,
                    'cilac': serving_cilac,
                    'pci': confusion_pci,
                    'strongestsectorweight': strongest_traffic,
                    'bl': is_bl,
                    'nearestcilac': nearest_cilac,
                    'strongestcilac': strongest_cilac,
                    'distance_near': round(nearest_distance, 2) if nearest_distance != float('inf') else 0,
                    'distance_strong': round(self._get_impact_data(serving, strongest_cell).get('distance', 0), 2) if strongest_cell else 0,
                    'totalweight': total_weight,
                    'is_level1': 'No',
                    'regionid': 0,
                })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = self.output_dir / 'daily_pci_recommendations.csv'
        df.to_csv(output_path, index=False)

        logger.info(
            "pci_combined_generated",
            path=str(output_path),
            records=len(df),
        )

        return df

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(
        self,
        overshooting_df: Optional[pd.DataFrame] = None,
        undershooting_df: Optional[pd.DataFrame] = None,
        overshooting_grids_df: Optional[pd.DataFrame] = None,
        undershooting_grids_df: Optional[pd.DataFrame] = None,
        pci_collisions_df: Optional[pd.DataFrame] = None,
        pci_confusions_df: Optional[pd.DataFrame] = None,
        pci_blacklist_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all 6 PostgreSQL-ready output files.

        Returns:
            Dictionary with output file names as keys and DataFrames as values.
        """
        logger.info("generating_all_pg_tables")

        results = {}

        # File 1: Overshooter Resolution
        results['overshooter_resolution'] = self.generate_overshooter_resolution(
            overshooting_df=overshooting_df,
            undershooting_df=undershooting_df,
            overshooting_grids_df=overshooting_grids_df,
            undershooting_grids_df=undershooting_grids_df,
        )

        # File 2: Tier 3 Neighbors
        results['tier3_neighbors'] = self.generate_tier3_neighbors(
            overshooting_df=overshooting_df,
            overshooting_grids_df=overshooting_grids_df,
        )

        # File 3: PCI Level 1
        results['pci_level1'] = self.generate_pci_level1(
            pci_collisions_df=pci_collisions_df,
        )

        # File 4: PCI Blacklisting
        results['pci_blacklisting'] = self.generate_pci_blacklisting(
            pci_blacklist_df=pci_blacklist_df,
        )

        # File 5: PCI Level 2
        results['pci_level2'] = self.generate_pci_level2(
            pci_collisions_df=pci_collisions_df,
            pci_confusions_df=pci_confusions_df,
            pci_blacklist_df=pci_blacklist_df,
        )

        # File 6: PCI Combined
        results['pci_combined'] = self.generate_pci_combined(
            pci_collisions_df=pci_collisions_df,
            pci_confusions_df=pci_confusions_df,
            pci_blacklist_df=pci_blacklist_df,
        )

        logger.info(
            "all_pg_tables_generated",
            output_dir=str(self.output_dir),
            files_generated=len(results),
            total_records=sum(len(df) for df in results.values()),
        )

        return results
