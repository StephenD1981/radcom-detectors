"""
Environment classification for cells based on intersite distance.

Classifies cells as urban, suburban, or rural based on the average
distance to their nearest neighboring sites using KD-tree spatial indexing.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from scipy.spatial import cKDTree

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


# Default classification thresholds (km)
DEFAULT_URBAN_THRESHOLD_KM = 1.0      # <= 1km = urban
DEFAULT_SUBURBAN_THRESHOLD_KM = 3.0   # <= 3km = suburban, > 3km = rural

# Number of nearest sites to consider for average
DEFAULT_NEAREST_SITES_COUNT = 3

# Minimum distance to consider (filters out co-located sectors)
DEFAULT_MIN_SITE_DISTANCE_KM = 0.1  # 100m

# Approximate degrees to km conversion at Irish latitudes
DEGREES_TO_KM = 111.0


class EnvironmentClassifier:
    """
    Classifies cell environments based on intersite distance.

    Uses scipy's cKDTree for efficient nearest-neighbor queries.

    Environment types:
    - URBAN: Dense areas with mean intersite distance <= 1km
    - SUBURBAN: Medium density with mean intersite distance 1-3km
    - RURAL: Sparse areas with mean intersite distance > 3km

    Example:
        >>> classifier = EnvironmentClassifier(gis_df)
        >>> cell_environments = classifier.classify()
        >>> cell_environments.to_csv('cell_environment.csv', index=False)
    """

    def __init__(
        self,
        gis_df: pd.DataFrame,
        urban_threshold_km: float = DEFAULT_URBAN_THRESHOLD_KM,
        suburban_threshold_km: float = DEFAULT_SUBURBAN_THRESHOLD_KM,
        nearest_sites_count: int = DEFAULT_NEAREST_SITES_COUNT,
        min_site_distance_km: float = DEFAULT_MIN_SITE_DISTANCE_KM,
    ):
        """
        Initialize the classifier with GIS data.

        Args:
            gis_df: Cell GIS data with columns:
                - cell_name: Cell identifier (or cilac/CILAC)
                - latitude: Cell latitude (or Latitude)
                - longitude: Cell longitude (or Longitude)
            urban_threshold_km: Max intersite distance for urban (default 1.0km)
            suburban_threshold_km: Max intersite distance for suburban (default 3.0km)
            nearest_sites_count: Number of nearest sites to average (default 3)
            min_site_distance_km: Minimum distance to filter co-located sectors (default 0.1km)
        """
        self.gis_df = self._normalise_columns(gis_df.copy())
        self.urban_threshold_km = urban_threshold_km
        self.suburban_threshold_km = suburban_threshold_km
        self.nearest_sites_count = nearest_sites_count
        self.min_site_distance_km = min_site_distance_km

        # Extract unique sites
        self.sites_df = self._extract_unique_sites()

        logger.info(
            "Initialized EnvironmentClassifier",
            cells=len(self.gis_df),
            unique_sites=len(self.sites_df),
            urban_threshold_km=urban_threshold_km,
            suburban_threshold_km=suburban_threshold_km,
        )

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names to standard schema."""
        col_aliases = {
            'cell_name': ['cilac', 'CILAC'],
            'latitude': ['Latitude', 'lat'],
            'longitude': ['Longitude', 'lon'],
        }

        rename_map = {}
        for schema_col, aliases in col_aliases.items():
            if schema_col not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        rename_map[alias] = schema_col
                        break

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure cell_name is string
        if 'cell_name' in df.columns:
            df['cell_name'] = df['cell_name'].astype(str)

        return df

    def _extract_unique_sites(self) -> pd.DataFrame:
        """
        Extract unique sites from cell data.

        Sites are identified by unique lat/lon combinations (rounded to 6 decimals).
        """
        df = self.gis_df.copy()

        # Create site key from coordinates
        df['site_key'] = (
            df['latitude'].round(6).astype(str) + '_' +
            df['longitude'].round(6).astype(str)
        )

        # Get unique sites
        sites = df.groupby('site_key').agg({
            'latitude': 'first',
            'longitude': 'first',
        }).reset_index()

        logger.info(f"Extracted {len(sites)} unique sites from {len(df)} cells")

        return sites

    def _classify_environment(self, mean_intersite_km: float) -> str:
        """Classify environment based on mean intersite distance."""
        if mean_intersite_km <= self.urban_threshold_km:
            return 'URBAN'
        elif mean_intersite_km <= self.suburban_threshold_km:
            return 'SUBURBAN'
        else:
            return 'RURAL'

    def classify(self) -> pd.DataFrame:
        """
        Classify all cells by environment type.

        Returns:
            DataFrame with columns:
            - cell_name: Cell identifier
            - latitude: Cell latitude
            - longitude: Cell longitude
            - site_key: Site identifier (lat_lon)
            - intersite_distance_km: Mean distance to nearest sites
            - environment: 'URBAN', 'SUBURBAN', or 'RURAL'
        """
        # Build KD-tree for spatial queries
        coords = self.sites_df[['latitude', 'longitude']].values
        tree = cKDTree(coords)

        # Calculate intersite distances for each site
        site_env_map = {}

        for _, site in self.sites_df.iterrows():
            site_coords = [[site['latitude'], site['longitude']]]

            # Query more neighbors than we need to filter out close ones
            k = min(20, len(self.sites_df))
            distances, indices = tree.query(site_coords, k=k)

            # Convert from degrees to km (approximate)
            distances_km = distances[0] * DEGREES_TO_KM

            # Filter to sites >= min distance away (skip index 0 which is self)
            valid_distances = []
            for i in range(1, len(distances_km)):
                if distances_km[i] >= self.min_site_distance_km:
                    valid_distances.append(distances_km[i])
                    if len(valid_distances) == self.nearest_sites_count:
                        break

            # Calculate mean intersite distance
            if len(valid_distances) >= self.nearest_sites_count:
                mean_intersite_km = np.mean(valid_distances)
            elif len(valid_distances) >= 1:
                mean_intersite_km = np.mean(valid_distances)
            else:
                # Isolated site - default to suburban
                mean_intersite_km = 2.0

            environment = self._classify_environment(mean_intersite_km)

            site_env_map[site['site_key']] = {
                'intersite_distance_km': round(mean_intersite_km, 3),
                'environment': environment,
            }

        # Apply site classification to all cells
        df = self.gis_df.copy()
        df['site_key'] = (
            df['latitude'].round(6).astype(str) + '_' +
            df['longitude'].round(6).astype(str)
        )

        # Map site classification to cells
        df['intersite_distance_km'] = df['site_key'].map(
            lambda x: site_env_map.get(x, {}).get('intersite_distance_km', 2.0)
        )
        df['environment'] = df['site_key'].map(
            lambda x: site_env_map.get(x, {}).get('environment', 'SUBURBAN')
        )

        # Select output columns
        output_cols = ['cell_name', 'latitude', 'longitude', 'site_key',
                       'intersite_distance_km', 'environment']
        result = df[output_cols].copy()

        # Log summary
        env_counts = result['environment'].value_counts()
        logger.info(
            "Cell classification complete",
            total_cells=len(result),
            urban=env_counts.get('URBAN', 0),
            suburban=env_counts.get('SUBURBAN', 0),
            rural=env_counts.get('RURAL', 0),
        )

        return result


def classify_cell_environments(
    gis_df: pd.DataFrame,
    output_path: Optional[str] = None,
    urban_threshold_km: float = DEFAULT_URBAN_THRESHOLD_KM,
    suburban_threshold_km: float = DEFAULT_SUBURBAN_THRESHOLD_KM,
) -> pd.DataFrame:
    """
    Convenience function to classify cell environments.

    Args:
        gis_df: Cell GIS data with cell_name, latitude, longitude
        output_path: Optional path to save CSV output
        urban_threshold_km: Max intersite distance for urban (default 1.0km)
        suburban_threshold_km: Max intersite distance for suburban (default 3.0km)

    Returns:
        DataFrame with cell environment classifications

    Example:
        >>> gis_df = pd.read_csv('cork-gis.csv')
        >>> env_df = classify_cell_environments(
        ...     gis_df,
        ...     output_path='data/vf-ie/output-data/cell_environment.csv'
        ... )
        >>> print(env_df['environment'].value_counts())
    """
    classifier = EnvironmentClassifier(
        gis_df,
        urban_threshold_km=urban_threshold_km,
        suburban_threshold_km=suburban_threshold_km,
    )

    cell_environments = classifier.classify()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cell_environments.to_csv(output_path, index=False)
        logger.info(f"Saved cell environments to {output_path}")

    return cell_environments


def load_or_create_cell_environments(
    gis_df: pd.DataFrame,
    output_path: str = 'data/vf-ie/output-data/cell_environment.csv',
    force_recreate: bool = False,
) -> pd.DataFrame:
    """
    Load existing cell environments or create if not present.

    This is the recommended entry point for algorithms that need
    environment classification.

    Args:
        gis_df: Cell GIS data (used if creating new classification)
        output_path: Path to cell_environment.csv
        force_recreate: If True, recreate even if file exists

    Returns:
        DataFrame with cell environment classifications

    Example:
        >>> gis_df = pd.read_csv('cork-gis.csv')
        >>> env_df = load_or_create_cell_environments(gis_df)
        >>> # Now use env_df with detection algorithms
    """
    output_file = Path(output_path)

    if output_file.exists() and not force_recreate:
        logger.info(f"Loading existing cell environments from {output_path}")
        env_df = pd.read_csv(output_path)
        env_df['cell_name'] = env_df['cell_name'].astype(str)
        return env_df

    logger.info("Creating new cell environment classification")
    return classify_cell_environments(gis_df, output_path=output_path)
