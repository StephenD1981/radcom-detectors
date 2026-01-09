"""
Data source abstraction layer.

Provides a unified interface for loading data from multiple sources:
- CSV files
- PostgreSQL database
- Mixed mode (some tables from CSV, others from Postgres)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Set
import pandas as pd

from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.pipeline_config import PipelineConfig, InputConfig
from ran_optimizer.utils.exceptions import DataLoadError

logger = get_logger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def load_coverage(self) -> pd.DataFrame:
        """Load cell coverage grid data."""
        pass

    @abstractmethod
    def load_gis(self) -> pd.DataFrame:
        """Load cell GIS configuration data."""
        pass

    @abstractmethod
    def load_hulls(self) -> 'gpd.GeoDataFrame':
        """Load cell convex hull data."""
        pass

    @abstractmethod
    def load_impacts(self) -> pd.DataFrame:
        """Load cell impact/relation data."""
        pass

    def get_cell_names(self) -> Set[str]:
        """Get set of valid cell names from GIS data."""
        gis_df = self.load_gis()
        return set(gis_df['cell_name'].unique())


class CSVDataSource(DataSource):
    """Data source implementation for CSV files."""

    # Column mappings for standardization
    COVERAGE_COLUMNS = {
        'grid': 'geohash7',
        'avg_rsrp': 'rsrp',
        'avg_rsrq': 'rsrq',
        'avg_sinr': 'sinr',
        'event_count': 'total_traffic',
        'distance_to_cell': 'distance_m',
        'Band': 'band',
        'Bearing': 'azimuth_deg',
        'TiltE': 'electrical_tilt',
        'TiltM': 'mechanical_tilt',
        'Height': 'height_m',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
    }

    GIS_COLUMNS = {
        'bearing': 'azimuth_deg',
        'tilt_elc': 'electrical_tilt',
        'tilt_mech': 'mechanical_tilt',
        'antenna_height': 'height_m',
        'hbw': 'horizontal_beamwidth',
    }

    IMPACTS_COLUMNS = {
        'cell_impact_name': 'to_cell_name',
        'cell_pci': 'pci',
        'cell_impact_pci': 'to_pci',
        'cell_band': 'band',
        'cell_impact_band': 'to_band',
        'co_site': 'intra_site',
        'co_sectored': 'intra_cell',
        'total_cell_traffic_data': 'weight',
        'relation_impact_data_perc': 'cell_perc_weight',
    }

    def __init__(self, config: InputConfig):
        """
        Initialize CSV data source.

        Args:
            config: Input configuration with CSV paths
        """
        self.config = config.csv
        self._cache: Dict[str, Any] = {}

        logger.info(
            "csv_source_initialized",
            base_path=str(self.config.base_path),
            files=list(self.config.files.keys())
        )

    def _get_path(self, file_key: str) -> Path:
        """Get full path for a file key."""
        return self.config.get_file_path(file_key)

    def _standardize_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Rename columns to standard names."""
        rename_map = {old: new for old, new in mapping.items() if old in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def load_coverage(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load cell coverage grid data from CSV.

        Returns:
            DataFrame with columns: geohash7, cell_name, rsrp, rsrq, sinr,
                                   distance_m, total_traffic, band, etc.
        """
        cache_key = 'coverage'
        if use_cache and cache_key in self._cache:
            logger.info("using_cached_coverage")
            return self._cache[cache_key]

        file_path = self._get_path('coverage')
        logger.info("loading_coverage_csv", path=str(file_path))

        if not file_path.exists():
            raise DataLoadError(f"Coverage file not found: {file_path}")

        df = pd.read_csv(file_path)
        df = self._standardize_columns(df, self.COVERAGE_COLUMNS)

        logger.info("coverage_loaded", rows=len(df), columns=len(df.columns))

        if use_cache:
            self._cache[cache_key] = df

        return df

    def load_gis(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load cell GIS configuration data from CSV.

        Returns:
            DataFrame with columns: cell_name, site, latitude, longitude,
                                   azimuth_deg, electrical_tilt, mechanical_tilt,
                                   height_m, band, pci, etc.
        """
        cache_key = 'gis'
        if use_cache and cache_key in self._cache:
            logger.info("using_cached_gis")
            return self._cache[cache_key]

        file_path = self._get_path('gis')
        logger.info("loading_gis_csv", path=str(file_path))

        if not file_path.exists():
            raise DataLoadError(f"GIS file not found: {file_path}")

        df = pd.read_csv(file_path)
        df = self._standardize_columns(df, self.GIS_COLUMNS)

        logger.info("gis_loaded", rows=len(df), columns=len(df.columns))

        if use_cache:
            self._cache[cache_key] = df

        return df

    def load_hulls(self, use_cache: bool = True) -> 'gpd.GeoDataFrame':
        """
        Load cell convex hull data from CSV.

        Returns:
            GeoDataFrame with columns: cell_name, geometry, area_km2
        """
        import geopandas as gpd
        from shapely import wkt

        cache_key = 'hulls'
        if use_cache and cache_key in self._cache:
            logger.info("using_cached_hulls")
            return self._cache[cache_key]

        file_path = self._get_path('hulls')
        logger.info("loading_hulls_csv", path=str(file_path))

        if not file_path.exists():
            raise DataLoadError(f"Hulls file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Parse WKT geometry
        try:
            geometries = df['geometry'].apply(wkt.loads)
        except Exception as e:
            raise DataLoadError(f"Failed to parse WKT geometry in hulls: {e}")

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

        logger.info("hulls_loaded", rows=len(gdf))

        if use_cache:
            self._cache[cache_key] = gdf

        return gdf

    def load_impacts(self, use_cache: bool = True, filter_to_gis: bool = True) -> pd.DataFrame:
        """
        Load cell impact/relation data from CSV.

        Args:
            use_cache: Use cached data if available
            filter_to_gis: If True, filter to cells present in GIS data

        Returns:
            DataFrame with columns: cell_name, to_cell_name, distance, pci, to_pci,
                                   band, to_band, intra_site, intra_cell, weight,
                                   cell_perc_weight, etc.
        """
        cache_key = 'impacts'
        if use_cache and cache_key in self._cache:
            logger.info("using_cached_impacts")
            df = self._cache[cache_key]
        else:
            file_path = self._get_path('impacts')
            logger.info("loading_impacts_csv", path=str(file_path))

            if not file_path.exists():
                raise DataLoadError(f"Impacts file not found: {file_path}")

            df = pd.read_csv(file_path)
            df = self._standardize_columns(df, self.IMPACTS_COLUMNS)

            logger.info("impacts_loaded", rows=len(df), columns=len(df.columns))

            if use_cache:
                self._cache[cache_key] = df

        # Filter to cells in GIS if requested
        if filter_to_gis:
            valid_cells = self.get_cell_names()
            original_count = len(df)

            df = df[
                df['cell_name'].isin(valid_cells) &
                df['to_cell_name'].isin(valid_cells)
            ].copy()

            logger.info(
                "impacts_filtered_to_gis",
                original=original_count,
                filtered=len(df),
                removed=original_count - len(df)
            )

        return df

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("cache_cleared")


class PostgresDataSource(DataSource):
    """Data source implementation for PostgreSQL database."""

    def __init__(self, config: InputConfig):
        """
        Initialize PostgreSQL data source.

        Args:
            config: Input configuration with Postgres connection details
        """
        self.config = config.postgres
        self._engine = None
        self._cache: Dict[str, Any] = {}

        logger.info(
            "postgres_source_initialized",
            host=self.config.host,
            database=self.config.database,
            tables=list(self.config.tables.keys())
        )

    def _get_engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                connection_string = self.config.get_connection_string()
                self._engine = create_engine(connection_string)
                logger.info("postgres_engine_created")
            except ImportError:
                raise DataLoadError("SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary")
            except Exception as e:
                raise DataLoadError(f"Failed to create database connection: {e}")
        return self._engine

    def _load_table(self, table_key: str, cache_key: str, use_cache: bool = True) -> pd.DataFrame:
        """Load a table from PostgreSQL."""
        if use_cache and cache_key in self._cache:
            logger.info(f"using_cached_{cache_key}")
            return self._cache[cache_key]

        table_name = self.config.tables.get(table_key)
        if not table_name:
            raise DataLoadError(f"Table not configured for: {table_key}")

        logger.info(f"loading_{cache_key}_postgres", table=table_name)

        engine = self._get_engine()
        df = pd.read_sql_table(table_name, engine)

        logger.info(f"{cache_key}_loaded", rows=len(df), columns=len(df.columns))

        if use_cache:
            self._cache[cache_key] = df

        return df

    def load_coverage(self, use_cache: bool = True) -> pd.DataFrame:
        """Load cell coverage grid data from PostgreSQL."""
        return self._load_table('coverage', 'coverage', use_cache)

    def load_gis(self, use_cache: bool = True) -> pd.DataFrame:
        """Load cell GIS configuration data from PostgreSQL."""
        return self._load_table('gis', 'gis', use_cache)

    def load_hulls(self, use_cache: bool = True) -> 'gpd.GeoDataFrame':
        """Load cell convex hull data from PostgreSQL with PostGIS."""
        import geopandas as gpd

        cache_key = 'hulls'
        if use_cache and cache_key in self._cache:
            logger.info("using_cached_hulls")
            return self._cache[cache_key]

        table_name = self.config.tables.get('hulls')
        if not table_name:
            raise DataLoadError("Table not configured for: hulls")

        logger.info("loading_hulls_postgres", table=table_name)

        engine = self._get_engine()

        # Use geopandas to read PostGIS geometry
        gdf = gpd.read_postgis(
            f"SELECT * FROM {table_name}",
            engine,
            geom_col='geometry'
        )

        logger.info("hulls_loaded", rows=len(gdf))

        if use_cache:
            self._cache[cache_key] = gdf

        return gdf

    def load_impacts(self, use_cache: bool = True, filter_to_gis: bool = True) -> pd.DataFrame:
        """Load cell impact/relation data from PostgreSQL."""
        df = self._load_table('impacts', 'impacts', use_cache)

        if filter_to_gis:
            valid_cells = self.get_cell_names()
            original_count = len(df)

            df = df[
                df['cell_name'].isin(valid_cells) &
                df['to_cell_name'].isin(valid_cells)
            ].copy()

            logger.info(
                "impacts_filtered_to_gis",
                original=original_count,
                filtered=len(df),
                removed=original_count - len(df)
            )

        return df

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        logger.info("cache_cleared")


def create_data_source(config: PipelineConfig) -> DataSource:
    """
    Factory function to create appropriate data source based on config.

    Args:
        config: Pipeline configuration

    Returns:
        DataSource implementation (CSV or Postgres)

    Example:
        >>> from ran_optimizer.utils.pipeline_config import load_pipeline_config
        >>> config = load_pipeline_config(Path("config/pipeline_config.json"))
        >>> source = create_data_source(config)
        >>> coverage_df = source.load_coverage()
    """
    source_type = config.inputs.source_type

    if source_type == "csv":
        return CSVDataSource(config.inputs)
    elif source_type == "postgres":
        if not config.inputs.postgres.enabled:
            raise DataLoadError("PostgreSQL source selected but not enabled in config")
        return PostgresDataSource(config.inputs)
    elif source_type == "mixed":
        # For mixed mode, we'd need a more complex implementation
        # For now, default to CSV
        logger.warning("mixed_source_not_implemented", fallback="csv")
        return CSVDataSource(config.inputs)
    else:
        raise DataLoadError(f"Unknown source type: {source_type}")


class DataManager:
    """
    High-level data manager that coordinates data loading across sources.

    Provides convenient methods for common data operations and ensures
    consistent filtering and validation.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize data manager.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.source = create_data_source(config)
        self._valid_cells: Optional[Set[str]] = None

        logger.info(
            "data_manager_initialized",
            source_type=config.inputs.source_type,
            operator=config.operator,
            region=config.region
        )

    @property
    def valid_cells(self) -> Set[str]:
        """Get set of valid cell names (from GIS data)."""
        if self._valid_cells is None:
            self._valid_cells = self.source.get_cell_names()
        return self._valid_cells

    def get_coverage_for_detector(
        self,
        detector_name: str,
        band: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get coverage data prepared for a specific detector.

        Args:
            detector_name: Name of detector (for logging)
            band: Optional band filter (e.g., 'L800', 'L1800')

        Returns:
            DataFrame with coverage data
        """
        df = self.source.load_coverage()

        # Filter by band if specified
        if band and 'band' in df.columns:
            original_count = len(df)
            df = df[df['band'] == band].copy()
            logger.info(
                f"coverage_filtered_for_{detector_name}",
                band=band,
                original=original_count,
                filtered=len(df)
            )

        return df

    def get_impacts_for_detector(
        self,
        detector_name: str,
        filter_na_pci: bool = False,
        band: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get impacts data prepared for a specific detector.

        Args:
            detector_name: Name of detector (for logging)
            filter_na_pci: If True, remove rows where PCI is 'N/A'
            band: Optional band filter

        Returns:
            DataFrame with impacts data
        """
        df = self.source.load_impacts(filter_to_gis=True)

        # Filter out N/A PCI values if requested (for PCI detector)
        if filter_na_pci:
            original_count = len(df)
            df = df[
                (df['pci'].astype(str) != 'N/A') &
                (df['to_pci'].astype(str) != 'N/A')
            ].copy()
            logger.info(
                f"impacts_na_pci_filtered_for_{detector_name}",
                original=original_count,
                filtered=len(df),
                removed=original_count - len(df)
            )

        # Filter by band if specified
        if band and 'band' in df.columns:
            original_count = len(df)
            df = df[df['band'] == band].copy()
            logger.info(
                f"impacts_band_filtered_for_{detector_name}",
                band=band,
                original=original_count,
                filtered=len(df)
            )

        return df

    def get_hulls_with_gis(self) -> 'gpd.GeoDataFrame':
        """
        Get hulls data enriched with GIS attributes.

        Returns:
            GeoDataFrame with hull geometry and GIS attributes
        """
        import geopandas as gpd

        hulls = self.source.load_hulls()
        gis = self.source.load_gis()

        # Merge GIS attributes onto hulls
        merged = hulls.merge(
            gis[['cell_name', 'band', 'pci', 'site']],
            on='cell_name',
            how='left'
        )

        return gpd.GeoDataFrame(merged, geometry='geometry', crs=hulls.crs)
