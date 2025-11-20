"""
Tests for data loaders (load_grid_data and load_gis_data).
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
from ran_optimizer.data.loaders import (
    load_grid_data,
    load_gis_data,
    get_data_summary,
    _validate_dataframe
)
from ran_optimizer.data.schemas import GridMeasurement, CellGIS
from ran_optimizer.utils.exceptions import DataLoadError, DataValidationError


@pytest.fixture
def valid_grid_csv():
    """Create temporary CSV with valid grid data."""
    data = """geohash7,latitude,longitude,rsrp,rsrq,sinr,cell_pci,cell_id,total_traffic,distance_m
9xj648q,39.7392,-104.9903,-85.5,-10.2,12.5,123,Cell_1,1500.0,2000.0
9xj649r,39.7393,-104.9904,-88.0,-11.0,10.0,124,Cell_2,1200.0,2500.0
9xj650s,39.7394,-104.9905,-90.5,-12.5,8.5,125,Cell_3,900.0,3000.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()  # Clean up


@pytest.fixture
def invalid_grid_csv():
    """Create temporary CSV with invalid grid data."""
    data = """geohash7,latitude,longitude,rsrp,rsrq,sinr,cell_pci,cell_id,total_traffic
9xj648q,39.7392,-104.9903,-85.5,-10.2,12.5,123,Cell_1,1500.0
abc,39.7393,-104.9904,-88.0,-11.0,10.0,124,Cell_2,1200.0
9xj650s,39.7394,-104.9905,-200.0,-12.5,8.5,125,Cell_3,900.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


@pytest.fixture
def valid_gis_csv():
    """Create temporary CSV with valid GIS data."""
    data = """cell_id,site_name,latitude,longitude,height_m,azimuth_deg,mechanical_tilt,electrical_tilt,on_air
Cell_1,Site_1,39.7392,-104.9903,30.0,45.0,3.0,6.0,True
Cell_2,Site_1,39.7392,-104.9903,30.0,135.0,3.0,6.0,True
Cell_3,Site_2,39.7400,-104.9910,25.0,90.0,5.0,4.0,True
Cell_4,Site_3,39.7410,-104.9920,28.0,180.0,4.0,5.0,False
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


@pytest.fixture
def invalid_gis_csv():
    """Create temporary CSV with invalid GIS data."""
    data = """cell_id,site_name,latitude,longitude,height_m,azimuth_deg,mechanical_tilt,electrical_tilt
Cell_1,Site_1,39.7392,-104.9903,30.0,45.0,3.0,6.0
Cell_2,Site_1,95.0,-104.9903,30.0,135.0,3.0,6.0
Cell_3,Site_2,39.7400,-104.9910,-10.0,90.0,5.0,4.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


class TestLoadGridData:
    """Tests for load_grid_data function."""

    def test_load_valid_grid_data(self, valid_grid_csv):
        """Test loading valid grid data."""
        df = load_grid_data(valid_grid_csv, validate=True)

        assert len(df) == 3
        assert 'geohash7' in df.columns
        assert 'rsrp' in df.columns
        assert df['cell_id'].tolist() == ['Cell_1', 'Cell_2', 'Cell_3']

    def test_load_without_validation(self, valid_grid_csv):
        """Test loading without validation."""
        df = load_grid_data(valid_grid_csv, validate=False)

        assert len(df) == 3
        # Should load all data without validation

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(DataLoadError, match="Grid file not found"):
            load_grid_data(Path("nonexistent.csv"))

    def test_sample_rows(self, valid_grid_csv):
        """Test loading only first N rows."""
        df = load_grid_data(valid_grid_csv, validate=False, sample_rows=2)

        assert len(df) == 2
        assert df['cell_id'].tolist() == ['Cell_1', 'Cell_2']

    def test_invalid_data_validation(self, invalid_grid_csv):
        """Test validation errors with invalid data."""
        # Should fail because 2/3 (66%) rows are invalid (exceeds 10% threshold)
        with pytest.raises(DataValidationError, match="Grid data validation failed"):
            load_grid_data(invalid_grid_csv, validate=True)

    def test_validation_failure_threshold(self):
        """Test validation failure when >10% rows invalid."""
        # Create CSV with >10% invalid rows (4 out of 5 = 80%)
        data = """geohash7,rsrp,rsrq,cell_pci,cell_id
9xj648q,-85.5,-10.2,123,Cell_1
abc,-88.0,-11.0,124,Cell_2
def,-90.5,-12.5,125,Cell_3
ghi,-92.0,-13.0,126,Cell_4
jkl,-93.5,-14.0,127,Cell_5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            with pytest.raises(DataValidationError, match="Grid data validation failed"):
                load_grid_data(temp_path, validate=True)
        finally:
            temp_path.unlink()


class TestLoadGISData:
    """Tests for load_gis_data function."""

    def test_load_valid_gis_data(self, valid_gis_csv):
        """Test loading valid GIS data."""
        df = load_gis_data(valid_gis_csv, validate=True)

        # Should have 3 cells (Cell_4 is off-air and filtered)
        assert len(df) == 3
        assert 'cell_id' in df.columns
        assert 'azimuth_deg' in df.columns

    def test_load_without_on_air_filter(self, valid_gis_csv):
        """Test loading without filtering on-air cells."""
        df = load_gis_data(valid_gis_csv, validate=True, filter_on_air=False)

        # Should have all 4 cells
        assert len(df) == 4

    def test_load_without_validation(self, valid_gis_csv):
        """Test loading without validation."""
        df = load_gis_data(valid_gis_csv, validate=False)

        assert len(df) == 3  # Still filters on_air by default

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(DataLoadError, match="GIS file not found"):
            load_gis_data(Path("nonexistent.csv"))

    def test_invalid_data_validation(self, invalid_gis_csv):
        """Test validation with invalid data."""
        # Should fail because 2/3 (66%) rows are invalid (exceeds 5% threshold)
        with pytest.raises(DataValidationError, match="GIS data validation failed"):
            load_gis_data(invalid_gis_csv, validate=True)

    def test_validation_failure_threshold(self):
        """Test validation failure when >5% rows invalid (stricter for GIS)."""
        # Create CSV with >5% invalid rows
        data = """cell_id,site_name,latitude,longitude,height_m,azimuth_deg,mechanical_tilt
Cell_1,Site_1,39.7392,-104.9903,30.0,45.0,3.0
Cell_2,Site_1,95.0,-104.9903,30.0,135.0,3.0
Cell_3,Site_2,-95.0,-104.9910,25.0,90.0,5.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            with pytest.raises(DataValidationError, match="GIS data validation failed"):
                load_gis_data(temp_path, validate=True)
        finally:
            temp_path.unlink()


class TestValidateDataFrame:
    """Tests for _validate_dataframe helper function."""

    def test_validate_all_valid_rows(self):
        """Test validation with all valid rows."""
        df = pd.DataFrame([
            {'geohash7': '9xj648q', 'rsrp': -85.5, 'rsrq': -10.2, 'cell_pci': 123, 'cell_id': 'Cell_1'},
            {'geohash7': '9xj649r', 'rsrp': -88.0, 'rsrq': -11.0, 'cell_pci': 124, 'cell_id': 'Cell_2'},
        ])

        validated_df, errors = _validate_dataframe(df, GridMeasurement)

        assert len(validated_df) == 2
        assert len(errors) == 0

    def test_validate_with_invalid_rows(self):
        """Test validation with some invalid rows."""
        df = pd.DataFrame([
            {'geohash7': '9xj648q', 'rsrp': -85.5, 'rsrq': -10.2, 'cell_pci': 123, 'cell_id': 'Cell_1'},
            {'geohash7': 'abc', 'rsrp': -88.0, 'rsrq': -11.0, 'cell_pci': 124, 'cell_id': 'Cell_2'},
        ])

        validated_df, errors = _validate_dataframe(df, GridMeasurement)

        assert len(validated_df) == 1
        assert len(errors) == 1
        assert errors[0]['row_index'] == 1


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_grid_summary(self, valid_grid_csv):
        """Test summary statistics for grid data."""
        df = load_grid_data(valid_grid_csv, validate=False)
        summary = get_data_summary(df, data_type="grid")

        assert summary['total_rows'] == 3
        assert summary['unique_cells'] == 3
        assert summary['rsrp_min'] == -90.5
        assert summary['rsrp_max'] == -85.5
        assert summary['rsrp_mean'] == pytest.approx(-88.0)

    def test_gis_summary(self, valid_gis_csv):
        """Test summary statistics for GIS data."""
        df = load_gis_data(valid_gis_csv, validate=False)
        summary = get_data_summary(df, data_type="gis")

        assert summary['total_rows'] == 3
        assert summary['unique_sites'] == 2  # Site_1 and Site_2 (Site_3 off-air)
        assert summary['unique_cells'] == 3
        assert summary['avg_height'] == pytest.approx(28.33, abs=0.1)
        assert summary['on_air_cells'] == 3
