"""
Tests for data schemas (GridMeasurement and CellGIS).
"""
import pytest
from pydantic import ValidationError
from ran_optimizer.data.schemas import GridMeasurement, CellGIS


class TestGridMeasurement:
    """Tests for GridMeasurement schema."""

    def test_valid_measurement(self):
        """Test valid grid measurement creation."""
        measurement = GridMeasurement(
            geohash7='9xj648q',
            latitude=39.7392,
            longitude=-104.9903,
            rsrp=-85.5,
            rsrq=-10.2,
            sinr=12.5,
            cell_pci=123,
            cell_id='Denver_Site1_Sector1',
            total_traffic=1500.0
        )

        assert measurement.geohash7 == '9xj648q'
        assert measurement.rsrp == -85.5
        assert measurement.cell_pci == 123

    def test_minimal_measurement(self):
        """Test measurement with only required fields."""
        measurement = GridMeasurement(
            geohash7='9xj648q',
            rsrp=-90.0,
            rsrq=-12.0,
            cell_pci=100,
            cell_id='Cell_100'
        )

        assert measurement.geohash7 == '9xj648q'
        assert measurement.sinr is None  # Optional field

    def test_invalid_geohash_format(self):
        """Test geohash validation."""
        # Too short
        with pytest.raises(ValidationError, match="Invalid geohash format"):
            GridMeasurement(
                geohash7='abc',
                rsrp=-90.0,
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100'
            )

        # Invalid characters
        with pytest.raises(ValidationError, match="Invalid geohash format"):
            GridMeasurement(
                geohash7='9xj64@!',
                rsrp=-90.0,
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100'
            )

    def test_rsrp_out_of_range(self):
        """Test RSRP validation."""
        # Too low
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-150.0,  # Too low
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100'
            )

        # Too high
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-20.0,  # Too high
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100'
            )

    def test_rsrq_out_of_range(self):
        """Test RSRQ validation."""
        # Too low
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-90.0,
                rsrq=-50.0,  # Too low
                cell_pci=100,
                cell_id='Cell_100'
            )

        # Too high
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-90.0,
                rsrq=5.0,  # Too high
                cell_pci=100,
                cell_id='Cell_100'
            )

    def test_pci_validation(self):
        """Test PCI range validation (0-503 for LTE)."""
        # Valid PCI
        measurement = GridMeasurement(
            geohash7='9xj648q',
            rsrp=-90.0,
            rsrq=-12.0,
            cell_pci=503,  # Max valid
            cell_id='Cell_100'
        )
        assert measurement.cell_pci == 503

        # Invalid PCI (too high)
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-90.0,
                rsrq=-12.0,
                cell_pci=504,  # Too high
                cell_id='Cell_100'
            )

    def test_coordinates_validation(self):
        """Test latitude/longitude validation."""
        # Valid coordinates
        measurement = GridMeasurement(
            geohash7='9xj648q',
            latitude=39.7392,
            longitude=-104.9903,
            rsrp=-90.0,
            rsrq=-12.0,
            cell_pci=100,
            cell_id='Cell_100'
        )
        assert measurement.latitude == 39.7392

        # Invalid latitude
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                latitude=95.0,  # Too high
                longitude=-104.9903,
                rsrp=-90.0,
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100'
            )

    def test_bearing_validation(self):
        """Test bearing range validation."""
        # Valid bearing
        measurement = GridMeasurement(
            geohash7='9xj648q',
            rsrp=-90.0,
            rsrq=-12.0,
            cell_pci=100,
            cell_id='Cell_100',
            bearing_deg=359.9
        )
        assert measurement.bearing_deg == 359.9

        # Invalid bearing (>= 360)
        with pytest.raises(ValidationError):
            GridMeasurement(
                geohash7='9xj648q',
                rsrp=-90.0,
                rsrq=-12.0,
                cell_pci=100,
                cell_id='Cell_100',
                bearing_deg=360.0
            )


class TestCellGIS:
    """Tests for CellGIS schema."""

    def test_valid_cell(self):
        """Test valid cell GIS creation."""
        cell = CellGIS(
            cell_id='Denver_Site1_Sector1',
            site_name='Denver_Site1',
            latitude=39.7392,
            longitude=-104.9903,
            height_m=30.0,
            azimuth_deg=45.0,
            mechanical_tilt=3.0,
            electrical_tilt=6.0,
            antenna_model='Kathrein 80010541'
        )

        assert cell.cell_id == 'Denver_Site1_Sector1'
        assert cell.height_m == 30.0
        assert cell.azimuth_deg == 45.0

    def test_minimal_cell(self):
        """Test cell with only required fields."""
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=90.0,
            mechanical_tilt=5.0
        )

        assert cell.cell_id == 'Cell_100'
        assert cell.electrical_tilt == 0.0  # Default value
        assert cell.on_air is True  # Default value

    def test_total_tilt_calculation(self):
        """Test automatic total tilt calculation."""
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=90.0,
            mechanical_tilt=5.0,
            electrical_tilt=3.0
        )

        # Should automatically calculate total_tilt
        assert cell.total_tilt == 8.0

    def test_azimuth_validation(self):
        """Test azimuth validation."""
        # Valid azimuth
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=359.9,
            mechanical_tilt=5.0
        )
        assert cell.azimuth_deg == 359.9

        # Out of range
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=25.0,
                azimuth_deg=375.0,  # >360
                mechanical_tilt=5.0
            )

    def test_height_validation(self):
        """Test antenna height validation."""
        # Valid height
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=50.0,
            azimuth_deg=90.0,
            mechanical_tilt=5.0
        )
        assert cell.height_m == 50.0

        # Negative height
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=-10.0,
                azimuth_deg=90.0,
                mechanical_tilt=5.0
            )

        # Unusually high
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=250.0,  # >200m
                azimuth_deg=90.0,
                mechanical_tilt=5.0
            )

    def test_tilt_range_validation(self):
        """Test mechanical tilt range validation."""
        # Valid tilt
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=90.0,
            mechanical_tilt=15.0
        )
        assert cell.mechanical_tilt == 15.0

        # Out of range
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=25.0,
                azimuth_deg=90.0,
                mechanical_tilt=40.0  # >30
            )

    def test_pci_validation(self):
        """Test PCI range validation."""
        # Valid PCI
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=90.0,
            mechanical_tilt=5.0,
            cell_pci=503
        )
        assert cell.cell_pci == 503

        # Invalid PCI
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=25.0,
                azimuth_deg=90.0,
                mechanical_tilt=5.0,
                cell_pci=600  # >503
            )

    def test_frequency_validation(self):
        """Test frequency range validation."""
        # Valid frequency
        cell = CellGIS(
            cell_id='Cell_100',
            site_name='Site_1',
            latitude=39.7,
            longitude=-104.9,
            height_m=25.0,
            azimuth_deg=90.0,
            mechanical_tilt=5.0,
            frequency_mhz=2600.0
        )
        assert cell.frequency_mhz == 2600.0

        # Invalid frequency (too high)
        with pytest.raises(ValidationError):
            CellGIS(
                cell_id='Cell_100',
                site_name='Site_1',
                latitude=39.7,
                longitude=-104.9,
                height_m=25.0,
                azimuth_deg=90.0,
                mechanical_tilt=5.0,
                frequency_mhz=10000.0  # >6000
            )
