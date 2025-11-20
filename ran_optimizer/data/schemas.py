"""
Pydantic schemas for data validation.

Defines data models for grid measurements and cell GIS data with
validation rules for RF metrics, coordinates, and antenna parameters.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class GridMeasurement(BaseModel):
    """
    Schema for grid measurement data (drive test bins).

    Validates RF metrics, geohash format, and traffic data from
    enriched grid CSV files.

    Example:
        >>> measurement = GridMeasurement(
        ...     geohash7='9xj648q',
        ...     rsrp=-85.5,
        ...     rsrq=-10.2,
        ...     sinr=12.5,
        ...     cell_pci=123,
        ...     cell_id='Denver_Site1_Sector1',
        ...     total_traffic=1500.0
        ... )
    """
    # Geospatial identifiers
    geohash7: str = Field(..., description="7-character geohash of measurement location")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude in decimal degrees")

    # RF Measurements
    rsrp: float = Field(..., ge=-140, le=-30, description="Reference Signal Received Power (dBm)")
    rsrq: float = Field(..., ge=-40, le=0, description="Reference Signal Received Quality (dB)")
    sinr: Optional[float] = Field(None, ge=-20, le=40, description="Signal-to-Interference-plus-Noise Ratio (dB)")

    # Cell identifiers
    cell_pci: int = Field(..., ge=0, le=503, description="Physical Cell ID (LTE: 0-503)")
    cell_id: str = Field(..., min_length=1, description="Unique cell identifier")
    enodeb_id: Optional[int] = Field(None, ge=0, description="eNodeB identifier")
    sector_id: Optional[str] = Field(None, description="Sector identifier")

    # Traffic data
    total_traffic: Optional[float] = Field(None, ge=0, description="Total traffic volume (MB or sessions)")
    num_samples: Optional[int] = Field(None, ge=1, description="Number of samples in this bin")

    # Distance/geometry (enriched fields)
    distance_m: Optional[float] = Field(None, ge=0, description="Distance from serving cell (meters)")
    bearing_deg: Optional[float] = Field(None, ge=0, lt=360, description="Bearing from cell to bin (degrees)")

    @field_validator('geohash7')
    @classmethod
    def validate_geohash(cls, v: str) -> str:
        """Validate geohash format (7 characters, base32)."""
        if not re.match(r'^[0-9a-z]{7}$', v.lower()):
            raise ValueError(f"Invalid geohash format: {v} (must be 7 base32 characters)")
        return v.lower()

    # Note: RSRP and RSRQ range validation is handled by Field constraints (ge, le)

    model_config = {
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }


class CellGIS(BaseModel):
    """
    Schema for cell GIS (site/antenna) data.

    Validates cell location, antenna parameters, and configuration
    from GIS CSV files.

    Example:
        >>> cell = CellGIS(
        ...     cell_id='Denver_Site1_Sector1',
        ...     site_name='Denver_Site1',
        ...     latitude=39.7392,
        ...     longitude=-104.9903,
        ...     height_m=30.0,
        ...     azimuth_deg=45.0,
        ...     mechanical_tilt=3.0,
        ...     electrical_tilt=6.0,
        ...     antenna_model='Kathrein 80010541'
        ... )
    """
    # Cell identifiers
    cell_id: str = Field(..., min_length=1, description="Unique cell identifier (must match grid data)")
    site_name: str = Field(..., min_length=1, description="Site name")
    enodeb_id: Optional[int] = Field(None, ge=0, description="eNodeB identifier")
    sector_id: Optional[str] = Field(None, description="Sector identifier (e.g., '1', '2', '3')")
    cell_pci: Optional[int] = Field(None, ge=0, le=503, description="Physical Cell ID")

    # Location
    latitude: float = Field(..., ge=-90, le=90, description="Cell latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Cell longitude in decimal degrees")
    height_m: float = Field(..., ge=0, le=200, description="Antenna height above ground (meters)")

    # Antenna parameters
    azimuth_deg: float = Field(..., ge=0, le=360, description="Antenna azimuth/bearing (degrees, 0=North)")
    mechanical_tilt: float = Field(..., ge=-30, le=30, description="Mechanical downtilt (degrees)")
    electrical_tilt: float = Field(0.0, ge=-30, le=30, description="Electrical downtilt (degrees)")
    total_tilt: Optional[float] = Field(None, description="Total downtilt (mechanical + electrical)")

    # Antenna model and specs
    antenna_model: Optional[str] = Field(None, description="Antenna model name")
    beamwidth_horizontal: Optional[float] = Field(None, ge=0, le=360, description="Horizontal beamwidth (degrees)")
    beamwidth_vertical: Optional[float] = Field(None, ge=0, le=90, description="Vertical beamwidth (degrees)")
    antenna_gain: Optional[float] = Field(None, ge=0, le=30, description="Antenna gain (dBi)")

    # Technology and configuration
    technology: Optional[str] = Field(None, description="Technology (e.g., 'LTE', '5G NR')")
    frequency_mhz: Optional[float] = Field(None, ge=400, le=6000, description="Operating frequency (MHz)")
    bandwidth_mhz: Optional[float] = Field(None, ge=0, le=100, description="Channel bandwidth (MHz)")
    tx_power_dbm: Optional[float] = Field(None, ge=0, le=60, description="Transmit power (dBm)")

    # Status
    on_air: bool = Field(True, description="Whether cell is active/on-air")

    @model_validator(mode='after')
    def calculate_total_tilt(self):
        """Calculate total tilt if not provided."""
        if self.total_tilt is None:
            self.total_tilt = self.mechanical_tilt + self.electrical_tilt
        return self

    # Note: Height and azimuth validation is handled by Field constraints (ge, le)

    model_config = {
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }
