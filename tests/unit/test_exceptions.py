"""
Tests for custom exceptions.
"""
import pytest
from ran_optimizer.utils.exceptions import (
    RANOptimizerError,
    ConfigurationError,
    DataValidationError,
    DataLoadError,
    ProcessingError,
    AlgorithmError,
    GeometryError
)


def test_base_exception():
    """Test base exception."""
    with pytest.raises(RANOptimizerError):
        raise RANOptimizerError("Base error")


def test_configuration_error():
    """Test configuration error."""
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid config")

    # Should also be catchable as base class
    with pytest.raises(RANOptimizerError):
        raise ConfigurationError("Invalid config")


def test_data_validation_error():
    """Test data validation error with details."""
    error = DataValidationError(
        "Validation failed",
        invalid_rows=150,
        details={'min_rsrp': -200, 'max_rsrp': -40}
    )

    assert error.invalid_rows == 150
    assert error.details['min_rsrp'] == -200
    assert "invalid_rows=150" in str(error)


def test_data_load_error():
    """Test data load error."""
    with pytest.raises(DataLoadError):
        raise DataLoadError("File not found")


def test_processing_error_with_stage():
    """Test processing error with stage information."""
    error = ProcessingError(
        "Processing failed",
        stage="enrichment",
        details={'rows_processed': 1000}
    )

    assert error.stage == "enrichment"
    assert error.details['rows_processed'] == 1000
    assert "stage=enrichment" in str(error)


def test_algorithm_error():
    """Test algorithm error."""
    with pytest.raises(AlgorithmError):
        raise AlgorithmError("Algorithm failed")


def test_geometry_error():
    """Test geometry error."""
    with pytest.raises(GeometryError):
        raise GeometryError("Invalid coordinates")


def test_exception_inheritance():
    """Test that all custom exceptions inherit from base."""
    exceptions = [
        ConfigurationError,
        DataValidationError,
        DataLoadError,
        ProcessingError,
        AlgorithmError,
        GeometryError
    ]

    for exc_class in exceptions:
        assert issubclass(exc_class, RANOptimizerError)
