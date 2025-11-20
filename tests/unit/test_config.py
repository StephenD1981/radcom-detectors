"""
Tests for configuration management.
"""
import pytest
from pathlib import Path
from pydantic import ValidationError
from ran_optimizer.utils.config import (
    load_config,
    OperatorConfig,
    OvershooterParams,
    CrossedFeederParams,
    get_default_config
)


def test_load_dish_denver_config():
    """Test loading DISH Denver configuration."""
    config = load_config(Path("config/operators/dish_denver.yaml"))

    assert config.operator == "DISH"
    assert config.region == "Denver"
    assert isinstance(config.features['overshooters'], OvershooterParams)
    assert config.features['overshooters'].enabled is True
    assert config.features['overshooters'].min_cell_distance == 5000


def test_load_vf_ireland_config():
    """Test loading Vodafone Ireland configuration."""
    config = load_config(Path("config/operators/vf_ireland_cork.yaml"))

    assert config.operator == "Vodafone_Ireland"
    assert config.region == "Cork"
    assert config.features['overshooters'].min_cell_distance == 4000  # Urban tuning


def test_config_file_not_found():
    """Test error handling when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("config/operators/nonexistent.yaml"))


def test_overshooter_params_validation():
    """Test validation of overshooting parameters."""
    # Valid params
    params = OvershooterParams(
        edge_traffic_percent=0.1,
        min_cell_distance=5000
    )
    assert params.edge_traffic_percent == 0.1

    # Invalid: edge_traffic_percent > 1.0
    with pytest.raises(ValidationError):
        OvershooterParams(edge_traffic_percent=1.5)

    # Invalid: negative distance
    with pytest.raises(ValidationError):
        OvershooterParams(min_cell_distance=-1000)


def test_crossed_feeder_params_validation():
    """Test validation of crossed feeder parameters."""
    # Valid params
    params = CrossedFeederParams(
        min_angular_deviation=90,
        top_percent_threshold=0.05
    )
    assert params.min_angular_deviation == 90

    # Invalid: angular deviation > 180
    with pytest.raises(ValidationError):
        CrossedFeederParams(min_angular_deviation=200)


def test_get_default_config():
    """Test default configuration generation."""
    config = get_default_config()

    assert config.operator == "UNKNOWN"
    assert config.region == "UNKNOWN"
    assert 'overshooters' in config.features
    assert isinstance(config.features['overshooters'], OvershooterParams)


def test_config_features_parsed_correctly():
    """Test that feature configs are parsed to correct types."""
    config = load_config(Path("config/operators/dish_denver.yaml"))

    # Check types
    assert isinstance(config.features['overshooters'], OvershooterParams)
    assert isinstance(config.features['crossed_feeders'], CrossedFeederParams)

    # Check values
    assert config.features['overshooters'].enabled is True
    assert config.features['crossed_feeders'].enabled is True
    assert config.features['interference'].enabled is False  # Disabled


def test_processing_params():
    """Test processing configuration."""
    config = load_config(Path("config/operators/dish_denver.yaml"))

    assert config.processing.chunk_size == 100000
    assert config.processing.n_workers == 4
    assert config.processing.timeout_minutes == 60
    assert config.processing.cache_intermediate is True
