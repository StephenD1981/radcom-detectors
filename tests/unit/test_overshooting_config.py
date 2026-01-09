"""
Unit tests for JSON-based overshooting configuration system.
"""
import pytest
from pathlib import Path

from ran_optimizer.recommendations import OvershooterParams
from ran_optimizer.utils.overshooting_config import (
    load_overshooting_config,
    save_overshooting_config,
    get_environment_params,
    validate_config,
    get_default_config_path,
)


def test_load_default_config():
    """Test loading default configuration file."""
    config_path = get_default_config_path()
    config = load_overshooting_config(str(config_path))

    assert 'version' in config
    assert 'default' in config
    assert 'environment_specific' in config
    print(f"✅ Default config loaded: version {config['version']}")


def test_validate_config():
    """Test configuration validation."""
    config_path = get_default_config_path()
    config = load_overshooting_config(str(config_path))

    # Should pass validation
    assert validate_config(config) is True
    print("✅ Config validation passed")


def test_get_environment_params():
    """Test retrieving environment-specific parameters."""
    config_path = get_default_config_path()
    config = load_overshooting_config(str(config_path))

    # Test default params
    default_params = get_environment_params(config, None)
    assert 'edge_traffic_percent' in default_params
    assert 'min_cell_distance' in default_params
    print(f"✅ Default params: {len(default_params)} parameters")

    # Test urban params (should override some defaults)
    urban_params = get_environment_params(config, 'urban')
    assert urban_params['min_cell_distance'] == 2000  # Urban override
    assert urban_params['min_cell_count_in_grid'] == 5  # Urban override
    print(f"✅ Urban params: min_cell_distance={urban_params['min_cell_distance']}m")

    # Test rural params
    rural_params = get_environment_params(config, 'rural')
    assert rural_params['min_cell_distance'] == 6000  # Rural override
    assert rural_params['min_cell_count_in_grid'] == 3  # Rural override
    print(f"✅ Rural params: min_cell_distance={rural_params['min_cell_distance']}m")


def test_overshooter_params_from_config_default():
    """Test creating OvershooterParams from config (default)."""
    params = OvershooterParams.from_config()

    # Check default values
    assert params.edge_traffic_percent == 0.15
    assert params.min_cell_distance == 4000
    assert params.min_cell_count_in_grid == 4
    assert params.min_relative_reach == 0.70
    print(f"✅ Default params loaded: min_cell_distance={params.min_cell_distance}m")


def test_overshooter_params_from_config_urban():
    """Test creating OvershooterParams from config (urban)."""
    params = OvershooterParams.from_config(environment='urban')

    # Check urban overrides
    assert params.min_cell_distance == 2000  # Urban override
    assert params.min_cell_count_in_grid == 5  # Urban override
    assert params.edge_traffic_percent == 0.10  # Urban override
    print(f"✅ Urban params loaded: min_cell_distance={params.min_cell_distance}m, min_cell_count={params.min_cell_count_in_grid}")


def test_overshooter_params_from_config_rural():
    """Test creating OvershooterParams from config (rural)."""
    params = OvershooterParams.from_config(environment='rural')

    # Check rural overrides
    assert params.min_cell_distance == 6000  # Rural override
    assert params.min_cell_count_in_grid == 3  # Rural override
    assert params.edge_traffic_percent == 0.20  # Rural override
    assert params.min_relative_reach == 0.65  # Rural override
    assert params.min_overshooting_grids == 20  # Rural override
    print(f"✅ Rural params loaded: min_cell_distance={params.min_cell_distance}m, min_cell_count={params.min_cell_count_in_grid}")


def test_param_consistency():
    """Test that hardcoded defaults match JSON config defaults."""
    # Create with hardcoded defaults
    hardcoded = OvershooterParams()

    # Create from JSON config
    from_config = OvershooterParams.from_config()

    # Compare
    assert hardcoded.edge_traffic_percent == from_config.edge_traffic_percent
    assert hardcoded.min_cell_distance == from_config.min_cell_distance
    assert hardcoded.min_cell_count_in_grid == from_config.min_cell_count_in_grid
    assert hardcoded.max_percentage_grid_events == from_config.max_percentage_grid_events
    assert hardcoded.min_relative_reach == from_config.min_relative_reach
    assert hardcoded.min_overshooting_grids == from_config.min_overshooting_grids
    assert hardcoded.percentage_overshooting_grids == from_config.percentage_overshooting_grids

    print("✅ Hardcoded defaults match JSON config defaults")


if __name__ == "__main__":
    print("🧪 Testing JSON Configuration System")
    print("=" * 80)

    test_load_default_config()
    test_validate_config()
    test_get_environment_params()
    test_overshooter_params_from_config_default()
    test_overshooter_params_from_config_urban()
    test_overshooter_params_from_config_rural()
    test_param_consistency()

    print("=" * 80)
    print("✅ ALL TESTS PASSED")
