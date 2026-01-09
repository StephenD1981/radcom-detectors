"""
JSON-based configuration management for overshooting detection parameters.

Supports:
- Loading parameters from JSON files
- Environment-specific parameter overrides (urban/suburban/rural)
- Saving configuration snapshots
- Parameter validation
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_overshooting_config(config_path: str) -> Dict[str, Any]:
    """
    Load overshooting detection configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If required fields are missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info("Loading configuration", config_file=str(config_path))

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ['version', 'default']
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")

    logger.info(
        "Configuration loaded",
        version=config.get('version'),
        description=config.get('description', 'N/A'),
    )

    return config


def save_overshooting_config(
    config: Dict[str, Any],
    output_path: str,
    add_timestamp: bool = False
) -> Path:
    """
    Save overshooting detection configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
        add_timestamp: If True, append timestamp to filename

    Returns:
        Path to saved configuration file
    """
    output_path = Path(output_path)

    # Add timestamp to filename if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"

    # Update last_updated timestamp
    config['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Configuration saved", output_file=str(output_path))

    return output_path


def get_environment_params(
    config: Dict[str, Any],
    environment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific environment, merging with defaults.

    Args:
        config: Full configuration dictionary
        environment: Environment type ('urban', 'suburban', 'rural') or None for default

    Returns:
        Dictionary of parameters with environment-specific overrides applied

    Raises:
        ValueError: If environment is invalid or disabled
    """
    # Start with default parameters
    params = config['default'].copy()

    # Remove non-parameter fields
    params.pop('description', None)

    # If no environment specified, return defaults
    if environment is None or environment.lower() == 'default':
        logger.info("Using default parameters")
        return params

    # Validate environment
    environment = environment.lower()
    valid_environments = ['urban', 'suburban', 'rural']
    if environment not in valid_environments:
        raise ValueError(
            f"Invalid environment '{environment}'. "
            f"Must be one of: {valid_environments}"
        )

    # Check if environment-specific config exists
    if 'environment_specific' not in config:
        logger.warning(
            "No environment-specific config found, using defaults",
            environment=environment,
        )
        return params

    env_config = config['environment_specific'].get(environment, {})

    # Check if environment is enabled
    if not env_config.get('enable', True):
        raise ValueError(f"Environment '{environment}' is disabled in config")

    # Apply environment-specific overrides
    overrides_count = 0
    for key, value in env_config.items():
        if key not in ['description', 'enable']:
            params[key] = value
            overrides_count += 1
            logger.debug(
                f"Overriding parameter for {environment}",
                parameter=key,
                value=value,
            )

    logger.info(
        "Using environment-specific parameters",
        environment=environment,
        overrides_applied=overrides_count,
    )

    return params


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and parameter values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check version
    if 'version' not in config:
        raise ValueError("Configuration missing 'version' field")

    # Check default parameters exist
    if 'default' not in config:
        raise ValueError("Configuration missing 'default' section")

    default_params = config['default']

    # Validate required parameters
    required_params = [
        'edge_traffic_percent',
        'min_cell_distance',
        'min_cell_count_in_grid',
        'max_percentage_grid_events',
        'min_relative_reach',
        'min_overshooting_grids',
        'percentage_overshooting_grids',
    ]

    missing_params = [p for p in required_params if p not in default_params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    # Validate parameter ranges
    validations = [
        ('edge_traffic_percent', 0.0, 1.0),
        ('max_percentage_grid_events', 0.0, 1.0),
        ('min_relative_reach', 0.0, 1.0),
        ('percentage_overshooting_grids', 0.0, 1.0),
        ('min_cell_distance', 0, 50000),
        ('min_cell_count_in_grid', 1, 20),
        ('min_overshooting_grids', 0, 10000),
    ]

    for param, min_val, max_val in validations:
        value = default_params.get(param)
        if value is not None and not (min_val <= value <= max_val):
            raise ValueError(
                f"Parameter '{param}' value {value} outside valid range "
                f"[{min_val}, {max_val}]"
            )

    logger.info("Configuration validation passed")
    return True


def get_default_config_path() -> Path:
    """
    Get path to default configuration file.

    Returns:
        Path to config/overshooting_params.json
    """
    # Assume we're in ran_optimizer/utils/overshooting_config.py
    # Navigate up to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    config_path = project_root / "config" / "overshooting_params.json"

    return config_path
