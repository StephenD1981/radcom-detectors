"""
Basic Usage Example for RAN Optimizer

Demonstrates:
- Loading configuration
- Setting up logging
- Using custom exceptions
"""
from pathlib import Path
from ran_optimizer.utils.config import load_config
from ran_optimizer.utils.logging_config import configure_logging, get_logger
from ran_optimizer.utils.exceptions import DataValidationError, ConfigurationError


def main():
    """Main example function."""

    # 1. Configure logging (console output for development)
    configure_logging(log_level="INFO", json_output=False)
    logger = get_logger(__name__)

    logger.info("ran_optimizer_starting", version="0.1.0")

    try:
        # 2. Load configuration
        logger.info("loading_config", config_file="config/operators/dish_denver.yaml")

        config = load_config(Path("config/operators/dish_denver.yaml"))

        logger.info("config_loaded",
                    operator=config.operator,
                    region=config.region,
                    features_enabled=[f for f, v in config.features.items()
                                      if hasattr(v, 'enabled') and v.enabled])

        # 3. Display configuration
        print(f"\n{'='*60}")
        print(f"Operator: {config.operator}")
        print(f"Region: {config.region}")
        print(f"{'='*60}\n")

        print("Data Paths:")
        print(f"  Grid: {config.data.grid}")
        print(f"  GIS: {config.data.gis}")
        print(f"  Output: {config.data.output_base}\n")

        print("Features:")
        for feature_name, feature_config in config.features.items():
            if hasattr(feature_config, 'enabled'):
                status = "✅ ENABLED" if feature_config.enabled else "❌ DISABLED"
                print(f"  {feature_name}: {status}")
        print()

        print("Processing:")
        print(f"  Chunk size: {config.processing.chunk_size:,} rows")
        print(f"  Workers: {config.processing.n_workers}")
        print(f"  Timeout: {config.processing.timeout_minutes} minutes\n")

        # 4. Demonstrate overshooting parameters
        if 'overshooters' in config.features:
            overshoot = config.features['overshooters']
            print("Overshooting Detection Parameters:")
            print(f"  Min cell distance: {overshoot.min_cell_distance:,.0f} meters")
            print(f"  Edge traffic: {overshoot.edge_traffic_percent * 100:.0f}%")
            print(f"  Min overshooting grids: {overshoot.min_overshooting_grids}")
            print()

        # 5. Demonstrate exception handling
        logger.info("simulating_validation_error")

        # Simulate a validation error
        raise DataValidationError(
            "Example validation error",
            invalid_rows=150,
            details={
                'rsrp_out_of_range': 120,
                'missing_coordinates': 30
            }
        )

    except DataValidationError as e:
        logger.error("validation_error_caught",
                     error=str(e),
                     invalid_rows=e.invalid_rows,
                     details=e.details)
        print(f"\n⚠️  Validation Error: {e}")
        print(f"    Invalid rows: {e.invalid_rows}")
        print(f"    Details: {e.details}\n")

    except ConfigurationError as e:
        logger.error("configuration_error", error=str(e), exc_info=True)
        print(f"\n❌ Configuration Error: {e}\n")

    except Exception as e:
        logger.error("unexpected_error", error=str(e), exc_info=True)
        print(f"\n❌ Unexpected Error: {e}\n")

    finally:
        logger.info("ran_optimizer_finished")


if __name__ == "__main__":
    main()
