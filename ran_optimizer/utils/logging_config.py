"""
Structured logging configuration using structlog.

Provides JSON-formatted logs suitable for production monitoring
and human-readable console output for development.
"""
import sys
import logging
import structlog
from pathlib import Path
from typing import Optional


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_output: bool = False
):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_output: If True, output JSON logs; else human-readable console

    Example:
        >>> from ran_optimizer.utils.logging_config import configure_logging, get_logger
        >>> configure_logging(log_level="INFO", json_output=False)
        >>> logger = get_logger(__name__)
        >>> logger.info("processing_started", operator="DISH", region="Denver")
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Processors for structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add JSON or console renderer
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        # JSON format for file logs (easier to parse)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)

        logging.getLogger().addHandler(file_handler)


def get_logger(name: str):
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger with bound context

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("grid_data_loaded", rows=881498, cells=3045)
        >>> logger.error("validation_failed", error_count=150, exc_info=True)
    """
    return structlog.get_logger(name)
