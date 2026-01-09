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
    # Clear existing handlers to allow reconfiguration (important for tests)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True  # Force reconfiguration even if already configured
    )

    # When using file logging, we need to route through stdlib logging
    # to ensure file handlers receive the messages
    if log_file:
        # Processors that work with stdlib logging
        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # ProcessorFormatter will handle rendering
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]

        # Setup file handler
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        # Use ProcessorFormatter for the file handler
        if json_output:
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        else:
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(),
            )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

        # Also keep console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

    else:
        # No file logging - use direct rendering
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


def flush_logs():
    """
    Flush all log handlers to ensure buffered content is written.

    Useful in tests or before application shutdown to ensure
    all log messages are persisted to disk.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
