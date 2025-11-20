"""
Tests for logging configuration.
"""
import pytest
from pathlib import Path
import tempfile
from ran_optimizer.utils.logging_config import configure_logging, get_logger


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger(__name__)
    assert logger is not None


def test_configure_logging_console():
    """Test console logging configuration."""
    configure_logging(log_level="INFO", json_output=False)
    logger = get_logger(__name__)

    # Should not raise exception
    logger.info("test_message", key="value")
    logger.debug("debug_message")  # May not print (INFO level)


def test_configure_logging_json():
    """Test JSON logging configuration."""
    configure_logging(log_level="DEBUG", json_output=True)
    logger = get_logger(__name__)

    # Should not raise exception
    logger.info("test_json", operator="DISH", count=100)


def test_configure_logging_with_file():
    """Test logging to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        configure_logging(
            log_level="INFO",
            log_file=log_file,
            json_output=False
        )

        logger = get_logger(__name__)
        logger.info("test_file_logging", message="hello")

        # Verify file was created
        assert log_file.exists()

        # Verify content
        content = log_file.read_text()
        assert "test_file_logging" in content


def test_logging_with_exception():
    """Test logging with exception traceback."""
    configure_logging(log_level="ERROR", json_output=False)
    logger = get_logger(__name__)

    try:
        raise ValueError("Test error")
    except ValueError:
        # Should not raise exception
        logger.error("exception_occurred", exc_info=True)
