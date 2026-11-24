"""Unit tests for NLPulse logging configuration."""

import json
import logging

from src.core.logging import JSONFormatter, get_logger, setup_logging


def test_json_formatter():
    """Verify JSONFormatter produces valid structured JSON with required fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=42,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test log message"
    assert parsed["line"] == 42
    assert "timestamp" in parsed


def test_setup_logging_text_and_json():
    """Verify setup_logging configures root logger without error."""
    setup_logging(log_level="DEBUG", log_format="json")
    logger = get_logger("test")
    assert logger is not None

    setup_logging(log_level="INFO", log_format="text")
    logger2 = get_logger("test2")
    assert logger2 is not None
