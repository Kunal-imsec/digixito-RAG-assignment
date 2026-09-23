"""
Structured logging configuration.

Sets up Python logging with a JSON-like structured format that includes
timestamps, log level, logger name, message, and optional request_id
from the observability context variable.

IMPORTANT: Never log API keys, credentials, or other secrets.
"""

import logging
import sys
from datetime import datetime, timezone

from app.observability.metrics import get_request_id


class StructuredFormatter(logging.Formatter):
    """Custom formatter that produces structured log lines with request context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        request_id = get_request_id() or "-"

        # Build structured log line
        log_line = (
            f'{{"timestamp": "{timestamp}", '
            f'"level": "{record.levelname}", '
            f'"logger": "{record.name}", '
            f'"request_id": "{request_id}", '
            f'"message": "{self._escape(record.getMessage())}"'
        )

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            exc_text = self.formatException(record.exc_info)
            log_line += f', "exception": "{self._escape(exc_text)}"'

        log_line += "}"
        return log_line

    @staticmethod
    def _escape(text: str) -> str:
        """Escape characters that would break JSON string."""
        return (
            text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )


def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide structured logging.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    # Console handler with structured formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging configured at level=%s", log_level)
