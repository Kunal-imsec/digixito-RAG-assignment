"""
Observability: request IDs, operation timing, and in-memory metrics.

Provides:
- Request ID generation and context-variable storage
- @timed decorator / context manager for measuring operation latency
- In-memory counters for requests, uploads, queries, and errors

No external infrastructure required — metrics are exposed via /health.
"""

import contextvars
import functools
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request ID context variable
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(request_id: str | None = None) -> str:
    """Set the request ID for the current context. Generates one if not provided."""
    rid = request_id or uuid.uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return _request_id_var.get()


# ---------------------------------------------------------------------------
# In-memory metrics store
# ---------------------------------------------------------------------------

@dataclass
class MetricsStore:
    """Simple in-memory counters and timing data for observability."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_uploads: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Operation latency records (keep last N for each operation)
    _latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _max_latency_records: int = 100

    def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record an operation's latency in milliseconds."""
        records = self._latencies[operation]
        records.append(duration_ms)
        # Keep only the most recent records
        if len(records) > self._max_latency_records:
            self._latencies[operation] = records[-self._max_latency_records:]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for the /health endpoint."""
        latency_summary = {}
        for op, durations in self._latencies.items():
            if durations:
                latency_summary[op] = {
                    "count": len(durations),
                    "avg_ms": round(sum(durations) / len(durations), 2),
                    "max_ms": round(max(durations), 2),
                    "min_ms": round(min(durations), 2),
                }

        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
            },
            "uploads": {
                "total": self.total_uploads,
                "successful": self.successful_uploads,
                "failed": self.failed_uploads,
            },
            "queries": {
                "total": self.total_queries,
                "successful": self.successful_queries,
                "failed": self.failed_queries,
            },
            "latencies": latency_summary,
        }


# Singleton instance
metrics = MetricsStore()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class timed:
    """Context manager and decorator that logs and records operation duration.

    Usage as context manager:
        with timed("embedding"):
            result = embed(texts)

    Usage as decorator:
        @timed("pdf_extraction")
        def extract(path): ...
    """

    def __init__(self, operation: str):
        self.operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        logger.info("operation=%s status=started", self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000
        metrics.record_latency(self.operation, duration_ms)

        if exc_type is not None:
            logger.error(
                "operation=%s status=failed duration_ms=%.2f error=%s",
                self.operation,
                duration_ms,
                str(exc_val),
            )
        else:
            logger.info(
                "operation=%s status=completed duration_ms=%.2f",
                self.operation,
                duration_ms,
            )
        return None  # Do not suppress exceptions

    def __call__(self, func):
        """Use as a decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed(self.operation):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with timed(self.operation):
                return await func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
