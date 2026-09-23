"""
FastAPI application entry point.

Sets up:
- Application instance with metadata
- CORS middleware for local development
- Observability middleware (request IDs, timing, counting)
- API router inclusion
- Startup logging configuration
"""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.logging_config import setup_logging
from app.observability.metrics import metrics, set_request_id

# ---------------------------------------------------------------------------
# Initialize logging first
# ---------------------------------------------------------------------------
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Document Intelligence System",
    description=(
        "RAG-based document question-answering system with PDF ingestion, "
        "semantic retrieval, Gemini-powered answers, streaming, "
        "conversation history, and observability."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS middleware (for local development / browser clients)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Observability middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Assign request ID, log request/response, track metrics."""
    # Generate and set request ID
    request_id = set_request_id()

    # Track request
    metrics.total_requests += 1

    start_time = time.perf_counter()

    logger.info(
        "request_start method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )

    try:
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_latency("http_request", duration_ms)

        # Set request ID in response header
        response.headers["X-Request-ID"] = request_id

        status_code = response.status_code
        if status_code < 400:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        logger.info(
            "request_end method=%s path=%s status=%d duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            request_id,
        )

        return response

    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics.failed_requests += 1

        logger.error(
            "request_error method=%s path=%s error=%s duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            str(exc),
            duration_ms,
            request_id,
        )

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
            headers={"X-Request-ID": request_id},
        )


# ---------------------------------------------------------------------------
# Include API routes
# ---------------------------------------------------------------------------
app.include_router(router)

# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Log application startup information."""
    logger.info("=" * 60)
    logger.info("AI Document Intelligence System starting up")
    logger.info("Gemini model: %s", settings.GEMINI_MODEL)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)
    logger.info("Chunk size: %d, Overlap: %d", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info("Top-K: %d", settings.TOP_K)
    logger.info("ChromaDB path: %s", settings.CHROMA_PERSIST_DIR)
    logger.info("Log level: %s", settings.LOG_LEVEL)
    logger.info("=" * 60)
