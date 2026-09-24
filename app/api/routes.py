"""
FastAPI API routes.

Endpoints:
- POST /upload  — Upload and process a PDF document
- POST /ask     — Ask a question with streaming response (SSE)
- GET  /health  — Application health and metrics
"""

import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.config import settings
from app.conversation.history import ConversationHistory, ConversationError
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pdf_processor import PDFProcessor, PDFProcessingError
from app.llm.groq import GroqClient, GroqError
from app.llm.prompts import build_grounded_prompt
from app.observability.metrics import get_request_id, metrics, timed
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Service singletons (initialized once, reused across requests)
# ---------------------------------------------------------------------------

_pdf_processor = PDFProcessor()
_embedding_service = EmbeddingService()
_chroma_store = ChromaStore()
_retriever = Retriever(_embedding_service, _chroma_store)
_groq_client = GroqClient()
_conversation_history = ConversationHistory()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateConversationResponse(BaseModel):
    """Response body for the /conversations endpoint."""
    conversation_id: str


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str
    conversation_id: str


class UploadRequest(BaseModel):
    """Request body for the /upload endpoint."""
    conversation_id: str


class UploadResponse(BaseModel):
    """Response body for the /upload endpoint."""
    message: str
    file_name: str
    num_pages: int
    num_chunks: int
    conversation_id: str


# ---------------------------------------------------------------------------
# POST /conversations
# ---------------------------------------------------------------------------

@router.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation():
    """Create a new conversation and return its ID."""
    request_id = get_request_id()
    
    conversation_id = _conversation_history.create_conversation()
    
    logger.info(
        "Created new conversation=%s request_id=%s",
        conversation_id,
        request_id,
    )
    
    return CreateConversationResponse(conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(conversation_id: str = Form(...), file: UploadFile = File(...)):
    """Upload and process a PDF document.

    Flow: Upload → Validate → Extract → Clean → Chunk → Embed → Store in ChromaDB
    """
    request_id = get_request_id()
    logger.info(
        "Upload request received: file=%s content_type=%s conversation_id=%s request_id=%s",
        file.filename,
        file.content_type,
        conversation_id,
        request_id,
    )

    metrics.total_uploads += 1

    # --- Validate conversation_id ---
    if not conversation_id:
        metrics.failed_uploads += 1
        raise HTTPException(status_code=400, detail="conversation_id is required.")

    # --- Validate file ---
    if not file.filename:
        metrics.failed_uploads += 1
        raise HTTPException(status_code=400, detail="No file name provided.")

    if not file.filename.lower().endswith(".pdf"):
        metrics.failed_uploads += 1
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected PDF, got '{file.filename}'.",
        )

    # --- Save uploaded file ---
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        content = await file.read()
        if not content:
            metrics.failed_uploads += 1
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path.write_bytes(content)
        logger.info("Saved uploaded file to %s (%d bytes)", file_path, len(content))
    except HTTPException:
        raise
    except Exception as exc:
        metrics.failed_uploads += 1
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc

    # --- Process PDF ---
    try:
        chunks = _pdf_processor.process(str(file_path), file.filename, conversation_id)
    except PDFProcessingError as exc:
        metrics.failed_uploads += 1
        logger.warning("PDF processing failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        metrics.failed_uploads += 1
        logger.error("Unexpected error during PDF processing: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to process PDF."
        ) from exc

    # --- Generate embeddings ---
    try:
        texts = [chunk.text for chunk in chunks]
        embeddings = _embedding_service.encode(texts)
    except Exception as exc:
        metrics.failed_uploads += 1
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to generate embeddings."
        ) from exc

    # --- Store in ChromaDB ---
    try:
        _chroma_store.add_chunks(chunks, embeddings)
    except Exception as exc:
        metrics.failed_uploads += 1
        logger.error("ChromaDB storage failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to store document chunks."
        ) from exc

    # --- Determine number of unique pages ---
    page_numbers = {chunk.page_number for chunk in chunks}

    metrics.successful_uploads += 1
    logger.info(
        "Upload complete: file=%s pages=%d chunks=%d conversation_id=%s doc_id=%s",
        file.filename,
        len(page_numbers),
        len(chunks),
        conversation_id,
        chunks[0].doc_id if chunks else "unknown",
    )

    return UploadResponse(
        message="PDF uploaded and processed successfully.",
        file_name=file.filename,
        num_pages=len(page_numbers),
        num_chunks=len(chunks),
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# POST /ask (SSE streaming)
# ---------------------------------------------------------------------------

@router.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question about uploaded documents.

    Returns a Server-Sent Events stream with:
    - event: token  — Individual text chunks as Groq generates them
    - event: done   — Final complete answer with citations and conversation_id
    - event: error  — Error information if something fails
    """
    request_id = get_request_id()
    metrics.total_queries += 1

    question = request.question.strip()
    if not question:
        metrics.failed_queries += 1
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # --- Conversation setup ---
    conversation_id = request.conversation_id
    if not conversation_id:
        metrics.failed_queries += 1
        raise HTTPException(status_code=400, detail="conversation_id is required.")

    logger.info(
        "Ask request: question='%s' conversation_id=%s request_id=%s",
        question[:80],
        conversation_id,
        request_id,
    )

    # --- Save user question to history ---
    try:
        _conversation_history.add_message(conversation_id, "user", question)
    except ConversationError as exc:
        logger.error("Failed to save user question: %s", exc)
        # Continue anyway — conversation persistence is not critical for answering

    # --- Get conversation history ---
    try:
        history = _conversation_history.get_history(conversation_id)
        # Exclude the message we just added (last one) to avoid duplication
        history_for_prompt = history[:-1] if history else []
    except ConversationError:
        history_for_prompt = []

    # --- Retrieve relevant chunks ---
    try:
        retrieved_chunks = _retriever.retrieve(question, conversation_id)
    except Exception as exc:
        metrics.failed_queries += 1
        logger.error("Retrieval failed: conversation_id=%s error=%s", conversation_id, exc)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve relevant documents."
        ) from exc

    logger.info("Retrieved %d chunks for question conversation_id=%s", len(retrieved_chunks), conversation_id)

    # --- Build context and prompt ---
    context_for_prompt = [
        {
            "text": chunk.text,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
        }
        for chunk in retrieved_chunks
    ]

    prompt = build_grounded_prompt(
        question=question,
        context_chunks=context_for_prompt,
        conversation_history=history_for_prompt,
    )

    # --- Build citations from retrieved chunks ---
    citations = []
    seen_citations = set()
    for chunk in retrieved_chunks:
        key = (chunk.file_name, chunk.page_number, chunk.chunk_id)
        if key not in seen_citations:
            seen_citations.add(key)
            citations.append({
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            })
    
    logger.info(
        "Built %d unique citations from retrieved chunks conversation_id=%s",
        len(citations),
        conversation_id,
    )

    # --- Stream response ---
    async def event_stream():
        """SSE event generator."""
        full_answer = []
        stream_start = time.perf_counter()

        try:
            async for text_chunk in _groq_client.generate_stream(prompt):
                full_answer.append(text_chunk)
                event_data = json.dumps({"text": text_chunk})
                yield f"event: token\ndata: {event_data}\n\n"

            # --- Stream complete ---
            stream_duration_ms = (time.perf_counter() - stream_start) * 1000
            complete_answer = "".join(full_answer)

            metrics.record_latency("streaming", stream_duration_ms)
            logger.info(
                "Streaming complete: conversation=%s duration_ms=%.2f answer_length=%d",
                conversation_id,
                stream_duration_ms,
                len(complete_answer),
            )

            # Save assistant answer to conversation history
            try:
                _conversation_history.add_message(
                    conversation_id,
                    "assistant",
                    complete_answer,
                    citations=citations,
                )
            except ConversationError as exc:
                logger.error("Failed to save assistant response: %s", exc)

            # Send final done event
            done_data = json.dumps({
                "answer": complete_answer,
                "citations": citations,
                "conversation_id": conversation_id,
            })
            yield f"event: done\ndata: {done_data}\n\n"

            metrics.successful_queries += 1

        except GroqError as exc:
            metrics.failed_queries += 1
            logger.error("Streaming error: %s", exc)
            error_data = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error_data}\n\n"

        except Exception as exc:
            metrics.failed_queries += 1
            logger.error("Unexpected streaming error: %s", exc)
            error_data = json.dumps({"error": "An unexpected error occurred during generation."})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id or "",
            "X-Conversation-ID": conversation_id,
        },
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """Return application health status and metrics summary."""
    chroma_status = _chroma_store.get_status()

    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.GROQ_MODEL,
        "chromadb": chroma_status,
        "metrics": metrics.get_summary(),
    }
