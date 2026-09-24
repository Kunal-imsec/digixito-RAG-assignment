"""
Semantic retrieval pipeline.

Combines the embedding service and ChromaDB store to retrieve
the most relevant document chunks for a given query.
"""

import logging

from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore, RetrievedChunk
from app.config import settings
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves top-K semantically similar chunks for a user query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_store: ChromaStore,
    ):
        self.embedding_service = embedding_service
        self.chroma_store = chroma_store

    @timed("retrieval_pipeline")
    def retrieve(
        self,
        query: str,
        conversation_id: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: User's question text.
            conversation_id: The conversation ID to filter results by.
            top_k: Number of chunks to retrieve (defaults to settings.TOP_K).

        Returns:
            List of RetrievedChunk objects ordered by relevance.
        """
        k = top_k or settings.TOP_K

        logger.info(
            "Retrieving top-%d chunks for query='%s' conversation_id=%s",
            k,
            query[:80] + "..." if len(query) > 80 else query,
            conversation_id,
        )

        # 1. Embed the query
        query_embedding = self.embedding_service.encode_query(query)

        # 2. Search ChromaDB with conversation filter
        chunks = self.chroma_store.query(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=k,
        )

        logger.info(
            "Retrieval complete: %d chunks returned (top_k=%d)",
            len(chunks),
            k,
        )

        return chunks
