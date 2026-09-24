"""
ChromaDB vector store management.

Handles collection creation, chunk storage, and semantic similarity queries.
Uses PersistentClient for disk-based storage.
"""

import logging
from dataclasses import dataclass

import chromadb

from app.config import settings
from app.ingestion.pdf_processor import TextChunk
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from ChromaDB with its metadata and similarity score."""

    text: str
    file_name: str
    page_number: int
    chunk_id: str
    conversation_id: str
    doc_id: str
    distance: float


class ChromaStore:
    """Manages a ChromaDB collection for document chunk storage and retrieval."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-initialize the ChromaDB persistent client."""
        if self._client is None:
            logger.info("Initializing ChromaDB at path=%s", self._persist_dir)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info("ChromaDB client initialized")
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the document collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (count=%d)",
                self._collection_name,
                self._collection.count(),
            )
        return self._collection

    @timed("chromadb_store")
    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Store text chunks with their pre-computed embeddings.

        Args:
            chunks: List of TextChunk objects from the PDF processor.
            embeddings: Corresponding embedding vectors.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "count mismatch."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "conversation_id": chunk.conversation_id,
                "doc_id": chunk.doc_id,
            }
            for chunk in chunks
        ]

        # ChromaDB has a batch size limit; add in batches of 500
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(
            "Stored %d chunks in collection '%s'",
            len(chunks),
            self._collection_name,
        )
        return len(chunks)

    @timed("chromadb_query")
    def query(
        self,
        query_embedding: list[float],
        conversation_id: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Query the collection for semantically similar chunks.

        Args:
            query_embedding: The query's embedding vector.
            conversation_id: The conversation ID to filter results by.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by similarity.
        """
        k = top_k or settings.TOP_K

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"conversation_id": conversation_id},
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved.append(
                    RetrievedChunk(
                        text=doc,
                        file_name=meta["file_name"],
                        page_number=meta["page_number"],
                        chunk_id=meta["chunk_id"],
                        conversation_id=meta["conversation_id"],
                        doc_id=meta["doc_id"],
                        distance=dist,
                    )
                )

        logger.info(
            "Retrieved %d chunks (top_k=%d, conversation_id=%s, collection_size=%d)",
            len(retrieved),
            k,
            conversation_id,
            self.collection.count(),
        )
        
        # Log retrieved chunk metadata for debugging
        for chunk in retrieved:
            logger.info(
                "Retrieved chunk: conversation_id=%s file_name=%s page=%d chunk_id=%s",
                chunk.conversation_id,
                chunk.file_name,
                chunk.page_number,
                chunk.chunk_id,
            )
        
        return retrieved

    def get_status(self) -> dict:
        """Return basic status information for health checks."""
        try:
            count = self.collection.count()
            return {"status": "healthy", "collection": self._collection_name, "document_count": count}
        except Exception as e:
            logger.error("ChromaDB health check failed: %s", e)
            return {"status": "unhealthy", "error": str(e)}
