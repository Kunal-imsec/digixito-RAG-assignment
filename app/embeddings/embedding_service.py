"""
Sentence Transformer embedding service.

Wraps the sentence-transformers library to provide a simple encode interface
used for both document chunk embeddings and query embeddings.
"""

import logging

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages the Sentence Transformer model and provides encoding."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info("Loading embedding model=%s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding model loaded: %s (dimension=%d)",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()

    @timed("embedding_generation")
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        logger.info("Encoding %d text(s) with model=%s", len(texts), self._model_name)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Convert numpy arrays to plain Python lists for ChromaDB compatibility
        result = [emb.tolist() for emb in embeddings]

        logger.info(
            "Generated %d embeddings (dimension=%d)",
            len(result),
            len(result[0]) if result else 0,
        )
        return result

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string.

        Args:
            query: The user's question text.

        Returns:
            Single embedding vector as a list of floats.
        """
        result = self.encode([query])
        return result[0]
