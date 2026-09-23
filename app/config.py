"""
Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-check all configuration values.
API keys and secrets are never logged or exposed in error responses.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central application settings loaded from environment variables / .env file."""

    # --- Gemini LLM ---
    GEMINI_API_KEY: str = Field(
        ...,
        description="Google Gemini API key (required)",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name for generation",
    )

    # --- Embedding ---
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-Transformers model name",
    )

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = Field(
        default="data/chromadb",
        description="Directory for ChromaDB persistent storage",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="documents",
        description="ChromaDB collection name",
    )

    # --- Chunking ---
    CHUNK_SIZE: int = Field(
        default=500,
        description="Character count per text chunk",
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Character overlap between consecutive chunks",
    )

    # --- Retrieval ---
    TOP_K: int = Field(
        default=5,
        description="Number of chunks to retrieve for each query",
    )

    # --- Conversation ---
    CONVERSATION_DIR: str = Field(
        default="data/conversations",
        description="Directory for conversation history JSON files",
    )
    MAX_CONVERSATION_HISTORY: int = Field(
        default=10,
        description="Maximum number of recent messages included in prompt context",
    )

    # --- Logging ---
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application log level (DEBUG, INFO, WARNING, ERROR)",
    )

    # --- Server ---
    UPLOAD_DIR: str = Field(
        default="data/uploads",
        description="Directory for temporarily stored uploaded PDFs",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton settings instance — import this across the application
settings = Settings()
