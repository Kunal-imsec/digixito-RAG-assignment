"""
Conversation history management.

Stores conversation messages in JSON files on disk, one file per conversation.
Each conversation is identified by a UUID.

Provides:
- Create new conversations
- Append messages (user questions and assistant answers with citations)
- Retrieve recent history for prompt context
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


class ConversationError(Exception):
    """Raised when conversation operations fail."""


class ConversationHistory:
    """Manages per-conversation message history stored as JSON files."""

    def __init__(self, storage_dir: str | None = None):
        self._storage_dir = Path(storage_dir or settings.CONVERSATION_DIR)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Conversation storage directory: %s", self._storage_dir)

    def _conversation_path(self, conversation_id: str) -> Path:
        """Get the file path for a conversation."""
        # Sanitize conversation_id to prevent path traversal
        safe_id = conversation_id.replace("/", "").replace("\\", "").replace("..", "")
        return self._storage_dir / f"{safe_id}.json"

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID.

        Returns:
            A new UUID conversation_id.
        """
        conversation_id = uuid.uuid4().hex
        path = self._conversation_path(conversation_id)

        data = {
            "conversation_id": conversation_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Created new conversation=%s", conversation_id)
        except Exception as exc:
            logger.error("Failed to create conversation: %s", exc)
            raise ConversationError(f"Failed to create conversation: {exc}") from exc

        return conversation_id

    @timed("conversation_add_message")
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> None:
        """Append a message to a conversation.

        Args:
            conversation_id: The conversation UUID.
            role: 'user' or 'assistant'.
            content: The message text.
            citations: Optional list of citation dicts (for assistant messages).
        """
        path = self._conversation_path(conversation_id)

        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                # Auto-create if not found
                data = {
                    "conversation_id": conversation_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "messages": [],
                }

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if citations:
                message["citations"] = citations

            data["messages"].append(message)

            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            logger.info(
                "Added %s message to conversation=%s (total=%d)",
                role,
                conversation_id,
                len(data["messages"]),
            )

        except Exception as exc:
            logger.error(
                "Failed to add message to conversation=%s: %s",
                conversation_id,
                exc,
            )
            raise ConversationError(
                f"Failed to save message: {exc}"
            ) from exc

    @timed("conversation_get_history")
    def get_history(
        self,
        conversation_id: str,
        max_messages: int | None = None,
    ) -> list[dict]:
        """Retrieve recent conversation messages.

        Args:
            conversation_id: The conversation UUID.
            max_messages: Maximum number of recent messages to return.
                Defaults to settings.MAX_CONVERSATION_HISTORY.

        Returns:
            List of message dicts with keys: role, content, timestamp,
            and optionally citations.
        """
        limit = max_messages or settings.MAX_CONVERSATION_HISTORY
        path = self._conversation_path(conversation_id)

        if not path.exists():
            logger.info(
                "No history found for conversation=%s, returning empty",
                conversation_id,
            )
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])

            # Return only the most recent messages
            recent = messages[-limit:] if len(messages) > limit else messages

            logger.info(
                "Retrieved %d messages for conversation=%s (total=%d)",
                len(recent),
                conversation_id,
                len(messages),
            )
            return recent

        except Exception as exc:
            logger.error(
                "Failed to read conversation=%s: %s",
                conversation_id,
                exc,
            )
            raise ConversationError(
                f"Failed to read conversation history: {exc}"
            ) from exc

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        return self._conversation_path(conversation_id).exists()
