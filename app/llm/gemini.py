"""
Gemini LLM client with streaming support.

Uses the google-genai SDK to interact with Gemini models.
Provides both synchronous and streaming generation methods.
"""

import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.config import settings
from app.llm.prompts import SYSTEM_INSTRUCTION
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Raised when Gemini API calls fail."""


class GeminiClient:
    """Wrapper around the Google GenAI SDK for Gemini generation."""

    def __init__(self):
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            logger.info("Initializing Gemini client with model=%s", settings.GEMINI_MODEL)
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    @timed("gemini_generate")
    def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt: The grounded prompt to send to Gemini.

        Returns:
            The generated text response.

        Raises:
            GeminiError: If the API call fails.
        """
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,  # Low temperature for factual, grounded answers
                    max_output_tokens=2048,
                ),
            )

            if not response or not response.text:
                logger.warning("Gemini returned empty response")
                return "insufficient information"

            return response.text

        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            raise GeminiError(f"Failed to generate response: {exc}") from exc

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text chunks from Gemini.

        Yields text chunks as they are produced by the model.
        The caller is responsible for accumulating the full response.

        Args:
            prompt: The grounded prompt to send to Gemini.

        Yields:
            Text chunks (strings) as they arrive.

        Raises:
            GeminiError: If the API call fails.
        """
        logger.info("Starting Gemini streaming generation")

        try:
            response_stream = self.client.models.generate_content_stream(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise GeminiError(f"Streaming generation failed: {exc}") from exc
