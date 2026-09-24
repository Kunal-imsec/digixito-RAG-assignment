"""
Groq LLM client with streaming support.

Uses the OpenAI SDK to interact with Groq's OpenAI-compatible API.
Provides both synchronous and streaming generation methods.
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from app.config import settings
from app.llm.prompts import SYSTEM_INSTRUCTION
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


class GroqError(Exception):
    """Raised when Groq API calls fail."""


class GroqClient:
    """Wrapper around the OpenAI SDK for Groq generation."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the Groq client."""
        if self._client is None:
            logger.info("Initializing Groq client with model=%s", settings.GROQ_MODEL)
            self._client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
        return self._client

    @timed("groq_generate")
    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt: The grounded prompt to send to Groq.

        Returns:
            The generated text response.

        Raises:
            GroqError: If the API call fails.
        """
        try:
            response = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for factual, grounded answers
                max_tokens=2048,
            )

            if not response or not response.choices or not response.choices[0].message.content:
                logger.warning("Groq returned empty response")
                return "insufficient information"

            return response.choices[0].message.content

        except APIError as exc:
            logger.error("Groq API error: %s", exc)
            raise GroqError(f"Groq API error: {exc}") from exc
        except APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
            raise GroqError(f"Groq connection error: {exc}") from exc
        except RateLimitError as exc:
            logger.error("Groq rate limit error: %s", exc)
            raise GroqError(f"Groq rate limit error: {exc}") from exc
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            raise GroqError(f"Failed to generate response: {exc}") from exc

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream generated text chunks from Groq.

        Yields text chunks as they are produced by the model.
        The caller is responsible for accumulating the full response.

        Args:
            prompt: The grounded prompt to send to Groq.

        Yields:
            Text chunks (strings) as they arrive.

        Raises:
            GroqError: If the API call fails.
        """
        logger.info("Starting Groq streaming generation")

        try:
            response_stream = await self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                stream=True,
            )

            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APIError as exc:
            logger.error("Groq API error during streaming: %s", exc)
            raise GroqError(f"Groq API error: {exc}") from exc
        except APIConnectionError as exc:
            logger.error("Groq connection error during streaming: %s", exc)
            raise GroqError(f"Groq connection error: {exc}") from exc
        except RateLimitError as exc:
            logger.error("Groq rate limit error during streaming: %s", exc)
            raise GroqError(f"Groq rate limit error: {exc}") from exc
        except Exception as exc:
            logger.error("Groq streaming failed: %s", exc)
            raise GroqError(f"Streaming generation failed: {exc}") from exc
