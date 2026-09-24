"""
PDF processing pipeline: extraction, cleaning, and chunking.

Uses PyMuPDF (pymupdf) for text extraction.
Produces metadata-enriched text chunks ready for embedding and storage.
"""

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.config import settings
from app.observability.metrics import timed

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text extracted from a PDF, enriched with metadata."""

    chunk_id: str
    text: str
    file_name: str
    page_number: int
    conversation_id: str
    doc_id: str


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""


class PDFProcessor:
    """Handles PDF text extraction, cleaning, and chunking."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    @timed("pdf_processing")
    def process(self, file_path: str, file_name: str, conversation_id: str) -> list[TextChunk]:
        """Full pipeline: extract → clean → chunk a PDF file.

        Args:
            file_path: Path to the PDF file on disk.
            file_name: Original uploaded file name (used in metadata).
            conversation_id: The conversation ID for this upload.

        Returns:
            List of TextChunk objects ready for embedding.

        Raises:
            PDFProcessingError: If the PDF is invalid, empty, or has no text.
        """
        logger.info(
            "Processing PDF file=%s path=%s chunk_size=%d overlap=%d conversation_id=%s",
            file_name,
            file_path,
            self.chunk_size,
            self.chunk_overlap,
            conversation_id,
        )

        # 1. Extract raw text per page
        pages = self._extract_text(file_path, file_name)

        # 2. Clean text on each page
        cleaned_pages = []
        for page_num, raw_text in pages:
            cleaned = self._clean_text(raw_text)
            if cleaned:
                cleaned_pages.append((page_num, cleaned))

        if not cleaned_pages:
            raise PDFProcessingError(
                f"No extractable text found in '{file_name}'. "
                "The PDF may contain only images (OCR is not supported)."
            )

        logger.info(
            "Extracted text from %d pages (of %d total) in file=%s",
            len(cleaned_pages),
            len(pages),
            file_name,
        )

        # 3. Chunk the cleaned text
        doc_id = uuid.uuid4().hex
        chunks = self._chunk_pages(cleaned_pages, file_name, conversation_id, doc_id)

        logger.info(
            "Created %d chunks from file=%s",
            len(chunks),
            file_name,
        )

        return chunks

    def _extract_text(
        self, file_path: str, file_name: str
    ) -> list[tuple[int, str]]:
        """Extract raw text from each page of a PDF.

        Returns:
            List of (page_number, raw_text) tuples. Page numbers are 1-indexed.
        """
        try:
            doc = pymupdf.open(file_path)
        except Exception as exc:
            raise PDFProcessingError(
                f"Failed to open PDF '{file_name}': {exc}"
            ) from exc

        if doc.page_count == 0:
            doc.close()
            raise PDFProcessingError(f"PDF '{file_name}' has no pages.")

        pages: list[tuple[int, str]] = []
        try:
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                text = page.get_text(sort=True)
                pages.append((page_idx + 1, text))  # 1-indexed
        finally:
            doc.close()

        return pages

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text.

        - Unicode normalization (NFC)
        - Remove control characters (keep newlines, tabs, spaces)
        - Collapse multiple whitespace into single spaces
        - Strip leading/trailing whitespace
        """
        # Unicode normalize
        text = unicodedata.normalize("NFC", text)

        # Remove control characters except common whitespace
        text = re.sub(r"[^\S \n\t]", "", text)

        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Collapse multiple newlines into double newline (paragraph boundary)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _chunk_pages(
        self, pages: list[tuple[int, str]], file_name: str, conversation_id: str, doc_id: str
    ) -> list[TextChunk]:
        """Split cleaned page texts into overlapping chunks.

        Each chunk retains its source page number. When a chunk spans a page
        boundary during overlap, it is attributed to the page where the chunk
        starts.
        """
        chunks: list[TextChunk] = []

        for page_num, page_text in pages:
            if not page_text:
                continue

            page_chunks = self._split_text(page_text, page_num, file_name, conversation_id, doc_id)
            chunks.extend(page_chunks)

        return chunks

    def _split_text(
        self, text: str, page_number: int, file_name: str, conversation_id: str, doc_id: str
    ) -> list[TextChunk]:
        """Split a single page's text into fixed-size overlapping chunks."""
        chunks: list[TextChunk] = []
        start = 0
        text_len = len(text)
        chunk_index = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                # Generate deterministic chunk_id: doc_id_page_N_chunk_M
                chunk_id = f"{doc_id}_page_{page_number}_chunk_{chunk_index}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        file_name=file_name,
                        page_number=page_number,
                        conversation_id=conversation_id,
                        doc_id=doc_id,
                    )
                )
                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size  # Safety: avoid infinite loop
            start += step

        return chunks
