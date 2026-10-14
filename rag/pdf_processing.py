"""Extract page-aware text chunks from ordinary text-based PDFs."""
from __future__ import annotations

from dataclasses import dataclass
import re

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageChunk:
    page: int  # One-based PDF page number.
    chunk_number: int
    text: str


def normalize_text(raw: str) -> str:
    """Keep words readable while removing noisy PDF whitespace."""
    text = raw.replace("\u00ad", "")  # PDF soft-hyphen artifacts.
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_words(text: str, chunk_words: int = 170, overlap_words: int = 35) -> list[str]:
    """Split on word boundaries with a small overlap between adjacent chunks."""
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_words]
        if not part:
            break
        # Do not append tiny nearly-duplicate trailing chunks.
        if chunks and len(part) <= overlap_words:
            break
        chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_pdf_chunks(pdf_bytes: bytes, chunk_words: int = 170, overlap_words: int = 35) -> tuple[list[PageChunk], int]:
    """Return chunks tied to their original PDF pages and total page count.

    Scanned/image-only PDFs are deliberately not OCRed in this first version.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Unable to open this PDF. Check that it is a valid PDF file.") from exc

    with pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported yet.")
        count = len(pdf)
        result: list[PageChunk] = []
        for page_index, page in enumerate(pdf):
            text = normalize_text(page.get_text("text", sort=True))
            for part_index, chunk in enumerate(split_words(text, chunk_words, overlap_words)):
                result.append(PageChunk(page=page_index + 1, chunk_number=part_index + 1, text=chunk))
        return result, count
