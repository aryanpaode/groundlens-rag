"""Optional structure-aware and embedding-aware PDF ingestion.

Built-in PyMuPDF is always available in the base installation. Docling is
optional and intentionally opt-in due to its substantial CPU/memory overhead.
"""
from __future__ import annotations
from io import BytesIO
import re

from .pdf_processing import PageChunk, extract_pdf_chunks, normalize_text, split_words


def _sentence_units(text: str) -> list[str]:
    units = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalize_text(text))
    return [x.strip() for x in units if x.strip()]


def semantic_split(text: str, embedding_model, target_words: int = 170,
                   percentile: float = 80.0) -> list[str]:
    """Split when adjacent sentences are dissimilar, capped by word budget.

    Uses the already-loaded local embedding model. Does not call a second LLM.
    Unlike fixed word chunks, this preserves complete sentences where possible.
    """
    units = _sentence_units(text)
    if len(units) < 3:
        return split_words(text, target_words, 30)
    import numpy as np
    vec = np.asarray(embedding_model.encode(units, normalize_embeddings=True, show_progress_bar=False), dtype=float)
    similarity = np.einsum('ij,ij->i', vec[:-1], vec[1:])
    threshold = float(np.percentile(1.0 - similarity, percentile))
    result = []
    current = []
    words = 0
    for i, sentence in enumerate(units):
        count = len(sentence.split())
        if current and (words + count > target_words or
                        (i > 0 and words >= target_words // 3 and 1.0 - similarity[i - 1] >= threshold)):
            result.append(" ".join(current))
            current, words = [], 0
        if count > target_words:
            if current:
                result.append(" ".join(current))
                current, words = [], 0
            result.extend(split_words(sentence, target_words, 20))
        else:
            current.append(sentence)
            words += count
    if current:
        result.append(" ".join(current))
    return result


def extract_document(pdf_bytes: bytes, *, filename: str,
                     parser: str = "pymupdf", semantic_chunks: bool = False,
                     embedding_model=None) -> tuple[list[PageChunk], int]:
    if parser == "pymupdf" and not semantic_chunks:
        return extract_pdf_chunks(pdf_bytes)
    if parser == "pymupdf":
        import pymupdf as fitz
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            pages = [page.get_text('text', sort=True) for page in pdf]
    elif parser == "docling":
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import DocumentStream
        except ImportError as exc:
            raise RuntimeError("Docling is optional. Install requirements-advanced.txt to enable it.") from exc
        converter = DocumentConverter()
        result = converter.convert(DocumentStream(name=filename, stream=BytesIO(pdf_bytes)))
        pages = [result.document.export_to_markdown(page_no=i, image_mode="placeholder",
                                                    traverse_pictures=True)
                 for i in range(1, len(result.document.pages) + 1)]
    else:
        raise ValueError("Unsupported PDF parser")
    if semantic_chunks and embedding_model is None:
        raise ValueError("Semantic chunks require the existing CPU embedding model")
    chunks = []
    for page_no, raw in enumerate(pages, 1):
        text = normalize_text(raw)
        pieces = (semantic_split(text, embedding_model) if semantic_chunks
                  else split_words(text))
        for index, piece in enumerate(pieces, 1):
            chunks.append(PageChunk(page_no, index, piece))
    return chunks, len(pages)
