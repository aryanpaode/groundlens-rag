import fitz
import pytest

from rag.pdf_processing import extract_pdf_chunks, normalize_text, split_words


def test_word_chunk_overlap():
    text = " ".join(f"word{i}" for i in range(12))
    chunks = split_words(text, chunk_words=6, overlap_words=2)
    assert chunks[0] == "word0 word1 word2 word3 word4 word5"
    assert chunks[1].startswith("word4 word5")
    assert len(chunks) == 3


def test_invalid_chunk_parameters():
    with pytest.raises(ValueError):
        split_words("hello", chunk_words=3, overlap_words=3)


def test_pymupdf_extraction_preserves_page_numbers():
    doc = fitz.open()
    for index in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {index + 1} contains a distinct passage about project documentation.")
    pdf_bytes = doc.tobytes()
    doc.close()
    chunks, pages = extract_pdf_chunks(pdf_bytes)
    assert pages == 2
    assert [c.page for c in chunks] == [1, 2]
    assert "project documentation" in chunks[1].text


def test_empty_pdf_bytes_rejected():
    with pytest.raises(ValueError):
        extract_pdf_chunks(b"")


def test_normalize_text_removes_excess_spaces():
    assert normalize_text("a\n    b") == "a b"
