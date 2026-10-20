"""Validated, opt-in advanced RAG switches. Defaults retain the existing system.

Index-time choices apply only to *new* documents. Runtime choices can be changed
per question, without changing the Pinecone dense index/schema.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionOptions:
    parser: str = "pymupdf"          # pymupdf | docling
    semantic_chunks: bool = False
    contextual_embeddings: bool = False
    raptor: bool = False
    max_context_calls: int = 30
    max_raptor_summaries: int = 20

    def __post_init__(self):
        if self.parser not in ("pymupdf", "docling"):
            raise ValueError("parser must be 'pymupdf' or 'docling'")
        if self.max_context_calls < 0 or self.max_context_calls > 100:
            raise ValueError("max_context_calls must be between 0 and 100")
        if self.max_raptor_summaries < 0 or self.max_raptor_summaries > 100:
            raise ValueError("max_raptor_summaries must be between 0 and 100")


@dataclass(frozen=True)
class RetrievalOptions:
    hybrid: bool = False
    rerank: bool = False
    corrective: bool = False
    hyde: bool = False
    use_hierarchy: bool = False
    dspy_generation: bool = False


ADVANCED_FEATURES = {
    "docling": "Opt-in IBM Docling page-by-page PDF conversion",
    "semantic_chunks": "Embedding-aware sentence boundaries",
    "contextual_embeddings": "Gemini-generated chunk descriptions during indexing",
    "raptor": "RAPTOR-inspired recursively clustered and summarized index nodes",
    "hybrid": "Pinecone dense vectors + local SQLite FTS5 BM25 using RRF",
    "rerank": "CrossEncoder CPU reranking of candidate passages",
    "corrective": "One bounded corrective retrieval pass when evidence is weak",
    "hyde": "HyDE hypothetical-answer query embedding (Gemini)",
    "ragas": "Optional offline RAGAS evaluation script",
    "dspy": "Optional offline DSPy prompt optimization script",
}
