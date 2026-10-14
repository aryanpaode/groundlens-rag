"""Local, persistent, page-aware ChromaDB storage and retrieval.

All embeddings are created on the laptop with SentenceTransformers. Only
retrieved passages are later sent to the selected Gemini model.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

import chromadb

from .pdf_processing import extract_pdf_chunks


from .retrieval_utils import SearchHit, choose_hits


class LocalVectorStore:
    """Compatible with the v1 Chroma collection; no migration required."""

    def __init__(self, db_directory: str | Path):
        db_path = Path(db_directory)
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._name = "groundlens_pdf_chunks"
        self._collection = self._client.get_or_create_collection(name=self._name)

    @property
    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        data = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in data.get("metadatas", []) or []:
            if not meta:
                continue
            doc_id = str(meta["doc_id"])
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": str(meta["filename"]),
                    "page_count": int(meta["page_count"]),
                    "chunks": 0,
                }
            docs[doc_id]["chunks"] += 1
        return sorted(docs.values(), key=lambda d: (d["filename"].lower(), d["doc_id"]))

    def add_pdf(self, filename: str, content: bytes, embedding_model) -> tuple[str, int, int]:
        safe_name = Path(filename.replace("\\", "/")).name[:150] or "uploaded.pdf"
        doc_id = sha256(content).hexdigest()
        if self._collection.get(where={"doc_id": doc_id}, limit=1).get("ids"):
            return "already_indexed", 0, 0

        chunks, page_count = extract_pdf_chunks(content)
        if not chunks:
            raise ValueError("No selectable text found. Scanned PDFs need OCR (not included).")
        texts = [chunk.text for chunk in chunks]
        vectors = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vector_list = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        ids = [f"{doc_id}:{chunk.page}:{chunk.chunk_number}" for chunk in chunks]
        metadatas = [
            {"doc_id": doc_id, "filename": safe_name, "page": chunk.page,
             "chunk_number": chunk.chunk_number, "page_count": page_count}
            for chunk in chunks
        ]
        # Chroma add is atomic at the batch level for this small-project corpus.
        self._collection.add(ids=ids, documents=texts, embeddings=vector_list, metadatas=metadatas)
        return "indexed", len(chunks), page_count

    def search(self, question: str, embedding_model, n_results: int = 5,
               doc_ids: Sequence[str] | None = None, balanced: bool = False) -> list[SearchHit]:
        if not question.strip() or self.count == 0:
            return []
        if doc_ids is not None and not doc_ids:
            return []
        where = None
        if doc_ids is not None:
            unique_ids = list(dict.fromkeys(doc_ids))
            where = ({"doc_id": unique_ids[0]} if len(unique_ids) == 1
                     else {"doc_id": {"$in": unique_ids}})
        query_vector = embedding_model.encode([question], normalize_embeddings=True,
                                               show_progress_bar=False)[0]
        query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        candidate_count = min(self.count, max(5, n_results * 5) if balanced else n_results)
        kwargs = {"query_embeddings": [query_vector], "n_results": candidate_count,
                  "include": ["documents", "metadatas", "distances"]}
        if where is not None:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        hits = [
            SearchHit(filename=str(meta["filename"]), page=int(meta["page"]),
                      chunk_number=int(meta["chunk_number"]), text=text,
                      distance=float(distance), doc_id=str(meta["doc_id"]))
            for text, meta, distance in zip(result["documents"][0],
                                            result["metadatas"][0], result["distances"][0])
            if meta and text
        ]
        return choose_hits(hits, n_results, balanced=balanced)

    def delete_document(self, doc_id: str) -> None:
        if not doc_id or len(doc_id) != 64:
            raise ValueError("Invalid document identifier.")
        self._collection.delete(where={"doc_id": doc_id})

    def clear_all(self) -> None:
        self._client.delete_collection(name=self._name)
        self._collection = self._client.get_or_create_collection(name=self._name)
