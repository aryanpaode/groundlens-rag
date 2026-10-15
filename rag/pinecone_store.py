"""GroundLens cloud vector store: local MiniLM embeddings + Pinecone serverless.

Only original PDFs and local embedding weights remain on the laptop. Pinecone
receives chunk vectors, texts and source metadata. Its index is *never* deleted;
all operations are limited to the configured GroundLens namespace.

A small local manifest tracks files indexed by this single-user app. It does not
contain extracted text and must not be committed to GitHub.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence

from .pdf_processing import extract_pdf_chunks
from .retrieval_utils import SearchHit, choose_hits

DIMENSIONS = 384  # sentence-transformers/all-MiniLM-L6-v2


def _attribute(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _batched(items: list, batch_size: int):
    for offset in range(0, len(items), batch_size):
        yield items[offset:offset + batch_size]


class PineconeVectorStore:
    """Small-project document storage with consistent document/page citations.

    All index operations are scoped to ``namespace``; stored record text is
    intentionally accessible to Pinecone, so avoid confidential uploads.

    ``client``/``index`` injection permits local unit testing without API keys.
    """

    def __init__(self, api_key: str, index_name: str, namespace: str,
                 manifest_file: str | Path, cloud: str = "aws",
                 region: str = "us-east-1", client=None, index=None):
        if not index_name or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?", index_name):
            raise ValueError("PINECONE_INDEX must be 1–45 lowercase letters, numbers or hyphens.")
        if not namespace or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}", namespace):
            raise ValueError("Use a nonempty letters/numbers/underscores/hyphens PINECONE_NAMESPACE.")
        if (not api_key.strip() or api_key.startswith("REPLACE_")) and (client is None or index is None):
            raise ValueError("PINECONE_API_KEY is missing. Configure it in your private .env.")
        self.index_name = index_name
        self.namespace = namespace
        self.manifest_file = Path(manifest_file)
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self._documents = self._load_manifest()

        if index is not None:
            self._index = index
            self._pc = client
        else:
            if client is None:
                from pinecone import Pinecone
                client = Pinecone(api_key=api_key)
            self._pc = client
            index_info = self._ensure_index(cloud, region)
            # Prefer the already-fetched host: no second control-plane lookup.
            host = _attribute(index_info, "host", None)
            self._index = client.Index(host=host) if host else client.Index(index_name)

    def _load_manifest(self) -> dict:
        if not self.manifest_file.exists():
            return {}
        try:
            obj = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Pinecone local manifest is unreadable. Restore it or re-index documents.") from exc
        if obj.get("index") != self.index_name or obj.get("namespace") != self.namespace:
            raise RuntimeError("Manifest index/namespace differs from .env. Use a different runtime directory or restore settings.")
        return dict(obj.get("documents", {}))

    def _save_manifest(self) -> None:
        data = {"index": self.index_name, "namespace": self.namespace, "documents": self._documents}
        pending = self.manifest_file.with_name(self.manifest_file.name + ".tmp")
        pending.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        pending.replace(self.manifest_file)

    def _ensure_index(self, cloud: str, region: str):
        from pinecone import ServerlessSpec

        if not self._pc.has_index(self.index_name):
            # Pinecone Starter tiers may restrict indexes and regions. Explain
            # rather than silently deleting/reusing a user's other index.
            self._pc.create_index(
                name=self.index_name, dimension=DIMENSIONS, metric="cosine",
                vector_type="dense", spec=ServerlessSpec(cloud=cloud, region=region),
            )
        # Index readiness can lag initial creation; check before attempting upsert.
        info = None
        for _ in range(24):
            info = self._pc.describe_index(self.index_name)
            status = _attribute(info, "status", {})
            if bool(_attribute(status, "ready", True)):
                break
            time.sleep(2)
        if info is None or not bool(_attribute(_attribute(info, "status", {}), "ready", True)):
            raise RuntimeError("Pinecone index is still initializing. Wait and relaunch GroundLens.")
        dimension = _attribute(info, "dimension", None)
        metric = _attribute(info, "metric", None)
        if dimension is not None and int(dimension) != DIMENSIONS:
            raise ValueError(f"Pinecone index dimension is {dimension}; GroundLens requires {DIMENSIONS}. Create a matching index.")
        if metric is not None and str(metric).lower() != "cosine":
            raise ValueError(f"Pinecone index metric is {metric}; GroundLens requires cosine.")
        if _attribute(info, "embed", None):
            raise ValueError("Use an external-embedding dense index, NOT a Pinecone integrated-embedding index.")
        return info

    @property
    def count(self) -> int:
        # Local confirmed-manifest count; Pinecone writes may become queryable
        # after a short consistency delay. This is not a live Pinecone statistic.
        return sum(int(doc["chunks"]) for doc in self._documents.values())

    def list_documents(self) -> list[dict]:
        return sorted(
            [dict(doc) for doc in self._documents.values()],
            key=lambda d: (d["filename"].casefold(), d["doc_id"]),
        )

    def add_pdf(self, filename: str, content: bytes, embedding_model) -> tuple[str, int, int]:
        safe_name = Path(filename.replace("\\", "/")).name[:150] or "uploaded.pdf"
        if len(content) > 12 * 1024 * 1024:
            raise ValueError("PDF exceeds 12 MB upload limit.")
        doc_id = sha256(content).hexdigest()
        if doc_id in self._documents:
            return "already_indexed", 0, 0
        chunks, page_count = extract_pdf_chunks(content)
        if not chunks:
            raise ValueError("No selectable text found. Scanned PDFs need OCR first.")
        texts = [chunk.text for chunk in chunks]
        vectors = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding model returned the wrong number of vectors.")

        records = []
        for chunk, vector in zip(chunks, vectors):
            values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if len(values) != DIMENSIONS:
                raise ValueError(f"Embedding vector has {len(values)} dimensions; expected {DIMENSIONS}.")
            records.append({
                "id": f"{doc_id}:{chunk.page}:{chunk.chunk_number}",
                "values": [float(v) for v in values],
                "metadata": {
                    "doc_id": doc_id, "filename": safe_name,
                    "page": int(chunk.page), "chunk_number": int(chunk.chunk_number),
                    "page_count": int(page_count), "text": chunk.text,
                },
            })
        # The deterministic IDs allow safe retry after a partial upload. We
        # record success only after every upsert returns without error.
        try:
            for batch in _batched(records, 50):
                self._index.upsert(vectors=batch, namespace=self.namespace)
        except Exception:
            # Remove any partially uploaded chunks for this document only.
            try:
                self._index.delete(ids=[r["id"] for r in records], namespace=self.namespace)
            except Exception:
                pass  # Retrying overwrites deterministic IDs; never call delete_all.
            raise
        doc = {"doc_id": doc_id, "filename": safe_name, "page_count": page_count,
               "chunks": len(records)}
        self._documents[doc_id] = doc
        self._save_manifest()
        return "indexed", len(records), page_count

    def search(self, question: str, embedding_model, n_results: int = 5,
               doc_ids: Sequence[str] | None = None, balanced: bool = False) -> list[SearchHit]:
        if not question.strip() or not self.count or n_results < 1:
            return []
        if doc_ids is not None and not doc_ids:
            return []
        known_ids = set(self._documents)
        if doc_ids is not None:
            filtered = list(dict.fromkeys(doc_ids))
            if not set(filtered).issubset(known_ids):
                raise ValueError("Selected document is not indexed in this workspace.")
        else:
            filtered = list(known_ids)
        where = {"doc_id": {"$in": filtered}} if len(filtered) > 1 else {"doc_id": filtered[0]}
        embedding = embedding_model.encode([question], normalize_embeddings=True,
                                           show_progress_bar=False)[0]
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        if len(values) != DIMENSIONS:
            raise ValueError("Query vector dimension mismatch; use all-MiniLM-L6-v2.")
        candidate_count = min(self.count, max(5, n_results * 5) if balanced else n_results)
        result = self._index.query(vector=[float(v) for v in values], top_k=candidate_count,
                                   include_metadata=True, include_values=False,
                                   filter=where, namespace=self.namespace)
        hits = []
        for match in _attribute(result, "matches", []) or []:
            metadata = _attribute(match, "metadata", {}) or {}
            text = metadata.get("text", "")
            if not text:
                continue
            score = _attribute(match, "score", 0.0)
            hits.append(SearchHit(
                filename=str(metadata["filename"]), page=int(metadata["page"]),
                chunk_number=int(metadata["chunk_number"]), text=str(text),
                distance=1.0 - float(score), doc_id=str(metadata["doc_id"]),
            ))
        return choose_hits(hits, n_results, balanced=balanced)

    def delete_document(self, doc_id: str) -> None:
        if doc_id not in self._documents:
            raise ValueError("Unknown document ID.")
        # Restrict deletion to exact deterministic IDs; never delete globally.
        document = self._documents[doc_id]
        # Serverless index.list can enumerate only this document's IDs by prefix.
        # Avoid relying on potentially inconsistent eventual query results.
        for ids in self._index.list(prefix=f"{doc_id}:", namespace=self.namespace):
            if ids:
                self._index.delete(ids=list(ids), namespace=self.namespace)
        del self._documents[doc_id]
        self._save_manifest()

    def clear_all(self) -> None:
        """Explicitly remove GroundLens namespace ONLY, not the entire index."""
        self._index.delete(delete_all=True, namespace=self.namespace)
        self._documents.clear()
        self._save_manifest()
