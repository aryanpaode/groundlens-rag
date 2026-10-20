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
from .document_ingestion import extract_document
from .advanced_config import IngestionOptions
from .lexical_store import LexicalStore, reciprocal_rank_fusion
from .raptor import build_raptor_nodes, gemini_summarizer
from .model_helpers import chunk_context
from .reranking import rerank as cross_rerank
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
        self.lexical = LexicalStore(self.manifest_file.with_name("groundlens_bm25.sqlite3"))

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

    def add_pdf(self, filename: str, content: bytes, embedding_model, *,
                options: IngestionOptions | None = None,
                gemini_api_key: str = "", model_name: str = "") -> tuple[str, int, int]:
        safe_name = Path(filename.replace("\\", "/")).name[:150] or "uploaded.pdf"
        if len(content) > 12 * 1024 * 1024:
            raise ValueError("PDF exceeds 12 MB upload limit.")
        doc_id = sha256(content).hexdigest()
        options = options or IngestionOptions()
        if doc_id in self._documents:
            if not self.lexical.has_document(doc_id):
                # Existing Pinecone documents used PyMuPDF with default chunking.
                # Preserve their original chunk IDs during lexical backfill.
                original_chunks, _ = extract_pdf_chunks(content)
                if len(original_chunks) != self._documents[doc_id]["chunks"]:
                    raise ValueError("Legacy PDF chunks differ from the manifest; re-index in a fresh namespace.")
                self.lexical.replace_document(doc_id, safe_name, original_chunks)
                return "lexical_backfilled", len(original_chunks), self._documents[doc_id]["page_count"]
            return "already_indexed", 0, 0
        chunks, page_count = extract_document(content, filename=safe_name,
                                              parser=options.parser,
                                              semantic_chunks=options.semantic_chunks,
                                              embedding_model=embedding_model)
        if not chunks:
            raise ValueError("No usable text found. Try Docling/OCR, or upload a text-searchable PDF.")
        texts = []
        outline = chunks[0].text[:400]
        for index, chunk in enumerate(chunks):
            if options.contextual_embeddings and index < options.max_context_calls:
                context = chunk_context(gemini_api_key, model_name, filename=safe_name,
                                        page=chunk.page, excerpt=chunk.text, doc_outline=outline)
                texts.append("Context: " + context + "\nPassage: " + chunk.text)
            else:
                texts.append(chunk.text)
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
                    "record_type": "leaf", "embedding_context": texts[len(records)][:420]
                      if options.contextual_embeddings else "",
                },
            })
        summary_count = 0
        if options.raptor:
            summarizer = gemini_summarizer(gemini_api_key, model_name)
            nodes = build_raptor_nodes(doc_id, chunks, vectors, embedding_model,
                                       summarizer, max_nodes=options.max_raptor_summaries)
            for node in nodes:
                records.append({
                    "id": node.id, "values": node.vector,
                    "metadata": {"doc_id": doc_id, "filename": safe_name,
                                 "page": 0, "chunk_number": 0, "text": node.text,
                                 "record_type": "summary", "level": node.level,
                                 "child_leaf_ids": ";".join(node.child_leaf_ids)},
                })
            summary_count = len(nodes)
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
        # Local lexical side-index is built only after successful cloud writes.
        # If SQLite fails, deterministic cloud record IDs make reupload safe.
        self.lexical.replace_document(doc_id, safe_name, chunks)
        doc = {"doc_id": doc_id, "filename": safe_name, "page_count": page_count,
               "chunks": len(chunks), "summaries": summary_count,
               "parser": options.parser, "semantic": options.semantic_chunks,
               "contextual": options.contextual_embeddings, "raptor": options.raptor}
        self._documents[doc_id] = doc
        self._save_manifest()
        return "indexed", len(chunks), page_count

    @property
    def lexical_coverage(self) -> int:
        return sum(self.lexical.has_document(doc_id) for doc_id in self._documents)

    def search(self, question: str, embedding_model, n_results: int = 5,
               doc_ids: Sequence[str] | None = None, balanced: bool = False,
               *, hybrid: bool = False, rerank: bool = False,
               use_hierarchy: bool = False) -> list[SearchHit]:
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
            filtered = sorted(known_ids)
        where = {"doc_id": {"$in": filtered}} if len(filtered) > 1 else {"doc_id": filtered[0]}
        embedding = embedding_model.encode([question], normalize_embeddings=True,
                                           show_progress_bar=False)[0]
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        if len(values) != DIMENSIONS:
            raise ValueError("Query vector dimension mismatch; use all-MiniLM-L6-v2.")
        leaf_count = sum(int(self._documents[d]["chunks"]) for d in filtered)
        total_count = leaf_count + sum(int(self._documents[d].get("summaries", 0)) for d in filtered)
        needs_pool = balanced or hybrid or rerank or use_hierarchy
        if needs_pool:
            candidate_count = min(total_count, max(5, n_results * 5))
        else:
            # Even when hierarchy search is OFF, its stored summary vectors
            # may rank ahead of PDF leaves in Pinecone. Retrieve enough extra
            # matches to return the requested number of real source passages.
            summary_count = max(0, total_count - leaf_count)
            candidate_count = min(total_count, n_results + summary_count)
        result = self._index.query(vector=[float(v) for v in values], top_k=candidate_count,
                                   include_metadata=True, include_values=False,
                                   filter=where, namespace=self.namespace)
        # RAPTOR summary nodes are *navigation hints*. A summary's similarity to
        # the question must NOT be copied to all of its leaf passages: doing so
        # can allow the first five unrelated children to displace stronger
        # evidence and produce a false "information not found" answer.
        leaf_matches: list[SearchHit] = []
        summary_children: dict[tuple[str, int, int], tuple[SearchHit, float]] = {}
        for match in _attribute(result, "matches", []) or []:
            metadata = _attribute(match, "metadata", {}) or {}
            passage = metadata.get("text", "")
            if not passage:
                continue
            distance = 1.0 - float(_attribute(match, "score", 0.0))
            if metadata.get("record_type") == "summary":
                if not use_hierarchy:
                    continue
                # The original PDF passage is still the citable unit. Retrieve
                # the children from the local exact-ID cache; never cite a
                # generated hierarchy summary as an original PDF page.
                for child_id in str(metadata.get("child_leaf_ids", "")).split(";")[:60]:
                    try:
                        page_text, chunk_text = child_id.split(":", 1)
                        page, chunk_number = int(page_text), int(chunk_text)
                    except (ValueError, TypeError):
                        continue
                    leaf = self.lexical.get(str(metadata["doc_id"]), page, chunk_number)
                    if leaf is None:
                        continue
                    key = (leaf.doc_id, leaf.page, leaf.chunk_number)
                    if key not in summary_children or distance < summary_children[key][1]:
                        summary_children[key] = (leaf, distance)
                continue
            leaf_matches.append(SearchHit(
                filename=str(metadata["filename"]), page=int(metadata["page"]),
                chunk_number=int(metadata["chunk_number"]), text=str(passage),
                distance=distance, doc_id=str(metadata["doc_id"]),
            ))

        # Preserve directly measured Pinecone scores, deduplicating exact pages
        # before combining them with any hierarchical candidates.
        direct_by_key: dict[tuple[str, int, int], SearchHit] = {}
        for hit in sorted(leaf_matches, key=lambda h: h.distance):
            direct_by_key.setdefault((hit.doc_id, hit.page, hit.chunk_number), hit)
        direct = list(direct_by_key.values())
        candidates = direct
        if use_hierarchy and summary_children:
            # Score *individual leaf passages* against the actual query. Do not
            # assume every member of a semantically similar cluster is relevant.
            # Reuse Pinecone's scores when it already returned that leaf; embed
            # only missing children, with a strict CPU-work bound.
            descendants = sorted(summary_children.items(), key=lambda x: (x[1][1], x[0]))[:80]
            missing = [(key, item[0]) for key, item in descendants if key not in direct_by_key]
            missing_distances: dict[tuple[str, int, int], float] = {}
            if missing:
                vectors = embedding_model.encode(
                    [leaf.text for _, leaf in missing],
                    normalize_embeddings=True, show_progress_bar=False,
                )
                if len(vectors) != len(missing):
                    raise RuntimeError("Hierarchical leaf embedding count mismatch")
                for (key, _), vector in zip(missing, vectors):
                    if len(vector) != DIMENSIONS:
                        raise ValueError("Hierarchical leaf vector dimension mismatch")
                    # Model embeddings and query are both L2-normalized.
                    sim = sum(float(a) * float(b) for a, b in zip(values, vector))
                    missing_distances[key] = 1.0 - sim

            hierarchical: list[SearchHit] = []
            for key, (leaf, parent_distance) in descendants:
                leaf_distance = (direct_by_key[key].distance if key in direct_by_key
                                 else missing_distances[key])
                # A small parent-context signal helps relevant thematic groups,
                # while leaf relevance remains the primary ranking criterion.
                score = 0.90 * leaf_distance + 0.10 * parent_distance
                hierarchical.append(SearchHit(
                    leaf.filename, leaf.page, leaf.chunk_number,
                    leaf.text, score, leaf.doc_id, "hierarchy",
                ))
            hierarchical.sort(key=lambda h: h.distance)
            # A highly similar *summary* can have many irrelevant children.
            # Do not reward those children merely for belonging to the group.
            # Require child relevance to be competitive with direct retrieval;
            # the bounded margin is heuristic, NOT a factual confidence score.
            if direct:
                direct_cutoff = direct[min(n_results, len(direct)) - 1].distance
                competitive_cutoff = min(direct_cutoff + 0.08, direct[0].distance + 0.30)
                hierarchical = [h for h in hierarchical
                                if h.distance <= competitive_cutoff]
            if hierarchical:
                # RRF combines the direct and summary-guided ranks; raw cosine
                # distances are never mixed with BM25 or summary scores.
                candidates = reciprocal_rank_fusion(direct, hierarchical,
                                                     max(15, n_results * 5))
                # Make hierarchy participation visible in Evidence Explorer.
                candidates = [SearchHit(h.filename, h.page, h.chunk_number, h.text,
                                        h.distance, h.doc_id, "hierarchy-rrf")
                              for h in candidates]

        if hybrid:
            lexical = self.lexical.query(question, filtered, max(20, n_results * 5))
            candidates = reciprocal_rank_fusion(candidates, lexical,
                                                 max(15, n_results * 5))
        if rerank:
            # Rerank only candidate passages, using the real user question.
            return choose_hits(cross_rerank(question, candidates, max(n_results, len(candidates))),
                               n_results, balanced=balanced)
        return choose_hits(candidates, n_results, balanced=balanced)

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
        self.lexical.delete_document(doc_id)
        del self._documents[doc_id]
        self._save_manifest()

    def clear_all(self) -> None:
        """Explicitly remove GroundLens namespace ONLY, not the entire index."""
        self._index.delete(delete_all=True, namespace=self.namespace)
        self._documents.clear()
        self.lexical.clear()
        self._save_manifest()
