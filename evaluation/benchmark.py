"""Retrieval-only evaluation: uses the configured store; no Gemini or answer-quality grading."""
from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter


def load_benchmark(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_retrieval(store, embedder, rows: list[dict], k: int = 5,
                       balanced: bool = False,
                       allowed_doc_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    """A case passes iff ALL its required document/page/phrase passages appear.

    Only positive cases for indexed documents are scored. Questions with no
    document answer require a separate generation-stage/manual evaluation.
    """
    indexed_docs = {d["filename"] for d in store.list_documents()}
    results = []
    reciprocal_ranks = []
    positive = 0
    successful = 0
    skipped = 0
    for row in rows:
        required_docs = [x.strip() for x in row.get("expected_documents", "").split(";") if x.strip()]
        expected_pages = [x.strip() for x in row.get("expected_pages", "").split(";")]
        expected_phrases = [x.strip() for x in row.get("expected_phrases", "").split(";")]
        result = {
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "question": row.get("question", ""),
            "expected_documents": row.get("expected_documents", ""),
            "expected_pages": row.get("expected_pages", ""),
            "status": "",
            "required_found": "",
            "first_relevant_rank": "",
            "retrieval_seconds": "",
            "retrieved_sources": "",
        }
        if not required_docs:
            result["status"] = "MANUAL: answer/absence evaluation"
            skipped += 1
        elif not set(required_docs).issubset(indexed_docs):
            result["status"] = "SKIPPED: required PDF not indexed"
            skipped += 1
        else:
            positive += 1
            t0 = perf_counter()
            search_options = {"n_results": k, "balanced": balanced}
            if allowed_doc_ids is not None:
                search_options["doc_ids"] = allowed_doc_ids
            hits = store.search(row["question"], embedder, **search_options)
            result["retrieval_seconds"] = round(perf_counter() - t0, 3)
            result["retrieved_sources"] = " | ".join(
                f"{h.filename} p.{h.page}" for h in hits
            )
            rank_per_requirement = []
            for index, document in enumerate(required_docs):
                page_text = expected_pages[index] if index < len(expected_pages) else ""
                phrase = expected_phrases[index].casefold() if index < len(expected_phrases) else ""
                rank = next((position for position, hit in enumerate(hits, 1)
                             if hit.filename == document
                             and (not page_text or str(hit.page) == page_text)
                             and (not phrase or phrase in hit.text.casefold())), None)
                if rank is not None:
                    rank_per_requirement.append(rank)
            success = len(rank_per_requirement) == len(required_docs)
            if success:
                successful += 1
                reciprocal_ranks.append(1 / max(rank_per_requirement))
            else:
                reciprocal_ranks.append(0)
            result["status"] = "PASS" if success else "MISS"
            result["required_found"] = f"{len(rank_per_requirement)}/{len(required_docs)}"
            result["first_relevant_rank"] = min(rank_per_requirement) if rank_per_requirement else ""
        results.append(result)
    summary = {
        "evaluated": positive,
        "retrieval_hit_rate": successful / positive if positive else None,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / positive if positive else None,
        "skipped_or_manual": skipped,
    }
    return results, summary
