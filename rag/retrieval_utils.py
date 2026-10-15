"""Pure retrieval result structures and diversity selection, no DB dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class SearchHit:
    filename: str
    page: int
    chunk_number: int
    text: str
    distance: float
    doc_id: str = ""


def choose_hits(hits: Sequence[SearchHit], limit: int, balanced: bool = False) -> list[SearchHit]:
    """Choose closest hits, optionally reserving one passage per document.

    Balanced mode is a *coverage heuristic*, not an evidence-quality validator.
    All returned hits remain in ascending distance order, and irrelevant hits
    may still be retrieved from a selected document.
    """
    if limit < 1:
        return []
    ordered = sorted(hits, key=lambda h: h.distance)
    if not balanced:
        return ordered[:limit]
    chosen: list[SearchHit] = []
    seen: set[str] = set()
    for hit in ordered:
        identifier = hit.doc_id or hit.filename
        if identifier not in seen and len(chosen) < limit:
            seen.add(identifier)
            chosen.append(hit)
    for hit in ordered:
        if len(chosen) == limit:
            break
        if hit not in chosen:
            chosen.append(hit)
    return sorted(chosen, key=lambda h: h.distance)

