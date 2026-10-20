"""Bounded evidence heuristic for deciding when to attempt corrective retrieval.

This is deliberately not a calibrated relevance model. The correction strategy
stays inside selected PDFs, so it cannot invent external source citations.
"""
from __future__ import annotations
import re

STOP = {'what','when','where','why','how','does','are','the','and','for','from','into',
        'this','these','those','which','with','between','about','describe','explain'}


def keywords(text: str) -> set[str]:
    return {s for s in re.findall(r'[a-z0-9]{3,}', text.lower()) if s not in STOP}


def weak_evidence(question: str, hits) -> bool:
    if not hits:
        return True
    terms = keywords(question)
    if not terms:
        return False
    overlap = max(len(terms.intersection(keywords(h.text))) / len(terms) for h in hits)
    # Do not compare score thresholds between cosine, BM25, or RRF.
    return overlap < 0.25


def corrected_query(question: str) -> str:
    important = list(dict.fromkeys(re.findall(r'[a-zA-Z0-9]{3,}', question)))
    result = ' '.join(s for s in important if s.lower() not in STOP)
    return result[:260] or question
