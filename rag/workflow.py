"""LangGraph StateGraph for GroundLens document question answering.

START -> prepare -> retrieve -> conditional routing:
    * comparison hints -> expand -> generate or abstain -> audit -> END
    * no hints, hits   -> generate -> audit -> END
    * no hints or hits -> abstain -> audit -> END

Expansion performs bounded, filtered Pinecone searches for individual topics.
After expansion, the graph checks whether any evidence remains before generation.
"""
from __future__ import annotations

import re
from time import perf_counter
from typing import TypedDict

from .generation import INSUFFICIENT, answer_with_gemini, invalid_source_numbers
from .retrieval_utils import SearchHit


class QAState(TypedDict, total=False):
    question: str
    doc_ids: list[str]
    top_k: int
    balanced: bool
    history: list[dict]
    retrieval_query: str
    subqueries: list[str]
    hits: list[SearchHit]
    answer: str
    retrieval_seconds: float
    generation_seconds: float
    invalid_citations: list[int]
    graph_steps: list[str]


STOPWORDS = {"explain", "describe", "compare", "between", "about", "their", "which",
             "that", "what", "are", "the", "and", "from", "with", "this", "how",
             "these", "used", "when", "does", "for", "into", "is", "different", "difference"}


def _keywords(s: str) -> set[str]:
    return {word for word in re.findall(r"[A-Za-z]{3,}", s.casefold()) if word not in STOPWORDS}


def decompose_question(question: str) -> list[str]:
    """Break clear comparative/enumerated questions into bounded search hints.

    This is deliberately a transparent *heuristic*, not an LLM planner.
    It may return [] for questions that should be searched as a single whole.
    """
    clean = question.strip().rstrip("?. ")
    m = re.search(r"\b(?:how (?:do|does))\s+(.+?)\s+differ(?:s)?\s+from\s+(.+)$", clean, re.I)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    m = re.search(r"\b(?:compare|contrast)\s+(.+?)\s+(?:with|versus|vs\.?|and)\s+(.+)$", clean, re.I)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]

    # E.g. 'differences between economic, psychological, social and
    # organizational resistance to change' -> one request per category.
    m = re.search(r"\b(?:differences?|comparison)\s+(?:between|among)\s+(.+)$", clean, re.I)
    if not m:
        return []
    clause = m.group(1).strip()
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", clause) if p.strip()]
    if not 2 <= len(parts) <= 5:
        return []
    last_words = parts[-1].split()
    # If only the last item names the shared topic, carry it into each hint.
    # Example: 'organizational resistance to change' -> shared suffix.
    suffix = " ".join(last_words[1:]) if len(last_words) >= 3 else ""
    queries = [(p + (" " + suffix if suffix and len(p.split()) <= 2 else "")).strip()
               for p in parts]
    return list(dict.fromkeys(queries))[:5]


def _best_hint_hit(hits: list[SearchHit], hint: str) -> SearchHit | None:
    if not hits:
        return None
    key = _keywords(hint)
    # Prefer a source explicitly containing more of the requested terminology.
    # Use cosine distance as a tiebreaker, not a certainty score.
    return sorted(hits, key=lambda h: (-len(_keywords(h.text) & key), h.distance))[0]


def _merge_expanded(primary: list[SearchHit], per_hint: list[list[SearchHit]],
                    hints: list[str], limit: int) -> list[SearchHit]:
    unique = {}
    selected = []
    def token(h: SearchHit):
        return h.doc_id, h.page, h.chunk_number

    # Guarantee a candidate per hint where possible, before allocating extra.
    for query, matches in zip(hints, per_hint):
        best = _best_hint_hit(matches, query)
        if best is not None and token(best) not in unique:
            unique[token(best)] = best
            selected.append(best)
    pool = sorted(primary + [h for hits in per_hint for h in hits], key=lambda h: h.distance)
    for hit in pool:
        if len(selected) >= limit:
            break
        if token(hit) not in unique:
            unique[token(hit)] = hit
            selected.append(hit)
    return sorted(selected[:limit], key=lambda h: h.distance)


def build_workflow(store, embedder, *, gemini_api_key: str,
                   model_name: str, generate_fn=answer_with_gemini):
    """Compile the workflow; no cloud calls until a user invokes it."""
    from langgraph.graph import StateGraph, START, END

    def prepare(state: QAState) -> dict:
        question = state["question"].strip()
        previous = state.get("history", [])
        retrieval_query = question
        if (previous and len(question.split()) <= 12
                and re.search(r"\b(it|they|them|that|those|their|he|she)\b|\bwhat about\b",
                              question, flags=re.I)):
            retrieval_query = str(previous[-1].get("question", ""))[:260] + " " + question
        return {"retrieval_query": retrieval_query,
                "subqueries": decompose_question(retrieval_query),
                "graph_steps": ["prepare"]}

    def retrieve(state: QAState) -> dict:
        started = perf_counter()
        hits = store.search(state["retrieval_query"], embedder,
                            n_results=state["top_k"], doc_ids=state.get("doc_ids"),
                            balanced=state.get("balanced", False))
        return {"hits": hits, "retrieval_seconds": perf_counter() - started,
                "graph_steps": state.get("graph_steps", []) + ["retrieve"]}

    def route(state: QAState) -> str:
        if state.get("subqueries"):
            return "expand"
        return "generate" if state.get("hits") else "abstain"

    def expand(state: QAState) -> dict:
        started = perf_counter()
        hints = state.get("subqueries", [])
        partials = [store.search(hint, embedder, n_results=min(4, max(2, state["top_k"])),
                                 doc_ids=state.get("doc_ids"), balanced=False)
                    for hint in hints]
        # Cap extra evidence to avoid a huge Gemini context or runaway requests.
        limit = min(8, max(state["top_k"], len(hints)))
        merged = _merge_expanded(state.get("hits", []), partials, hints, limit)
        return {"hits": merged,
                "retrieval_seconds": state.get("retrieval_seconds", 0) + perf_counter() - started,
                "graph_steps": state.get("graph_steps", []) + ["expand"]}

    def route_after_expand(state: QAState) -> str:
        return "generate" if state.get("hits") else "abstain"

    def abstain(state: QAState) -> dict:
        return {"answer": INSUFFICIENT, "generation_seconds": 0.0,
                "graph_steps": state.get("graph_steps", []) + ["abstain"]}

    def generate(state: QAState) -> dict:
        started = perf_counter()
        answer = generate_fn(gemini_api_key, model_name, state["question"],
                             state.get("hits", []), state.get("history", []))
        return {"answer": answer, "generation_seconds": perf_counter() - started,
                "graph_steps": state.get("graph_steps", []) + ["generate"]}

    def audit(state: QAState) -> dict:
        return {"invalid_citations": invalid_source_numbers(state.get("answer", ""), len(state.get("hits", []))),
                "graph_steps": state.get("graph_steps", []) + ["audit"]}

    graph = StateGraph(QAState)
    graph.add_node("prepare", prepare)
    graph.add_node("retrieve", retrieve)
    graph.add_node("expand", expand)
    graph.add_node("abstain", abstain)
    graph.add_node("generate", generate)
    graph.add_node("audit", audit)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "retrieve")
    graph.add_conditional_edges("retrieve", route,
                                {"expand": "expand", "generate": "generate", "abstain": "abstain"})
    graph.add_conditional_edges("expand", route_after_expand,
                                {"generate": "generate", "abstain": "abstain"})
    graph.add_edge("abstain", "audit")
    graph.add_edge("generate", "audit")
    graph.add_edge("audit", END)
    return graph.compile()
