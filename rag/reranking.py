"""Optional CPU cross-encoder for reranking candidate source passages."""
from __future__ import annotations
from functools import lru_cache
from .retrieval_utils import SearchHit


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')


def rerank(question: str, hits: list[SearchHit], limit: int, model=None) -> list[SearchHit]:
    if not hits:
        return []
    model = model if model is not None else get_cross_encoder()
    pairs = [(question, hit.text[:2500]) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)
    if len(scores) != len(hits):
        raise RuntimeError('CrossEncoder returned unexpected number of scores')
    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)[:limit]
    return [SearchHit(h.filename, h.page, h.chunk_number, h.text,
                      -float(score), h.doc_id, 'cross-encoder') for h, score in ranked]
