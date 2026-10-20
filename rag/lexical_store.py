"""Local SQLite FTS5 BM25 companion to Pinecone's dense index.

No Pinecone schema migration is needed. This is *client-side hybrid fusion*,
not a native Pinecone sparse vector index. Existing PDFs need reuploading once
for local lexical backfill, and the local DB is workspace-specific.
"""
from __future__ import annotations
from pathlib import Path
import re
import sqlite3

from .retrieval_utils import SearchHit


class LexicalStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                doc_id UNINDEXED, filename UNINDEXED, page UNINDEXED,
                chunk_number UNINDEXED, text, tokenize='unicode61'
            )""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def has_document(self, doc_id: str) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM chunks WHERE doc_id=? LIMIT 1", (doc_id,)).fetchone() is not None

    def replace_document(self, doc_id: str, filename: str, chunks) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            db.executemany(
                "INSERT INTO chunks(doc_id,filename,page,chunk_number,text) VALUES (?,?,?,?,?)",
                [(doc_id, filename, c.page, c.chunk_number, c.text) for c in chunks],
            )

    def delete_document(self, doc_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    def clear(self) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM chunks")

    def get(self, doc_id: str, page: int, chunk_number: int) -> SearchHit | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT filename,page,chunk_number,text FROM chunks "
                "WHERE doc_id=? AND page=? AND chunk_number=? LIMIT 1",
                (doc_id, page, chunk_number),
            ).fetchone()
        if row is None:
            return None
        return SearchHit(row[0], int(row[1]), int(row[2]), row[3], 0.0, doc_id)

    def query(self, question: str, doc_ids: list[str], limit: int = 20) -> list[SearchHit]:
        if not doc_ids or limit < 1:
            return []
        # Safe tokenization: never directly interpolate a user-created FTS query.
        terms = list(dict.fromkeys(re.findall(r"[\w]{2,}", question.casefold(), flags=re.U)))[:16]
        if not terms:
            return []
        fts_query = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
        place = ",".join("?" for _ in doc_ids)
        sql = (f"SELECT filename,page,chunk_number,text,doc_id,bm25(chunks) "
               f"FROM chunks WHERE chunks MATCH ? AND doc_id IN ({place}) "
               "ORDER BY bm25(chunks) ASC LIMIT ?")
        with self.connect() as db:
            rows = db.execute(sql, [fts_query, *doc_ids, int(limit)]).fetchall()
        return [SearchHit(str(f), int(p), int(c), str(t), float(rank), str(d), "bm25")
                for f,p,c,t,d,rank in rows]


def reciprocal_rank_fusion(dense: list[SearchHit], lexical: list[SearchHit], limit: int,
                           rank_constant: int = 60) -> list[SearchHit]:
    """Combine independently ranked results without mixing incomparable scores."""
    pool = {}
    scores = {}
    for hits in (dense, lexical):
        for rank, hit in enumerate(hits, 1):
            key = (hit.doc_id, hit.page, hit.chunk_number)
            pool.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
    ranked = sorted(pool, key=lambda k: (-scores[k], k))[:max(0, limit)]
    return [SearchHit(pool[k].filename, pool[k].page, pool[k].chunk_number,
                      pool[k].text, 1.0 - scores[k], pool[k].doc_id, "rrf")
            for k in ranked]
