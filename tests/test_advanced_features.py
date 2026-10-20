"""Self-contained tests for new features; no Pinecone/Gemini credentials."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt

import fitz
import numpy as np
import pytest

from rag.advanced_config import IngestionOptions, RetrievalOptions
from rag.corrective import weak_evidence, corrected_query
from rag.document_ingestion import extract_document, semantic_split
from rag.lexical_store import LexicalStore, reciprocal_rank_fusion
from rag.raptor import build_raptor_nodes
from rag.reranking import rerank
from rag.retrieval_utils import SearchHit
from rag.pinecone_store import PineconeVectorStore, DIMENSIONS


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        vectors=[]
        for t in texts:
            v=np.zeros(DIMENSIONS, dtype=float)
            for word in t.lower().split():
                v[hash(word) % DIMENSIONS] += 1.0
            v /= max(np.linalg.norm(v),1.0)
            vectors.append(v)
        return np.array(vectors)


class Index:
    def __init__(self): self.vectors={}
    def upsert(self, vectors, namespace):
        for row in vectors:
            self.vectors[(namespace,row['id'])]=row
    def query(self, vector, top_k, include_metadata, include_values, filter, namespace):
        from types import SimpleNamespace
        targets=filter['doc_id']
        allowed=targets['$in'] if isinstance(targets, dict) else [targets]
        selected=[]
        for (ns,_),row in self.vectors.items():
            if ns==namespace and row['metadata']['doc_id'] in allowed:
                selected.append(SimpleNamespace(metadata=row['metadata'],
                                  score=sum(x*y for x,y in zip(vector,row['values']))))
        selected.sort(key=lambda x:-x.score)
        return SimpleNamespace(matches=selected[:top_k])
    def list(self, prefix, namespace):
        yield [i for (ns,i) in self.vectors if ns==namespace and i.startswith(prefix)]
    def delete(self, ids=None, namespace='', delete_all=False):
        if delete_all:
            self.vectors={key:v for key,v in self.vectors.items() if key[0]!=namespace}
        else:
            for i in ids: self.vectors.pop((namespace,i),None)


def pdf_bytes(sentence: str) -> bytes:
    doc=fitz.open(); page=doc.new_page()
    page.insert_text((72, 72), sentence)
    result=doc.tobytes();doc.close()
    return result


def fake_store(tmp_path, index=None):
    return PineconeVectorStore('fake','groundlens-rag-384','advanced-tests',
                               tmp_path/'manifest.json',client=object(),index=index or Index())


def hit(t, page=1, doc='a', dist=.1):
    return SearchHit('a.pdf',page,1,t,dist,doc)


def test_ingest_options_reject_invalid_parser():
    with pytest.raises(ValueError): IngestionOptions(parser='unapproved')


def test_local_bm25_respects_document_scope_and_delete(tmp_path):
    lexical=LexicalStore(tmp_path/'lex.sqlite3')
    lexical.replace_document('a','a.pdf',[hit('An alpha keyword appears in this passage')])
    lexical.replace_document('b','b.pdf',[hit('Another alpha section about sampling')])
    assert len(lexical.query('alpha', ['a']))==1
    assert all(h.doc_id=='b' for h in lexical.query('alpha',['b']))
    lexical.delete_document('b')
    assert lexical.query('alpha',['b'])==[]


def test_fts_unsafe_syntax_escapes_to_tokens(tmp_path):
    lexical=LexicalStore(tmp_path/'fts.db')
    lexical.replace_document('a','a.pdf',[hit('alpha beta gamma')])
    assert lexical.query('alpha" OR NOT * (', ['a'])


def test_rrf_rank_independent_of_raw_cosine_and_bm25():
    dense=[hit('alpha',dist=.03), hit('beta',page=2,dist=.05)]
    lexical=[hit('beta',page=2,dist=-4.3),hit('alpha',dist=-3.0)]
    fusion=reciprocal_rank_fusion(dense,lexical,5)
    assert len(fusion)==2
    assert {h.page for h in fusion}=={1,2}
    assert all(h.score_kind=='rrf' for h in fusion)


def test_hybrid_pipeline_retains_pdf_answering_and_no_index_rebuild(tmp_path):
    emb=FakeEmbedder(); idx=Index(); store=fake_store(tmp_path,idx)
    content=pdf_bytes('Gamma correction changes image brightness in the lecture.')
    status,chunks,pages=store.add_pdf('gamma.pdf',content,emb)
    assert (status,chunks,pages)==('indexed',1,1)
    dense=store.search('gamma correction',emb,n_results=3)
    hybrid=store.search('gamma correction',emb,n_results=3,hybrid=True)
    assert dense and hybrid and hybrid[0].text==dense[0].text
    assert hybrid[0].score_kind=='rrf'
    assert store.lexical_coverage==1
    doc_id=store.list_documents()[0]['doc_id']
    store.delete_document(doc_id)
    assert store.lexical.query('gamma',[doc_id])==[]


def test_backfill_old_manifest_without_reupload(tmp_path):
    emb=FakeEmbedder(); idx=Index(); store=fake_store(tmp_path,idx)
    pdf=pdf_bytes('Gamma enhancement uses transformations.')
    store.add_pdf('gamma.pdf',pdf,emb)
    store.lexical.clear()
    assert store.add_pdf('gamma.pdf',pdf,emb)[0]=='lexical_backfilled'
    assert store.count==1 and store.lexical_coverage==1


def test_semantic_processing_retains_page_provenance():
    emb=FakeEmbedder()
    pdf=pdf_bytes('Alpha changes images. Beta adjusts brightness. Gamma enhances contrast.')
    chunks,pages=extract_document(pdf, filename='sample.pdf', semantic_chunks=True,
                                  embedding_model=emb)
    assert pages==1 and chunks and all(c.page==1 for c in chunks)


def test_corrective_heuristic_flags_low_overlap_without_probability():
    assert weak_evidence('Explain gamma correction', [hit('A passage about biodiversity')])
    assert not weak_evidence('Explain gamma correction',[hit('Gamma correction changes brightness')])
    assert 'gamma' in corrected_query('Explain gamma correction').lower()


def test_cross_encoder_is_injected_and_applied_to_real_excerpts():
    class Cross:
        def predict(self,pairs,**kwargs):
            assert pairs[0][0]=='Explain alpha'
            return [0.2, 0.9]
    result=rerank('Explain alpha',[hit('first'),hit('second',page=2)],1,model=Cross())
    assert len(result)==1 and result[0].page==2 and result[0].score_kind=='cross-encoder'


def test_recursive_summary_nodes_trace_to_original_chunks():
    emb=FakeEmbedder()
    from rag.pdf_processing import PageChunk
    chunks=[PageChunk((i//4)+1,(i%4)+1, f'topic {i} lecture notes and details')
            for i in range(14)]
    vectors=emb.encode([c.text for c in chunks])
    summaries=[]
    def summarize(texts,level):
        summaries.append((len(texts),level))
        return 'Document-level summary about lecture notes'
    nodes=build_raptor_nodes('abcd',chunks,vectors,emb,summarize,max_levels=2)
    assert len(nodes)>=2
    assert all(node.child_leaf_ids and node.level in (1,2) for node in nodes)
    assert any('1:1' in node.child_leaf_ids for node in nodes)


def test_advanced_indexing_with_mock_llm_keeps_leaf_citations(tmp_path,monkeypatch):
    # No external requests; summary generation and context are stubbed at the
    # module boundary. The store must only surface actual PDF pages as evidence.
    import rag.pinecone_store as mod
    from rag.pdf_processing import PageChunk
    monkeypatch.setattr(mod,'extract_document', lambda *args, **kwargs:
                        ([PageChunk(1,i+1, f'Gamma brightness discussion part {i}')
                          for i in range(14)],1))
    monkeypatch.setattr(mod,'chunk_context', lambda *args, **kwargs:'Section about gamma')
    monkeypatch.setattr(mod,'gemini_summarizer', lambda *args, **kwargs:
                        (lambda excerpts,level:'Gamma chapter overview'))
    emb=FakeEmbedder(); store=fake_store(tmp_path)
    result=store.add_pdf('gamma.pdf',pdf_bytes('Gamma page'),emb,
                         options=IngestionOptions(raptor=True,contextual_embeddings=True),
                         gemini_api_key='mock',model_name='mock')
    assert result[1]==14
    assert store.list_documents()[0]['summaries']>0
    docs=store.list_documents()
    hits=store.search('gamma brightness',emb,n_results=5,use_hierarchy=True,
                      doc_ids=[docs[0]['doc_id']])
    assert hits and all(h.page>=1 for h in hits)
    assert all('Gamma brightness discussion' in h.text for h in hits)
