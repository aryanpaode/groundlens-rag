"""Regression checks for summary-child ranking and evidence provenance."""
from __future__ import annotations

from rag.pinecone_store import PineconeVectorStore
from rag.pdf_processing import PageChunk
from tests.test_advanced_features import FakeEmbedder, Index, DIMENSIONS


def test_unrelated_summary_children_cannot_displace_relevant_leaves(tmp_path):
    embedding = FakeEmbedder()
    index = Index()
    store = PineconeVectorStore('fake', 'groundlens-rag-384', 'raptor-regression',
                                tmp_path / 'manifest.json', client=object(), index=index)
    doc_id = 'a' * 64
    question = 'external internal organizational resistance change'
    data = [
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'the garden music theatre coffee and art',
        'external internal organizational resistance change',
        'external forces require organizational change',
        'internal organizational problems trigger change',
        'resistance to organizational change from employees',
    ]
    chunks = [PageChunk(i + 1, 1, text) for i, text in enumerate(data)]
    store.lexical.replace_document(doc_id, 'test.pdf', chunks)
    store._documents[doc_id] = dict(doc_id=doc_id, filename='test.pdf', page_count=12,
                                   chunks=len(chunks), summaries=1, parser='pymupdf')
    store._save_manifest()
    vectors = embedding.encode(data)
    records = []
    for i, (text, vector) in enumerate(zip(data, vectors), 1):
        records.append(dict(id=f'{doc_id}:{i}:1', values=vector.tolist(), metadata={
            'doc_id': doc_id, 'filename': 'test.pdf', 'page': i,
            'chunk_number': 1, 'text': text, 'record_type': 'leaf',
        }))
    # A summary has extremely high query similarity but all its listed children
    # are completely off-topic. The pre-fix code inherits its 1.0 similarity
    # for all eight child pages and incorrectly puts these passages in top 5.
    records.append(dict(id=f'{doc_id}:R1:1',
                        values=embedding.encode([question])[0].tolist(), metadata={
                            'doc_id': doc_id, 'filename': 'test.pdf', 'page': 0,
                            'chunk_number': 0, 'text': question,
                            'record_type': 'summary',
                            'child_leaf_ids': ';'.join(f'{i}:1' for i in range(1, 9)),
                        }))
    index.upsert(records, namespace='raptor-regression')
    result = store.search(question, embedding, n_results=5, use_hierarchy=True)
    assert result
    assert result[0].page == 9, result
    assert {h.page for h in result[:3]} >= {9, 10} or result[0].page == 9
    assert all(h.page > 8 for h in result[:3]), result
    assert all(h.page != 0 for h in result)
    without_hierarchy = store.search(question, embedding, n_results=5, use_hierarchy=False)
    assert len(without_hierarchy) == 5
    assert without_hierarchy[0].page == 9


def test_hierarchy_results_are_original_page_passages(tmp_path, monkeypatch):
    # Re-run existing integration coverage as a regression with precise source IDs.
    from tests.test_advanced_features import test_advanced_indexing_with_mock_llm_keeps_leaf_citations
    test_advanced_indexing_with_mock_llm_keeps_leaf_citations(tmp_path, monkeypatch)
