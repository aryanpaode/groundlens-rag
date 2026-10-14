"""Runs against real Chroma on the user's installed project; skipped if unavailable."""
import pytest
import fitz

chromadb = pytest.importorskip('chromadb')
from rag.vector_store import LocalVectorStore


class TinyFakeEmbedder:
    """Deterministic tiny vectors, replacing neural embeddings in this test only."""
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors=[]
        for text in texts:
            words=text.lower()
            vectors.append([float('alpha' in words), float('beta' in words),
                            float('registration' in words), 0.1])
        return vectors


def make_pdf(text):
    doc=fitz.open()
    page=doc.new_page()
    page.insert_text((65,85),text)
    content=doc.tobytes()
    doc.close()
    return content


def test_chroma_add_search_filter_delete(tmp_path):
    store=LocalVectorStore(tmp_path/'chroma')
    embedder=TinyFakeEmbedder()
    a=make_pdf('Alpha registration fee is 100 credits.')
    b=make_pdf('Beta registration fee is 150 credits.')
    assert store.add_pdf('a.pdf',a,embedder)[0]=='indexed'
    assert store.add_pdf('a.pdf',a,embedder)[0]=='already_indexed'
    assert store.add_pdf('b.pdf',b,embedder)[0]=='indexed'
    assert store.count == 2
    ids={d['filename']:d['doc_id'] for d in store.list_documents()}
    filtered=store.search('beta registration?',embedder,n_results=2,doc_ids=[ids['b.pdf']])
    assert len(filtered)==1
    assert filtered[0].filename=='b.pdf'
    result=store.search('registration fee',embedder,n_results=2,balanced=True)
    assert {h.filename for h in result}=={'a.pdf','b.pdf'}
    store.delete_document(ids['a.pdf'])
    assert {d['filename'] for d in store.list_documents()} == {'b.pdf'}
    store.clear_all()
    assert store.count == 0
