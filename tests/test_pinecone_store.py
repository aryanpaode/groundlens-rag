"""Tests use a fake Pinecone index (no cloud services or secrets)."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from types import SimpleNamespace

import fitz
import pytest

from rag.pinecone_store import DIMENSIONS, PineconeVectorStore


class TinyEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors=[]
        for text in texts:
            s=text.lower()
            vec=[0.0]*DIMENSIONS
            vec[0] = 1.0 if 'alpha' in s else 0.0
            vec[1] = 1.0 if 'beta' in s else 0.0
            vec[2] = 1.0 if 'fee' in s else 0.0
            vec[3] = .05
            mag=sqrt(sum(v*v for v in vec))
            vectors.append([v/mag for v in vec])
        return vectors


class FakeIndex:
    def __init__(self):
        self.db={}
        self.calls=[]
    def upsert(self, vectors, namespace):
        self.calls.append(('upsert',namespace,len(vectors)))
        self.db.setdefault(namespace,{})
        for v in vectors:
            self.db[namespace][v['id']]=v
    def query(self, vector, top_k, include_metadata, include_values, filter, namespace):
        self.calls.append(('query',namespace,top_k,filter))
        results=[]
        for v in self.db.get(namespace,{}).values():
            meta=v['metadata']
            allowed=filter['doc_id']
            if isinstance(allowed,dict):
                allowed=allowed['$in']
            else:
                allowed=[allowed]
            if meta['doc_id'] not in allowed:
                continue
            score=sum(x*y for x,y in zip(vector,v['values']))
            results.append(SimpleNamespace(score=score,metadata=meta))
        results.sort(key=lambda x:-x.score)
        return SimpleNamespace(matches=results[:top_k])
    def list(self, prefix,namespace):
        ids=[key for key in self.db.get(namespace,{}) if key.startswith(prefix)]
        for i in range(0,len(ids),100):
            yield ids[i:i+100]
    def delete(self, ids=None, delete_all=False, namespace=''):
        self.calls.append(('delete',namespace,delete_all))
        if delete_all:
            self.db[namespace]={}
        else:
            for item in ids:
                self.db.get(namespace,{}).pop(item,None)


def pdf(text):
    d=fitz.open(); p=d.new_page(); p.insert_text((70,90),text)
    data=d.tobytes(); d.close(); return data


def store(tmp_path, idx):
    return PineconeVectorStore('fake','groundlens-rag-384','test-namespace',
                               tmp_path/'manifest.json',index=idx,client=object())


def test_insert_query_filter_duplicate_delete_and_namespace(tmp_path):
    fake=FakeIndex(); em=TinyEmbeddingModel()
    s=store(tmp_path,fake)
    a=pdf('Alpha registration fee is 100 credits.')
    b=pdf('Beta registration fee is 150 credits.')
    assert s.add_pdf('a.pdf',a,em)==('indexed',1,1)
    assert s.add_pdf('a.pdf',a,em)[0]=='already_indexed'
    assert s.add_pdf('b.pdf',b,em)[0]=='indexed'
    assert s.count==2
    docs={d['filename']:d for d in s.list_documents()}
    selected=s.search('beta fee',em,n_results=2,doc_ids=[docs['b.pdf']['doc_id']])
    assert len(selected)==1 and selected[0].filename=='b.pdf'
    assert selected[0].page==1
    assert fake.calls[-1][1]=='test-namespace'
    all_hits=s.search('registration fee',em,n_results=2,balanced=True)
    assert {hit.filename for hit in all_hits}=={'a.pdf','b.pdf'}
    s.delete_document(docs['a.pdf']['doc_id'])
    assert [d['filename'] for d in s.list_documents()]==['b.pdf']
    reloaded=store(tmp_path,fake)
    assert reloaded.count==1
    fake.db['other-namespace']={'untouched':{'metadata':{'text':'do not delete'}}}
    s.clear_all()
    assert s.count==0
    assert 'untouched' in fake.db['other-namespace']
    assert fake.db['test-namespace']=={}


def test_dimension_safety(tmp_path):
    idx=FakeIndex()
    class WrongEmb:
        def encode(self, texts, **kwargs):
            return [[1,0,0] for _ in texts]
    s=store(tmp_path,idx)
    with pytest.raises(ValueError,match='dimensions'):
        s.add_pdf('a.pdf',pdf('A valid text.'),WrongEmb())


def test_unknown_doc_id_rejected(tmp_path):
    s=store(tmp_path,FakeIndex())
    with pytest.raises(ValueError,match='Unknown'):
        s.delete_document('f'*64)


def test_missing_key_rejected(tmp_path):
    with pytest.raises(ValueError,match='PINECONE_API_KEY'):
        PineconeVectorStore('', 'groundlens-rag-384','test-namespace',tmp_path/'manifest.json')
