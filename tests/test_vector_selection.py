from rag.retrieval_utils import SearchHit, choose_hits


def hit(document, distance):
    return SearchHit(document, 1, 1, f'{document} content', distance, doc_id=document)


def test_relevant_mode_keeps_nearest():
    hits=[hit('a',0.1),hit('a',0.2),hit('b',0.3)]
    assert [h.doc_id for h in choose_hits(hits,2)] == ['a','a']


def test_balanced_mode_includes_other_document():
    hits=[hit('a',0.1),hit('a',0.2),hit('b',0.3)]
    assert {h.doc_id for h in choose_hits(hits,2,balanced=True)} == {'a','b'}


def test_empty_selection():
    assert choose_hits([hit('a',0.1)],0) == []
