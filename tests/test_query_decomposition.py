from rag.workflow import decompose_question, _merge_expanded
from rag.retrieval_utils import SearchHit


def test_four_way_comparison_decomposes_without_hardcoded_subject():
    qs = decompose_question(
        'Explain the difference between economic, psychological, social, and organizational resistance to change.'
    )
    assert qs == [
        'economic resistance to change',
        'psychological resistance to change',
        'social resistance to change',
        'organizational resistance to change',
    ]


def test_cross_document_question_decomposes():
    qs = decompose_question(
        'How do sampling and quantization differ from intensity transformation techniques?'
    )
    assert len(qs) == 2
    assert qs[0] == 'sampling and quantization'
    assert qs[1] == 'intensity transformation techniques'


def test_simple_question_does_not_expand():
    assert decompose_question('What are the economic reasons for resistance to change?') == []


def test_expansion_preserves_distinct_passages():
    def h(name, chunk, dist):
        return SearchHit(filename=name, page=chunk, chunk_number=1,
                         text=name + ' reasons', distance=dist, doc_id=name)
    p8,p9,p10=(h('MTM',8,.1),h('MTM',9,.15),h('MTM',10,.12))
    result=_merge_expanded([p10],[[p8],[p9],[p10]],['economic','psychological','organizational'],limit=5)
    assert {hit.page for hit in result} == {8,9,10}
