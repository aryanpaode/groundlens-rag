"""Uses a real LangGraph compiler but fake retrieval/generation (no API charges)."""
import pytest

pytest.importorskip('langgraph')
from rag.retrieval_utils import SearchHit
from rag.workflow import build_workflow


class FakeStore:
    def __init__(self):
        self.queries=[]
    def search(self,query,embedder,n_results,doc_ids,balanced):
        self.queries.append(query)
        labels={
            'economic':'job insecurity',
            'psychological':'ego defensiveness',
            'social':'peer pressure',
            'organizational':'rigid structure',
        }
        category=next((k for k in labels if query.lower().startswith(k)),None)
        if category:
            page=8 if category=='economic' else 9 if category in ('psychological','social') else 10
            return [SearchHit('mtm.pdf',page,len(self.queries),labels[category],0.3,doc_id='mtm')]
        if 'nonsense' in query:
            return []
        return [SearchHit('mtm.pdf',3,1,'Overview of resistance',0.1,doc_id='mtm')]


def fake_generator(key,model,question,hits,history):
    return f"Evidence: {len(hits)} sources [S1]"


def test_langgraph_expansion_and_grounded_generation():
    st=FakeStore()
    graph=build_workflow(st,object(),gemini_api_key='fake',model_name='fake',generate_fn=fake_generator)
    result=graph.invoke({'question':'Explain the difference between economic, psychological, social, and organizational resistance to change.',
                         'doc_ids':['mtm'],'top_k':5,'balanced':False,'history':[]})
    assert result['graph_steps']==['prepare','retrieve','expand','generate','audit']
    assert len(st.queries)==5  # initial plus 4 subtopic searches
    assert {h.page for h in result['hits']} >= {8,9,10}
    assert result['invalid_citations']==[]
    assert 'Evidence:' in result['answer']


def test_langgraph_empty_retrieval_abstains_without_gemini():
    st=FakeStore()
    graph=build_workflow(st,object(),gemini_api_key='fake',model_name='fake',generate_fn=lambda *args: (_ for _ in ()).throw(AssertionError('must not call Gemini')))
    result=graph.invoke({'question':'nonsense question','doc_ids':['mtm'],'top_k':5,'balanced':False,'history':[]})
    assert result['graph_steps']==['prepare','retrieve','abstain','audit']
    assert "couldn't find" in result['answer']
    assert result['generation_seconds']==0
