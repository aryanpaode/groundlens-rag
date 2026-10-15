from types import SimpleNamespace
from evaluation.benchmark import evaluate_retrieval


class DummyStore:
    def list_documents(self):
        return [{'filename': 'alpha.pdf'}, {'filename': 'beta.pdf'}]

    def search(self, question, embedder, n_results=5, balanced=False):
        if 'compare' in question:
            return [SimpleNamespace(filename='alpha.pdf',page=1,text='The fee is 100.',distance=.2),
                    SimpleNamespace(filename='beta.pdf',page=1,text='The fee is 150.',distance=.4)]
        return [SimpleNamespace(filename='alpha.pdf',page=1,text='The fee is 100.',distance=.2)]


def test_benchmark_scores_sources_and_skips_missing():
    rows=[
        {'id':'1','category':'fact','question':'fee?', 'expected_documents':'alpha.pdf',
         'expected_pages':'1','expected_phrases':'fee is 100'},
        {'id':'2','category':'compare','question':'compare fees?', 'expected_documents':'alpha.pdf;beta.pdf',
         'expected_pages':'1;1','expected_phrases':'fee is 100;fee is 150'},
        {'id':'3','category':'fact','question':'other?', 'expected_documents':'missing.pdf',
         'expected_pages':'1','expected_phrases':'none'},
        {'id':'4','category':'missing','question':'password?', 'expected_documents':'',
         'expected_pages':'','expected_phrases':''},
    ]
    results, metrics = evaluate_retrieval(DummyStore(), None, rows, 5)
    assert [r['status'] for r in results] == ['PASS','PASS',
        'SKIPPED: required PDF not indexed','MANUAL: answer/absence evaluation']
    assert metrics['evaluated'] == 2
    assert metrics['retrieval_hit_rate'] == 1.0
    assert abs(metrics['mean_reciprocal_rank'] - .75)<1e-9
