from pathlib import Path
from evaluation.benchmark import load_benchmark
from rag.pdf_processing import extract_pdf_chunks


def test_all_positive_benchmark_phrases_exist_in_original_pdf_chunks():
    root=Path(__file__).resolve().parents[1]
    rows=load_benchmark(root/'evaluation'/'benchmark_questions.csv')
    corpus={}
    for f in (root/'sample_documents').glob('*.pdf'):
        corpus[f.name]=extract_pdf_chunks(f.read_bytes())[0]
    assert len(corpus)==3
    assert len(rows)==22
    for row in rows:
        names=row['expected_documents'].split(';')
        pages=row['expected_pages'].split(';')
        phrases=row['expected_phrases'].split(';')
        for i, name in enumerate(names):
            if not name:
                continue
            assert any(c.page==int(pages[i]) and phrases[i].casefold() in c.text.casefold()
                       for c in corpus[name]), f"Missing expected fragment for benchmark {row['id']}"
