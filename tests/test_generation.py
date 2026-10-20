from collections import namedtuple
from rag.generation import make_prompt, cited_source_numbers, invalid_source_numbers, INSUFFICIENT

SearchHit = namedtuple('SearchHit', 'filename page chunk_number text distance')


def test_prompt_has_correct_evidence_label():
    hits = [SearchHit('a.pdf', 2, 1, 'Registration closes Friday.', 0.11)]
    prompt = make_prompt('When does registration close?', hits)
    assert '[S1] Document: a.pdf | Page: 2' in prompt
    assert 'Registration closes Friday.' in prompt
    assert 'When does registration close?' in prompt


def test_previous_answers_are_not_source_evidence():
    history = [{'question': 'Who won?', 'answer': 'Earlier model said Anita.'}]
    prompt = make_prompt('And when?', [], history)
    assert 'NOT evidence' in prompt
    assert 'SOURCE EXCERPTS (the only evidence):\nNone' in prompt


def test_citation_id_audit_is_syntactic_only():
    answer = 'First claim [S1]. Another [S9].'
    assert cited_source_numbers(answer) == [1,9]
    assert invalid_source_numbers(answer, source_count=3) == [9]
    assert invalid_source_numbers('Nothing known.', source_count=3) == []
    assert 'couldn\'t find' in INSUFFICIENT
