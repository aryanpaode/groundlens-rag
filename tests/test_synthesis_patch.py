"""Regression tests for bounded multi-hop synthesis and answer rules."""
from rag.workflow import decompose_question
from rag.generation import INSUFFICIENT, SYSTEM_INSTRUCTION, make_prompt
from rag.retrieval_utils import SearchHit


def test_interconnected_question_triggers_three_bounded_search_hints():
    q = ("Explain how external forces, internal organizational problems, "
         "and resistance to change are interconnected in organizational transformation.")
    assert decompose_question(q) == [
        "external forces", "internal organizational problems", "resistance to change"
    ]


def test_unrelated_questions_do_not_gain_artificial_subqueries():
    assert decompose_question("What is the definition of organizational change?") == []


def test_excessively_long_topics_are_not_decomposed():
    q = "Explain how " + "long " * 20 + ", short ideas are related."
    assert decompose_question(q) == []


def test_generation_uses_claim_level_grounding_and_partial_coverage():
    assert "answer" in SYSTEM_INSTRUCTION.lower()
    assert "synthesis questions" in SYSTEM_INSTRUCTION
    assert "Do NOT reject" in SYSTEM_INSTRUCTION
    assert "Never invent causal links" in SYSTEM_INSTRUCTION
    assert "I couldn't find that information in the uploaded documents." == INSUFFICIENT
    hits = [SearchHit('mtm.pdf', 4, 1, 'External forces are technological changes.', 0.2)]
    prompt = make_prompt("How do internal and external forces relate?", hits)
    assert '[S1] Document: mtm.pdf | Page: 4' in prompt
    assert 'SOURCE EXCERPTS' in prompt
    assert 'synthesize well-supported relationships' in prompt
