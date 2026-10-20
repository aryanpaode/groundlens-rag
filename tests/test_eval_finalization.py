import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from evaluation.ragas_runner import prepare_rows, parse_metrics, main as ragas_main
from evaluation.prepare_ragas import merge
from rag.observability import tracing_enabled, trace_stage, usage_from_gemini


def test_ragas_validation_and_no_api_dry_run(tmp_path, capsys):
    f = tmp_path / 'real.csv'
    with f.open('w', newline='', encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=['question','answer','contexts_json','ground_truth'])
        w.writeheader()
        w.writerow(dict(question='What is a pixel?', answer='A picture element.',
                        contexts_json=json.dumps(['A pixel is a picture element.']),
                        ground_truth='A picture element.'))
        w.writerow(dict(question='Unreviewed', answer='no score', contexts_json='[]', ground_truth=''))
    assert len(prepare_rows(f)) == 1
    ragas_main(['--dry-run', '--limit', '1', str(f)])
    output = capsys.readouterr().out
    assert 'DRY RUN ONLY' in output
    assert not (tmp_path / 'ragas_results.csv').exists()


def test_ragas_rejects_empty_context_and_bad_metric(tmp_path):
    f = tmp_path/'empty.csv'
    f.write_text('question,answer,contexts_json,ground_truth\na,b,[],c\n')
    with pytest.raises(ValueError, match='nonempty'):
        prepare_rows(f)
    with pytest.raises(ValueError, match='Metrics'):
        parse_metrics('madeup')


def test_merge_only_real_answers_and_matching_references(tmp_path):
    export = tmp_path/'export.csv'
    refs = tmp_path/'references.csv'
    output = tmp_path/'ready.csv'
    with export.open('w', newline='', encoding='utf-8') as handle:
        w=csv.DictWriter(handle, fieldnames=['question','answer','contexts_json','ground_truth'])
        w.writeheader()
        w.writerow(dict(question='What is a pixel?', answer='A picture element',
                        contexts_json='["A pixel is a picture element"]',ground_truth=''))
        w.writerow(dict(question='No matching reference?',answer='Unknown',
                        contexts_json='["X"]',ground_truth=''))
    refs.write_text('question,reference\nWhat is a pixel?,A picture element\n', encoding='utf-8')
    assert merge(export, refs, output) == (1, 2)
    assert prepare_rows(output)[0]['ground_truth'] == 'A picture element'


def test_langfuse_disabled_is_noop(monkeypatch):
    monkeypatch.setenv('GROUNDLENS_LANGFUSE_ENABLED','0')
    monkeypatch.delenv('LANGFUSE_PUBLIC_KEY',raising=False)
    monkeypatch.delenv('LANGFUSE_SECRET_KEY',raising=False)
    assert not tracing_enabled()
    with trace_stage('test') as span:
        assert span is None


def test_gemini_token_counts_are_only_reported_when_present():
    assert usage_from_gemini(SimpleNamespace()) == {}
    response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=17,candidates_token_count=6))
    assert usage_from_gemini(response) == {'input_tokens':17,'output_tokens':6}
