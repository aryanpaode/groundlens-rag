"""RAGAS evaluation of REAL GroundLens answers and verified reference text.

Input CSV columns: question, answer, contexts_json, ground_truth.
Start with --dry-run (no API calls), then --limit 1 to avoid Gemini RPM bursts.
No scores are fabricated or stored unless an evaluation actually completes.

Example:
    python -m evaluation.ragas_runner --dry-run evaluation/ragas_ready.csv
    python -m evaluation.ragas_runner --limit 1 evaluation/ragas_ready.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REQUIRED = ('question', 'answer', 'contexts_json', 'ground_truth')
AVAILABLE_METRICS = ('faithfulness', 'context_precision', 'context_recall', 'answer_relevancy')


def prepare_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'Evaluation input not found: {path}')
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(c not in reader.fieldnames for c in REQUIRED):
            raise ValueError(f'Expected CSV columns: {", ".join(REQUIRED)}')
        for line, row in enumerate(reader, 2):
            if not all(str(row.get(k, '')).strip() for k in REQUIRED):
                continue  # unlabeled exports are not scored
            try:
                contexts = json.loads(row['contexts_json'])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f'Line {line}: contexts_json is not valid JSON') from exc
            if not isinstance(contexts, list) or not contexts or not all(
                isinstance(context, str) and context.strip() for context in contexts
            ):
                raise ValueError(f'Line {line}: contexts_json must be a nonempty list of passages')
            rows.append({'question': row['question'].strip(),
                         'answer': row['answer'].strip(),
                         'contexts': contexts,
                         'ground_truth': row['ground_truth'].strip()})
    if not rows:
        raise ValueError('No completed rows. Export actual GroundLens answers and fill ground_truth.')
    return rows


def parse_metrics(value: str) -> tuple[str, ...]:
    requested = tuple(x.strip() for x in value.split(',') if x.strip())
    if not requested or any(x not in AVAILABLE_METRICS for x in requested):
        raise ValueError('Metrics must be a comma-separated subset of: ' + ', '.join(AVAILABLE_METRICS))
    if len(requested) != len(set(requested)):
        raise ValueError('Duplicate metric names are not allowed')
    return requested


def cli_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='CSV exported by GroundLens with filled ground_truth')
    parser.add_argument('--output', type=Path, default=Path('evaluation/ragas_results.csv'))
    parser.add_argument('--limit', type=int, default=1, help='Max rows for quota safety (default 1)')
    parser.add_argument('--metrics', default='faithfulness',
                        help='Comma-separated; start with faithfulness to reduce API usage')
    parser.add_argument('--model', help='Gemini judge model, defaults to GEMINI_MODEL in private .env')
    parser.add_argument('--dry-run', action='store_true', help='Validate input without calling API')
    return parser.parse_args(argv)


def main(argv=None):
    args = cli_args(argv)
    metrics = parse_metrics(args.metrics)
    if args.limit < 1:
        raise ValueError('--limit must be >= 1')
    rows = prepare_rows(args.input)
    rows = rows[:args.limit]
    print(f'Validated {len(rows)} real answer(s); metrics: {", ".join(metrics)}')
    if args.dry_run:
        print('DRY RUN ONLY — no Gemini API requests and no RAGAS score calculated.')
        return

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    api_key = (os.getenv('GEMINI_API_KEY') or '').strip()
    if not api_key or api_key.startswith('REPLACE_'):
        raise RuntimeError('GEMINI_API_KEY missing from private .env')
    os.environ.setdefault('GOOGLE_API_KEY', api_key)
    model = args.model or os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')

    # Keep RAGAS 0.3.x pinned: these are its documented legacy evaluation APIs.
    # All imports occur only after input validation, minimizing surprise downloads.
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from langchain_google_genai import ChatGoogleGenerativeAI
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key, temperature=0))
    available = {'faithfulness': faithfulness,
                 'context_precision': context_precision,
                 'context_recall': context_recall,
                 'answer_relevancy': answer_relevancy}
    needs_embeddings = 'answer_relevancy' in metrics
    embeddings = None
    if needs_embeddings:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001', google_api_key=api_key))
    print('Sending evaluation requests to Gemini; an API quota/429 error may occur.')
    report = evaluate(dataset=Dataset.from_list(rows),
                      metrics=[available[m] for m in metrics],
                      llm=judge, embeddings=embeddings,
                      run_config=RunConfig(max_workers=1, max_retries=1, timeout=240),
                      raise_exceptions=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_pandas().to_csv(args.output, index=False)
    print(report)
    print(f'MEASURED RAGAS results saved to {args.output} ({len(rows)} scored row(s)).')
    print('LLM-judge results are estimates; inspect individual rows and validate manually.')


if __name__ == '__main__':
    main()
