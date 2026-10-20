"""Fill references for ACTUAL GroundLens answers, without calling any model.

Matches normalized question text to the reviewed reference CSV. Only questions
with a real answer, retrieved contexts and a reviewed reference are written.
Do not invent references for unmatched questions.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def normalize(value: str) -> str:
    return ' '.join(value.casefold().split()).rstrip(' ?.\t')


def merge(export: Path, references: Path, output: Path) -> tuple[int, int]:
    from .ragas_runner import prepare_rows
    with references.open(newline='', encoding='utf-8-sig') as f:
        reference_rows = list(csv.DictReader(f))
    ref_map = {normalize(row['question']): row['reference'].strip()
               for row in reference_rows if row.get('question') and row.get('reference')}
    with export.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        gold = row.get('ground_truth', '').strip() or ref_map.get(normalize(row.get('question', '')), '')
        if gold and row.get('answer', '').strip() and row.get('contexts_json', '').strip():
            results.append(dict(question=row['question'], answer=row['answer'],
                                contexts_json=row['contexts_json'], ground_truth=gold))
    if not results:
        raise ValueError('No reviewed questions matched. Edit the exported CSV ground_truth cells manually.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'contexts_json', 'ground_truth'])
        writer.writeheader()
        writer.writerows(results)
    prepare_rows(output)  # fail fast if contexts are malformed
    return len(results), len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--references', type=Path,
                   default=Path('evaluation/reviewed_references.csv'))
    p.add_argument('--output', type=Path, default=Path('evaluation/ragas_ready.csv'))
    args = p.parse_args()
    used, total = merge(args.input, args.references, args.output)
    print(f'Matched {used}/{total} real responses to reviewed references: {args.output}')
    print('Review ground_truth manually before any RAGAS scoring.')

if __name__ == '__main__':
    main()
