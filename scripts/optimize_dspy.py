"""Offline DSPy MIPROv2 optimization using *labelled* training examples.

Input CSV: question,evidence,reference
Each evidence field should include correct [S1] source excerpts.
Optimization makes additional paid Gemini calls. It never touches Pinecone.

Example:
  python scripts/optimize_dspy.py --csv evaluation/dspy_training_template.csv
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_examples(csv_path: Path):
    import dspy
    rows=[]
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        reader=csv.DictReader(handle)
        for row in reader:
            if not all(row.get(key, '').strip() for key in ('question','evidence','reference')):
                continue
            rows.append(dspy.Example(question=row['question'], evidence=row['evidence'],
                                     answer=row['reference']).with_inputs('question','evidence'))
    if len(rows) < 8:
        raise ValueError('Provide at least 8 reviewed labelled examples; the template is not enough.')
    return rows


def answer_metric(example, pred, trace=None):
    """Conservative word overlap heuristic; supplement with manual review."""
    import re
    wanted=set(re.findall(r'[a-z0-9]+', str(example.answer).lower()))
    got=set(re.findall(r'[a-z0-9]+', str(pred.answer).lower()))
    if not wanted:
        return 0.0
    return len(wanted & got)/len(wanted)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'dspy_program.json')
    parser.add_argument('--model', default=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
    args=parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    key=os.getenv('GEMINI_API_KEY', '').strip()
    if not key or key.startswith('REPLACE_'):
        raise RuntimeError('Private GEMINI_API_KEY required to optimize')
    import dspy
    from dspy.teleprompt import MIPROv2
    from rag.dspy_program import new_program
    dspy.configure(lm=dspy.LM('gemini/'+args.model, api_key=key, temperature=0.1))
    samples=load_examples(args.csv)
    optimizer=MIPROv2(metric=answer_metric, auto='light', num_threads=1)
    optimized=optimizer.compile(new_program(), trainset=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(args.output))
    print(f'Optimized DSPy module saved: {args.output}')
    print('Validate on a separate held-out dataset; overlap metric is not a correctness guarantee.')

if __name__ == '__main__':
    main()
