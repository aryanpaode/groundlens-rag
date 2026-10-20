"""Check optional Langfuse configuration; --send transmits a synthetic test only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--send', action='store_true', help='Send a trivial synthetic trace to Langfuse')
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    from rag.observability import tracing_enabled, trace_stage
    if not tracing_enabled():
        print('Langfuse OFF. Add private Langfuse keys and GROUNDLENS_LANGFUSE_ENABLED=1 to .env.')
        return
    from langfuse import get_client
    print('Langfuse installed and opt-in environment variables found.')
    if not args.send:
        print('No network requests sent. Use --send to transmit ONE synthetic test trace.')
        return
    with trace_stage('groundlens-synthetic-check', input_data={'question': 'health check, no user PDFs'}) as span:
        span.update(output={'message': 'synthetic trace produced'})
    get_client().flush()
    print('Trace export attempted. Check the Langfuse dashboard to confirm receipt.')


if __name__ == '__main__':
    main()
