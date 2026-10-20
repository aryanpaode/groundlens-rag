"""Opt-in Langfuse spans for GroundLens.

OFF by default. Source content, questions and answers are sent to a separate
service ONLY when the operator explicitly enables tracing and supplies keys.
No Langfuse import or network call occurs otherwise.
"""
from __future__ import annotations

from contextlib import contextmanager
import os


def tracing_enabled() -> bool:
    return (
        os.getenv('GROUNDLENS_LANGFUSE_ENABLED', '0').strip() == '1'
        and bool(os.getenv('LANGFUSE_PUBLIC_KEY', '').strip())
        and bool(os.getenv('LANGFUSE_SECRET_KEY', '').strip())
    )


@contextmanager
def trace_stage(name: str, *, input_data: dict | None = None,
                as_type: str = 'span', model: str | None = None):
    """Yield an active observation, or None when tracing is disabled.

    A configured-but-missing Langfuse installation raises a clear actionable
    error rather than silently claiming the trace succeeded.
    """
    if not tracing_enabled():
        yield None
        return
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError('Langfuse tracing enabled but package missing: pip install "langfuse>=4,<5"') from exc
    params = {'as_type': as_type, 'name': name}
    if input_data is not None:
        params['input'] = input_data
    if as_type == 'generation' and model:
        params['model'] = model
    with get_client().start_as_current_observation(**params) as span:
        yield span


def usage_from_gemini(response) -> dict:
    """Read Gemini token counts when provided; never fabricate missing ones."""
    usage = getattr(response, 'usage_metadata', None)
    if usage is None:
        return {}
    prompt = getattr(usage, 'prompt_token_count', None)
    completion = getattr(usage, 'candidates_token_count', None)
    result = {}
    if prompt is not None:
        result['input_tokens'] = int(prompt)
    if completion is not None:
        result['output_tokens'] = int(completion)
    return result
