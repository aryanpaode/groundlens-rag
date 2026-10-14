"""Small one-attempt Gemini test with separate API timing. Never prints API keys."""
from pathlib import Path
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")
key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
if not key or key.startswith("REPLACE_"):
    raise SystemExit("Set GEMINI_API_KEY in .env before testing.")
print("Testing Gemini model:", model)
start = time.perf_counter()
try:
    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: OK",
        config=types.GenerateContentConfig(max_output_tokens=40))
    print("Response:", (response.text or "<empty>").strip())
except Exception as exc:
    print("Error:", type(exc).__name__, str(exc)[:600])
    raise SystemExit(1) from exc
finally:
    print("Measured request time:", round(time.perf_counter() - start, 2), "seconds")
