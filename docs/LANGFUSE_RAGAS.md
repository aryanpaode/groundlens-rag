# RAGAS and Langfuse

## Recorded completion

A supplied live run completed RAGAS faithfulness for **one exported answer**, scoring **1.0000** in **48.78 seconds**. The supplied Langfuse dashboard received `groundlens-question`, `pinecone-retrieval` and `gemini-answer`, in addition to the synthetic check. Evidence: [RESULTS.md](RESULTS.md). These are observed runs, not universal accuracy or availability guarantees.

## RAGAS: separate Python environment

```powershell
py -3.11 -m venv .venv-ragas
$ragasPy = ".\.venv-ragas\Scripts\python.exe"
& $ragasPy -m pip install -r requirements-ragas.txt
& $ragasPy -c "from ragas import evaluate; from ragas.metrics import faithfulness; from langchain_google_genai import ChatGoogleGenerativeAI; print('RAGAS imports OK')"
```

The pinned RAGAS 0.3.9 / LangChain 0.3 family retains the import path required by the existing evaluator. Keep it out of the app's environment. Do not replace it with the latest major versions during final submission preparation.

Export actual answers through **Evaluate retrieval**, save the CSV into `evaluation/`, then:

```powershell
& $ragasPy -m evaluation.prepare_ragas --input evaluation\groundlens_ragas_input.csv
& $ragasPy -m evaluation.ragas_runner --dry-run evaluation\ragas_ready.csv
& $ragasPy -m evaluation.ragas_runner --limit 1 --metrics faithfulness evaluation\ragas_ready.csv
```

Review matched `ground_truth` before scoring. Dry-run makes no Gemini requests. Live scoring uses one worker and a 240-second operation timeout; retries are bounded. Multiple model calls may be needed even for one answer. Quota/availability errors remain possible. A failed evaluation does not prove answer quality and is not assigned a fabricated score.

`ragas_results.csv` may use `user_input` for the question column. Inspect actual fields before selecting them in PowerShell:

```powershell
Import-Csv .\evaluation\ragas_results.csv | Select-Object -First 1 | Format-List
```

Do not interpret a blank selected `question` column as an evaluation failure when the actual field is named differently. The earlier LangchainLLMWrapper deprecation warning did not prevent the demonstrated run.

## Langfuse: main app environment, explicit opt-in

Add private keys to `.env`:

```ini
GROUNDLENS_LANGFUSE_ENABLED=1
LANGFUSE_PUBLIC_KEY=YOUR_PRIVATE_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_PRIVATE_PROJECT_SECRET_KEY
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Use your own regional base URL. The setting is `GROUNDLENS_LANGFUSE_ENABLED`, not `Langfuse_enabled`.

```powershell
python scripts\check_langfuse.py --send
```

The script attempts a synthetic export and flushes queued observations. Dashboard receipt confirms delivery. Restart Streamlit after changing `.env`, then ask a permitted-document question and inspect the root and child observations. Do not share screenshots containing API secrets.

Tracing sends the question, generated answer and instrumented stage metadata to Langfuse. It is disabled by default in `.env.example`. RAGAS scores are saved separately to CSV; automatic synchronization of those scores into Langfuse is not implemented.

## References

- RAGAS 0.3.9 RunConfig: https://docs.ragas.io/en/v0.3.9/references/run_config/
- RAGAS 0.3.9 evaluate: https://docs.ragas.io/en/v0.3.9/references/evaluate/
- Langfuse SDK overview: https://langfuse.com/docs/observability/sdk/overview

These references explain SDK behaviour. They are not evidence of this project's live runs; the screenshots provide that evidence.
