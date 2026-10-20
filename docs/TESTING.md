# Packaging verification — 20 October 2026

## Executed in this packaging environment

- `python -m pytest -q`: **62 passed, 1 skipped**.
- The skipped item is the LangGraph-dependent test module, because LangGraph is not installed here. Its two tests were not executed. An attempted dependency download was unavailable in this environment.
- Source compilation completed without syntax errors.
- State restoration tests cover database snapshotting, manifest consistency, no credential copying, no artifact fabrication, explicit overwrite permission and destination backups.
- Publication tests check exclusions for `.env`, virtual environments, private SQLite/manifest data, generated DSPy artifacts and live RAGAS CSVs.
- Archive integrity and report checksum checks are performed during packaging.

## Not executed here

No authenticated Pinecone, Gemini, DSPy, Docling, RAGAS or Langfuse integration run was performed for this archive. The main Streamlit UI and Windows PowerShell wrappers were not launched in this Linux packaging environment. The supplied laptop screenshots separately document the working app and the completed live RAGAS/Langfuse tests.

A passing local unit test is not a cloud success or an accuracy benchmark. The packaging test count is separate from earlier test counts in the report and historical records.

## Recheck locally

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-tests.txt
& .\.venv\Scripts\python.exe -m pytest -q
```

Reuse the full path of your working interpreter instead of `.venv` if needed. Do not install RAGAS pins into that environment. API diagnostic scripts make actual requests and should only be run when intended.
