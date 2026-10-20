# Final repository/laptop package

## Preserved

The consolidated GroundLens application, established advanced modules, Streamlit interface, source filtering, report terminology and existing Pinecone representation are retained. The updated 38-page report is copied unchanged. All available advanced-session screenshots are preserved, including intermediate failure evidence.

## Included corrections and packaging additions

- RAGAS timeout changed from 90 to 240 seconds, matching the successful laptop run.
- RAGAS compatibility dependencies moved to `requirements-ragas.txt` for a separate environment; they are no longer installed by the main advanced requirements file.
- `.env.example` uses the demonstrated `groundlens-raptor-test` namespace and the exact Langfuse opt-in variable. Secrets remain placeholders and tracing defaults to OFF.
- PyMuPDF imports now use `import pymupdf as fitz`; existing calls retain their alias.
- `restore_local_state.py` restores actual private laptop state without copying `.env` or touching Pinecone. It uses SQLite's backup API rather than copying live WAL files.
- `build_repo_zip.py` builds a clean publication archive even after private state has been added locally.
- Setup, architecture, privacy and result documentation now distinguish the completed RAGAS/Langfuse demonstrations from pending broad quality benchmarks.

No optimized DSPy JSON, live manifest, BM25 database, measured RAGAS CSV or API credential has been invented or included in place of the laptop originals. Those can be restored locally from the current working project. Source PDF originals and model/virtual-environment downloads are not part of the archive.

## Implementation reference for the state-backup helper

Python sqlite3 Connection.backup: https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup
