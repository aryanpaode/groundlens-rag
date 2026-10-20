# Advanced feature setup

Start with the base installation and a working `.env`. For the main app install `requirements-advanced.txt` to add Docling, DSPy with Optuna, and Langfuse. Hybrid BM25, hierarchical logic and local reranking use the base dependencies. No new language-model weights are trained.

## Index-time options

Docling, semantic chunking, contextual embeddings and RAPTOR-inspired summaries affect NEW PDFs. A duplicate content hash is not reprocessed merely because checkboxes change. Use a different namespace with a matching fresh manifest for controlled experiments; do not delete the existing collection to test an option.

Contextual enrichment has a default cap of 30 generated descriptions per document. Hierarchy summarization is bounded too. Both consume Gemini quota; `contextual=True` does not prove all chunks were enriched. Docling can download additional layout/OCR model files. CPU processing and memory requirements vary by PDF.

## Question-time options

Hybrid uses Pinecone dense results plus the local BM25 database. Hierarchical mode requires summary records AND local leaf text. Cross-Encoder, CRAG-lite and HyDE are optional; HyDE adds a Gemini request. DSPy needs a real fitted program at `artifacts/dspy_program.json`; a checkbox alone cannot create it. Restore the saved artifact rather than repeat optimization unnecessarily.

## DSPy optimization

```powershell
python scripts\optimize_dspy.py --csv evaluation\dspy_training.csv --output artifacts\dspy_program.json --model YOUR_AVAILABLE_GEMINI_MODEL
```

Review the supplied training references first and keep held-out examples separate. MIPROv2 uses a simple overlap objective in this implementation; its score is not answer accuracy. It may make many API requests. The saved JSON from the demonstrated laptop run is private local state and is not distributed in this source package.

## RAGAS and Langfuse

RAGAS MUST use `requirements-ragas.txt` in a separate `.venv-ragas` environment. Do not install the old RAGAS/LangChain constraints into the main app. Langfuse stays in the main environment. Full commands and privacy considerations are in [LANGFUSE_RAGAS.md](LANGFUSE_RAGAS.md).

The updated report and screenshots document the actual demonstrations. Historical packaging checks are not replacements for live provider tests. Use [TESTING.md](TESTING.md) for the current build checks.
