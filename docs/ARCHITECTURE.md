# GroundLens architecture

[README](../README.md) · [Updated report and diagrams](report/C033_GroundLens_Final_Report.docx)

## Ingestion

`app.py` passes opt-in settings to `PineconeVectorStore`. `rag/document_ingestion.py` uses PyMuPDF or IBM Docling, preserving original PDF page numbers. Semantic splitting uses the existing local sentence embedder, not a newly trained model. Optional contextual descriptions enrich embedding inputs with a bounded number of Gemini requests. RAPTOR-inspired indexing clusters chunks, recursively summarizes groups, and indexes summary records with leaf references.

CPU MiniLM creates normalized 384-dimensional vectors. Pinecone stores dense vectors, source text and metadata. `runtime/pinecone_manifest.json` stores document identities and options; `runtime/groundlens_bm25.sqlite3` stores original chunks for BM25 and summary-to-leaf expansion. The two files and Pinecone namespace must remain consistent. PDF originals are not bundled in a private index backup.

## Retrieval and control flow

```text
START -> prepare -> retrieve
                    |-> correct (conditional)
                    |-> expand  (subtopic hints)
                    |-> abstain (no evidence) -> audit -> END
                    |-> rerank  (optional)
                    -> generate [optional DSPy branch] -> audit -> END
```

This is a schematic; the detailed graph is in the report and `rag/workflow.py`. `prepare` creates bounded heuristic hints and optionally a HyDE search paragraph. Retrieval can fuse dense and local BM25 ranks using RRF. Hierarchical retrieval resolves matched summaries back to individual source passages and ranks leaves before fusion. Cross-Encoder reranking operates on retrieved candidates. Corrective retrieval makes at most one additional search inside the selected document scope; no web fallback is implemented.

The `dspy_generate` path label is emitted by the generation branch when a saved DSPy program is used; it is not a separate independently compiled graph node. `audit` checks recognized source-number labels, not factual entailment. The UI's route is a diagnostic sequence, not a complete causal explanation of model reasoning.

## Evaluation and observations

RAGAS runs offline from real exported questions, answers and contexts in a separate `.venv-ragas`, with one worker and a 240-second timeout. The main application runs without RAGAS installed. Langfuse tracing is explicit opt-in; root/question, retrieval and generation observations were received in the supplied live capture. Other optional spans execute only on their relevant branches. Provider-reported token usage may be attached; missing usage is not guessed.

## Performance and scope

The displayed total includes work outside the retrieval and generation timers. Do not attribute unexplained differences to one library without measuring it. Original screenshots include cold-start, quota and availability effects. The architecture provides optional methods, not proven accuracy gains for every configuration.

Original baseline diagrams, if retained under `docs/images/`, describe the earlier dense-only system. Use the updated report's architecture and graph diagrams for the complete implementation.
