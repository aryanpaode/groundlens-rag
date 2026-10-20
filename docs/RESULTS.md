# Recorded GroundLens results

Source: the actual screenshots supplied during 19-20 October 2026 testing. No authenticated model or database request was performed merely to manufacture packaging results.

## Final corpus

The final management capture shows 4 PDFs, 106 pages and 188 original-source passages. The Organizational Change PDF additionally has 4 hierarchical summary nodes. The dashboard's source-passage count excludes those summary nodes; it is not an independently sampled Pinecone cloud statistic.

| Document | Pages | Source passages | Summary nodes | Parser |
|---|---:|---:|---:|---|
| Organizational Change | 18 | 18 | 4 | PyMuPDF |
| Image Enhancement, Part A | 52 | 55 | 0 | Docling |
| Image Fundamentals | 14 | 41 | 0 | Docling |
| Unit-8 (Digital Video content) | 22 | 74 | 0 | PyMuPDF |

Evidence: [final collection](../screenshots/advanced/15-final-four-document-collection.png). The manifest screenshots record semantic=True for Image Enhancement and contextual=True for Unit-8. The contextual flag does not establish generated-context coverage for every chunk.

## Selected functional observations

| Configuration | Retrieved/generation/total seconds | Evidence |
|---|---|---|
| Hybrid | 1.47 / 1.59 / 3.07 | [RRF passage](../screenshots/advanced/01-hybrid-rrf-source-passages.png) |
| Hybrid + rerank | 1.59 / 39.75 / 41.37 | [rerank route](../screenshots/advanced/02-cross-encoder-execution.png) |
| Corrective branch | 1.90 / 1.28 / 3.20 | [unsupported question](../screenshots/advanced/03-corrective-routing.png) |
| HyDE enabled | 2.77 / 2.09 / 6.12 | [answer](../screenshots/advanced/04-hyde-enabled-response.png) |
| DSPy ON | 1.74 / 3.43 / 26.81 | [DSPy route](../screenshots/advanced/08-dspy-generation-path.png) |
| DSPy OFF control | 1.58 / 1.84 / 3.43 | [control](../screenshots/advanced/09-control-without-dspy.png) |
| Final live traced answer | 1.26 / 17.56 / 41.92 | [application capture](../screenshots/advanced/18-final-groundlens-answer.png) |

These are individual displayed measurements, not paired statistical benchmarks. Totals include unmeasured overhead; they need not equal the sum of the two stage timers. Rerank node execution is not an independent verification of every ranking score. A HyDE-enabled response is not a captured dump of its hypothetical search text.

## RAGAS

The console shows `{'faithfulness': 1.0000}`, one scored row, and 48.78 seconds. [Screenshot](../screenshots/advanced/16-ragas-faithfulness-measured.png). The evaluation used RAGAS 0.3.9 in its separate environment after increasing the runner timeout to 240 seconds.

This is a judge's support assessment for one exported answer, not 100% general correctness. The raw `ragas_results.csv` is local to the laptop and is not recreated from the screenshot. Restore it from the existing installation for local use.

## Langfuse

The [synthetic trace](../screenshots/advanced/17-langfuse-synthetic-trace.png) arrived. The [real dashboard capture](../screenshots/advanced/19-langfuse-real-trace.png) additionally shows `groundlens-question`, `pinecone-retrieval` and `gemini-answer`. This demonstrates live export of those observations. It does not independently prove that all optional branches, token accounting, costs or evaluation scores have been captured in Langfuse.

## Original baseline

The earlier 4-PDF, 99-page, 124-chunk baseline and its five-question results remain under `screenshots/` and `evaluation/recorded_results.csv`. Do not combine baseline collection counts with the advanced corpus above. The updated report distinguishes these datasets and the implementation boundaries.
