# Testing and recorded evidence

## Three distinct evidence types

1. **Automated code checks:** the supplied Windows console record states `23 passed in 1.66s`. The graph tests use real LangGraph compilation with fake retrieval and generation; they do not spend cloud credits. See [package verification](PACKAGE_VERIFICATION.md) for the separate packaging run.
2. **Observed academic answers:** the five rows in [recorded_results.csv](../evaluation/recorded_results.csv) were transcribed from genuine application screenshots. The full question, main result, timings, path and evidence limitations are preserved.
3. **Prepared retrieval benchmark:** [academic_benchmark.csv](../evaluation/academic_benchmark.csv) contains 24 questions for an optional future run. No hit-rate or reciprocal-rank result is reported for this dataset.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Reproduce the recorded academic questions

Use permitted copies of Organizational Change (18 pages), Image Fundamentals (14 pages), Image Enhancement Part A (52 pages), and User Interface Design (15 pages). Use five evidence passages. Questions 1–4 searched all four PDFs; question 5 searched only the two image-processing PDFs and enabled document coverage.

| ID | Exact recorded question | Expected content/source |
|---|---|---|
| Q1 | What are the external and internal forces that trigger organizational change? | MTM pages 4–6: external and internal drivers. |
| Q2 | Explain the difference between economic, psychological, social, and organizational resistance to change. | MTM pages 8–10: distinguish the four categories. |
| Q3 | What are the economic reasons for individual resistance to organizational change? | MTM page 8: job security, skill obsolescence, economic loss and reduced incentives. |
| Q4 | Explain power-law (gamma) transformation and how different gamma values affect image brightness. | Image Enhancement pages 25–26: formula and brightness cases. |
| Q5 | How do sampling and quantization in digital image fundamentals differ from intensity transformation techniques used in image enhancement? Explain using both documents. | Digitization versus point processing; two-source evidence. |

A content pass means the response answers the principal question correctly against the notes. It does not automatically establish every citation binding. Source-panel evidence is partial for Q1, not expanded for Q2/Q3, and available for the core claims of Q4/Q5.

## Optional retrieval-only evaluation

Select the relevant dataset in **Evaluate retrieval**. The academic CSV retains filenames of the supplied source copies. Your indexed filename may differ due to spaces, hyphens or an omitted `(1)`. Before running, edit `expected_documents` in the CSV to match the filenames shown in **Manage PDFs** exactly, including each semicolon-separated required document.

A positive case passes only if all specified filename/page/phrase requirements appear. Missing PDFs are skipped; unsupported cases need manual review. The rank summary for multi-source cases uses the rank at which **all** requirements have been found, with zero when any required source is missing. It is not generated-answer accuracy.

The utility calls the vector-store `search` method directly. It does not invoke the LangGraph expansion graph or Gemini. With Pinecone configured, it is **not an offline benchmark** and uses Pinecone requests.

The fictional event samples remain optional. `tests/test_sample_benchmark_integrity.py` checks their prepared phrases against extracted chunks; it is not one of the five academic answer observations.

## Interpretation of the recorded timings

The five displayed totals are 8.80, 6.68, 2.85, 3.28 and 4.53 seconds. Their mean is 5.23 seconds and median 4.53 seconds. Q1 has 2.02 seconds outside the two rounded node timings. Resource access and graph orchestration are included in total time; no independent breakdown of that residual was recorded.

Do not treat these selected questions as a corpus-wide accuracy estimate, a controlled performance comparison or proof that all missing-information and error paths have been validated.
