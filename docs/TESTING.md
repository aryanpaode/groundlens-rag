# Test record and reproducibility

## Distinguish three kinds of evidence

1. **Recorded laptop demonstration:** five completed academic question-answering outputs supplied during October 2026, with application timings. These results are preserved below and in `evaluation/observed_v2_tests.csv`.
2. **Packaging unit tests:** the local verification of this distribution, described in `PACKAGE_VERIFICATION.md`. No real Gemini request or neural embedding run was performed during packaging.
3. **Retrieval benchmarks:** included code and expected-source datasets. No new benchmark results are invented or prefilled as measured outputs.

## Final academic corpus (full lecture files not redistributed)

| Document | PDF pages | Subject area |
|---|---:|---|
| MTM — Organizational Change | 18 | Management Through Movies |
| Unit 1 — Image Fundamentals | 14 | Image and Video Processing |
| Unit 2 — Image Enhancement (Part A) | 52 | Image and Video Processing |
| Unit 6 — User Interface Design | 15 | Software Engineering |
| **Total** | **99** | **3 subject areas; 4 PDFs** |

The author's index displayed 124 chunks. The source names in uploaded files can vary by spaces and suffixes such as `(1)`; automated source matching requires exact filenames. The actual lecture files are deliberately absent from this repository.

## Recorded questions

| ID | Question | Outcome and evidence |
|---|---|---|
| Q1 | What are the external and internal forces that trigger organizational change? | Pass for the main facts; expected source is MTM pp. 4–6. |
| Q2 | Explain the difference between economic, psychological, social, and organizational resistance to change. | Fail: an incorrect abstention. Key evidence is on MTM pp. 8–10, but retrieved pages were 18, 2, 10, 7 and 3. |
| Q3 | What are the economic reasons for individual resistance to organizational change? | Pass: job security, skill obsolescence, economic loss and reduced incentives; MTM p. 8. |
| Q4 | Explain power-law (gamma) transformation and how different gamma values affect image brightness. | Pass for the formula and gamma cases in Image Enhancement pp. 25–26. |
| Q5 | How do sampling and quantization in digital image fundamentals differ from intensity transformation techniques used in image enhancement? Explain using both documents. | Pass for the central distinction. Visible evidence includes Image Fundamentals pp. 6–7 and Image Enhancement pp. 3–4. |

Q5 used only the two image-processing PDFs, five evidence passages and **Cover multiple documents** enabled. Complete source-filter and balancing settings for every earlier turn were not retained. Citation markers were visible, but this record does not constitute claim-by-claim entailment scoring.

## Observed latency

| ID | Retrieval (s) | Gemini (s) | Total (s) |
|---|---:|---:|---:|
| Q1 | 17.41 | 8.69 | 26.10 |
| Q2 | 0.73 | 18.32 | 19.05 |
| Q3 | 0.05 | 8.08 | 8.13 |
| Q4 | 0.05 | 5.41 | 5.46 |
| Q5 | 0.05 | 5.90 | 5.95 |

Four of five answers passed the main content check. The observed pass proportion is 80% **only within this selected five-question sample**, not a general accuracy estimate. The mean total time is 12.94 s; the median is 8.13 s. Initial retrieval overhead was not independently instrumented. One earlier Q5 attempt returned API error 503 and is not counted as another completed answer.

## Run the distributed synthetic benchmark

Upload all three PDFs in `sample_documents/`. Choose **Fictional event (original demo)** in **Evaluate retrieval** and click **Run local benchmark**. This path does not call Gemini. It measures whether each required source document, page and text phrase appears among the retrieved passages.

The dataset contains 20 positive retrieval cases and two manual missing-information cases. Cases with missing required documents are skipped. Verify that the source names match the CSV exactly. Record the actual exported CSV from your own run; no benchmark score is supplied as an observed result in this repository.

For multi-source cases, the reciprocal-rank summary uses the rank at which **all** requirements have been found, assigning zero when any requirement is missing. The `first_relevant_rank` table column has a different purpose: it shows the earliest found requirement. Neither quantity measures generated-answer correctness.

## Legacy academic benchmark

`evaluation/four_subject_benchmark.csv` is the older **Four academic subjects** dataset that includes Game Design. It is retained because the app and benchmark code are preserved from the demonstrated version, but it **does not match the final corpus above**. Do not use its score as the final academic result without first aligning filenames and source requirements.

`evaluation/manual_answer_tests.csv` is a synthetic-event checklist, not the five academic observations. `evaluation/observed_v1_tests.csv` contains historical early runs across different generation configurations; do not combine them with v2 timings as one controlled comparison.

## Automated code checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
```

The suite covers chunking, page references, empty input, prompt construction, citation-number helpers, document-diversity selection, metric calculation and consistency of sample benchmark phrases. The ChromaDB integration test uses small deterministic test embeddings, not the pretrained sentence model; a pass would verify database plumbing, not semantic-retrieval quality.
