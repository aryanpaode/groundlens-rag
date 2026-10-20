> Baseline screenshots (15 October 2026). For advanced evidence see [advanced/](advanced/) and [../docs/RESULTS.md](../docs/RESULTS.md).

# Original GroundLens screenshots

All fifteen PNG files below are unchanged copies of the user-supplied captures from 15 October 2026. Only filenames were standardized for publication. [SHA256SUMS.txt](SHA256SUMS.txt) records their hashes. Report figures may be cropped; these files preserve the original captured frames.

| Image | Capture | What it establishes / boundary |
|---|---|---|
| [01](01-pinecone-indexing.png) | Indexed library | Dashboard displays four PDFs and 124 manifest-backed Pinecone chunk records; this image alone is not a query execution trace. |
| [02](02-change-drivers-answer.png) | Q1 answer, first part | External and internal organizational-change drivers; source labels visible. |
| [03](03-change-drivers-timing-and-route.png) | Q1 timing and path | Retrieval 1.91 s; Gemini 4.87 s; total 8.80 s; direct generation route. |
| [04](04-change-drivers-evidence.png) | Q1 retrieved sources | [S1] MTM page 4 (0.776); [S2] MTM page 6 (0.679) visible. Partial source-panel capture. |
| [05](05-resistance-comparison-expanded-route.png) | Q2 multi-part answer | Four resistance categories; expand route; total 6.68 s. Expanded citation pages are not shown. |
| [06](06-economic-resistance-answer.png) | Q3 focused answer | Four economic reasons; direct route; total 2.85 s. [S3] is visible but its source page is not expanded. |
| [07](07-gamma-answer.png) | Q4 gamma answer | Formula and brightness cases; direct route; total 3.28 s. |
| [08](08-gamma-evidence-formula-and-brightness.png) | Q4 principal source evidence | [S1] page 26 for brightness; [S2] page 25 for the formula. Similarities 0.765 and 0.738. |
| [09](09-gamma-evidence-transformation-families.png) | Q4 classification evidence | [S3] page 12 and [S4] page 13 describe transformation families. |
| [10](10-gamma-evidence-related-log-transform.png) | Q4 related background | [S5] page 20 is about logarithmic transforms; not needed for the core gamma claim. |
| [11](11-cross-document-answer-and-route.png) | Q5 cross-document answer | Only two PDFs selected; coverage enabled; expand route; total 4.53 s. |
| [12](12-cross-document-evidence-enhancement-p12.png) | Q5 [S1] | Image Enhancement page 12: transformation families; similarity 0.786. |
| [13](13-cross-document-evidence-fundamentals-p6.png) | Q5 [S2] | Image Fundamentals page 6: sampling and quantization; similarity 0.749. |
| [14](14-cross-document-evidence-title-and-fundamentals-p7.png) | Q5 [S3] and [S4] | Enhancement page 1 is title-only (0.729); Fundamentals page 7 explains digitization (0.723). |
| [15](15-cross-document-evidence-point-processing-p11.png) | Q5 [S5] | Image Enhancement page 11: pixel mapping s = T(r), same location and unchanged geometry; similarity 0.704. |

## Interpretation

Q1–Q4 used the full four-document selection. Q5 used only Image Fundamentals and Image Enhancement. These are five selected content tests, not a general accuracy benchmark. The LangGraph `audit` route label denotes a source-number check, not semantic proof of every claim.

The captured interface contains a few contrast artifacts, including a white-on-white model field in some views and mauve evidence blocks. The images were not retouched to conceal those details.

[Recorded measurements](../evaluation/recorded_results.csv) · [Final report](../docs/report/C033_GroundLens_Final_Report.docx)
