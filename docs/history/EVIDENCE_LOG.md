> Historical baseline/build record. For the current release see ../../README.md and ../TESTING.md.

# Recorded execution evidence

Session date: **15 October 2026**. Sources: original application captures, supplied PowerShell output and the approved GroundLens report.

## Setup record supplied by the student

The supplied console text reported `23 passed in 1.66s`, a successful connection to `groundlens-rag-384`, and application startup. It did not show a cloud-vector count at that point because documents had not yet been indexed. Subsequent screenshots show four indexed PDFs and returned source excerpts.

## Functional measurements

| Test | Content outcome | Retrieval (s) | Gemini (s) | Total (s) | Displayed route |
|---|---|---:|---:|---:|---|
| Q1 | Pass | 1.91 | 4.87 | 8.80 | prepare → retrieve → generate → audit |
| Q2 | Pass | 3.15 | 3.33 | 6.68 | prepare → retrieve → expand → generate → audit |
| Q3 | Pass | 1.27 | 1.56 | 2.85 | prepare → retrieve → generate → audit |
| Q4 | Pass | 1.25 | 2.03 | 3.28 | prepare → retrieve → generate → audit |
| Q5 | Pass | 2.34 | 2.18 | 4.53 | prepare → retrieve → expand → generate → audit |

Q1–Q4 searched all four PDFs. Q5 searched the two image-processing PDFs with multi-document coverage. Every response shows five final passages. Total time is preserved as displayed; it includes work outside the two node timers.

## Citation evidence

Q1 has partial expanded evidence for MTM pages 4 and 6. Q2 and Q3 have visible answer labels without their expanded page bindings; content review against lecture notes does not replace a captured citation audit. Q4's key source claims are shown on Enhancement pages 25–26. Q5's main distinction is supported by Fundamentals pages 6–7 and Enhancement pages 11–12; a title-only result was retrieved but not cited in its answer.

## Files

The [screenshot catalogue](../../screenshots/README.md) links all fifteen original captures. [recorded_results.csv](../../evaluation/recorded_results.csv) contains machine-readable measurements and evidence notes. No automatic retrieval-benchmark outcome, new live cloud run or unobserved source-page binding has been added to the recorded session.
