# Evaluation summary (Week 5, frozen test set) and the Week-6 regression check

All numbers are produced by the project notebook (`notebooks/`), on a frozen test set that was never used for tuning:
40 booklets from 40 IAM *test* writers, 61 pages, 114 answers, 354 lines. 95 % CIs are cluster bootstraps over
booklets (1,000 resamples). Metric definitions are in `exam_digitizer.evaluation`.

## Systems compared

| ID | System | answer CER ↓ | WER ↓ | label-free CER ↓ | Seg F1 ↑ | ID acc / wrong | s/page |
|---|---|---|---|---|---|---|---|
| S0 | Tesseract page + regex (baseline) | 0.939 [0.900, 0.971] | 0.982 | 0.443 | 0.164 | 0.00 / 0.00 | 3.1 |
| S1 | our layout + Tesseract lines | 0.477 | 0.850 | 0.420 | 0.925 | 0.95 / 0.00 | 3.2 |
| **S2** | **our layout + TrOCR-small (the asset)** | **0.124 [0.070, 0.188]** | 0.184 | **0.045 [0.034, 0.057]** | **0.925 [0.867, 0.972]** | **0.95 / 0.00** | 2.8 |
| S2b | our layout + TrOCR-base | 0.117 | 0.165 | 0.038 | 0.925 | 0.95 / 0.00 | 19.7 |
| S3 | S2 + LLM post-correction (cloud) | 0.125 | 0.182 | 0.046 | 0.925 | 0.95 / 0.00 | 2.8 |
| S4 | vision-LLM page reader (cloud, 25 booklets) | 0.241 | 0.277 | 0.041 | 0.942 | 0.84 / 0.08 | 0.45 (API) |

Paired bootstrap: S2 vs S0 improves CER, label-free CER and Seg F1 with p ≤ 0.001 (105 of 114 answers better).
S2b vs S2: CER −0.007 (p = 0.002) at 7× the CPU time. S3 and S4 are not significantly different from S2.

## Error budget of the asset (S2)

| component | metric | value |
|---|---|---|
| deskew | mean absolute angle error | 0.06° |
| line segmentation | recall / precision (IoU ≥ 0.5) | 0.938 / 0.941 |
| margin-mark detection | recall | 0.956 |
| question-label reading | accuracy given detection | 0.936 |
| line recognition (TrOCR-small) | CER on correctly segmented lines | 0.044 |
| roll-number glyphs | character accuracy | 0.971 |

Of the answer-level CER of 0.124: 0.041 misread characters, 0.004 lost/merged lines, **0.079 answers filed under a
misread question number**. Error types per 100 words: real-word substitutions 3.6, non-word 2.9, punctuation 1.1,
deletions 1.4, insertions 2.2.

## Robustness (8 booklets, 21 conditions)

Within ≈ 0.03 CER of clean: rotation up to 9° on a dark background, 0.3× resolution, faded ink, shadows, JPEG quality
≥ 10. Breaks: blur σ ≥ 3.5 px (CER 0.47), sensor noise σ ≥ 30 (CER 0.23–0.39, Seg F1 0.64).

## Stress tests and failure cases

| situation | result |
|---|---|
| inline question numbers (not in the margin) | Seg F1 0.31 — the main structural failure |
| no printed margin line | unchanged (virtual margin) |
| reverse question order | unchanged |
| non-Latin (Pashto) answer | hallucinated English, but line confidence 0.21 vs 0.89 → 88 % flagged |
| blank roll-number boxes | `student_id` null in 100 % of booklets (no guessing) |
| student not on the roster | `digits-exact` assigned 1 of 8 to another student; `strict` 0; `snap` 8 |
| real photographed two-column page | columns interleaved; bag-of-words F1 0.78 |

## Confidence calibration (shipped in `assets/calibration.json`)

| confidence | AUROC usable vs not | Brier | ECE |
|---|---|---|---|
| raw (mean line prob × √header prob) | 0.867 | 0.133 | 0.088 |
| Platt scaling, 5-fold CV on test (deployed) | 0.855 | 0.109 | 0.061 |

## Week-6 regression check

`results/regression_week6.json` is written by the Week-6 section of the notebook: the packaged `Digitizer` is run on
the same 40 test booklets and its output is compared field by field with the notebook pipeline's output
(identical answers expected), and the test-set metrics of the package are re-reported with CIs.
