# Week 8 — recognition-model bake-off (final testing)

**Test input:** the top half of page 1 of a real handwritten assignment (UHV Assignment 1, student J004,
"classmate" ruled paper, cursive), scanned at 2368×1650. Ground truth was transcribed by hand from the scan
(`ground_truth_real_page.txt`), with crossed-out words excluded and verified by zooming into ambiguous ink.
All scores were computed with the project's own `exam_digitizer/evaluation/metrics.py` (CER / WER after NFKC
tokenisation); the header block is excluded from every system's text. Word-level diffs for every run:
`model_comparison.html`.

| # | System | Where it runs | CER ↓ | WER ↓ |
|---|--------|---------------|-------|-------|
| 1 | Gemini 3 Flash + verification pass (full page) | hosted API | **0.025** | **0.042** |
| 2 | Gemini 3 Flash, single pass (full page) | hosted API | 0.028 | 0.070 |
| 3 | Gemini — integrated pipeline (line-level) | hosted API | 0.055 | 0.092 |
| 4 | Qwen3-VL-8B Q4_K_M, llama.cpp (full page) | local, offline | 0.059 | 0.085 |
| 5 | Qwen3-VL-2B, tuned prompt (full page) | local, offline | 0.067 | 0.106 |
| 6 | Qwen3-VL-2B — integrated pipeline (line-level) | local, offline | 0.080 | 0.162 |
| 7 | TrOCR-small — Week-7 pipeline (baseline) | local, offline | 0.160 | 0.359 |
| 8 | PP-OCRv6 (PaddleOCR, English) | local, offline | 0.427 | 0.620 |

## Findings

1. **General vision-language models beat specialized OCR on real cursive by a wide margin.** The Week-7
   recogniser (TrOCR-small) that scored 0.045 text CER on the synthetic IAM-based booklets degrades to 0.16
   on real handwriting; a lightweight OCR pipeline (PP-OCRv6) collapses entirely (0.43).
2. **Model scale works:** within one family, 8B < 2B error; the hosted frontier model (Gemini) halves the
   best local result again.
3. **Full-page reading beats line-level reading** for every VLM (sentence context disambiguates words like
   *rightly*), at the cost of bypassing the pipeline's per-line confidence routing.
4. **A verification pass pays:** feeding Gemini its own draft plus the image cut WER from 0.070 to 0.042
   for one extra call.
5. **Residual errors are structural, not random:** struck-through text leaking into the transcript, one
   genuinely ambiguous plural flourish, one *form/from* cursive swap, and handwritten digits in IDs
   (Gemini read the roll number "J004" as "JODU") — which is why the pipeline's glyph-CNN + roster check
   still owns student-ID reading.
6. **Layout fix (Week 8):** text-wide lines written in the bottom page-number strip are no longer dropped
   (previously cost every system the page's last line); covered by a new unit test.

## Unit/integration tests

Full pytest run of the final repository: see `pytest_full_report.txt` (43 passing fast tests incl. 10 new
Week-8 backend/layout tests; slow end-to-end tests additionally exercise TrOCR when its weights are cached).

## Models ruled out without a run (hardware)

Chandra (9B, custom arch) and DeepSeek-OCR need GPUs beyond the 4 GB available here; MiniCPM-V 2.6 (8B)
was not downloaded for disk reasons. Published IAM figures suggest MiniCPM-V ≈ Qwen2-VL class.
