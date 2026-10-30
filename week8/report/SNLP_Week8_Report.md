# Handwritten Examination Digitizer — Week 8 Report
**Upgrading the recognition model for real handwriting**
NLP Course Project 2026 · AI University Platform · Group 1 (J001, J002, J004, J005, J006)
Asset version 0.8.0 · builds on the Week-7 report and repository

## 1. Summary

The Week-7 digitizer read our synthetic IAM-based booklets well (text CER 0.045) but failed on real
handwriting: on a page of a genuine handwritten assignment it reached **16.0 % CER / 35.9 % WER** — above
the 10 % CER level the HTR literature calls unusable. We therefore benchmarked recognition models from
four families on that page against a hand-verified ground truth, using the project's own metrics, and
rebuilt the recognition stage so the model is **swappable by configuration**. Result: **2.5 % CER / 4.2 %
WER** with a hosted model (Gemini, with a verification pass) and **5.9 % CER / 8.5 % WER** with the best
fully-local open model (Qwen3-VL-8B, 4-bit). Every clearly-written word on the test page is now read
correctly and in order; the residual errors are struck-through text leaking into the transcript, one
ambiguous plural flourish, one *form/from* cursive swap, and handwritten digits in the roll number.

## 2. Test setup

* **Input:** top half of page 1 of a real assignment (UHV Assignment 1, student J004; ruled "classmate"
  paper, cursive, with cross-outs and an inserted correction) — conditions our synthetic booklets do not
  cover.
* **Ground truth:** transcribed manually from the scan; ambiguous ink verified by zooming (one of our own
  first readings — "once" — was corrected to "Once" this way). Crossed-out words are excluded, as a
  grader-ready transcript would exclude them.
* **Metrics:** CER and WER from `exam_digitizer/evaluation/metrics.py` (NFKC + punctuation tokenisation),
  header block excluded for every system. Word-level diffs for all runs: `testing/model_comparison.html`.

## 3. Results

| # | System | Where it runs | CER ↓ | WER ↓ |
|---|--------|---------------|-------|-------|
| 1 | Gemini 3 Flash + verification pass (full page) | hosted API | **0.025** | **0.042** |
| 2 | Gemini 3 Flash, single pass (full page) | hosted API | 0.028 | 0.070 |
| 3 | Gemini — integrated pipeline (line-level) | hosted API | 0.055 | 0.092 |
| 4 | Qwen3-VL-8B Q4 (llama.cpp, full page) | local, offline | 0.059 | 0.085 |
| 5 | Qwen3-VL-2B, tuned prompt (full page) | local, offline | 0.067 | 0.106 |
| 6 | Qwen3-VL-2B — integrated pipeline (line-level) | local, offline | 0.080 | 0.162 |
| 7 | TrOCR-small — Week-7 pipeline (baseline) | local, offline | 0.160 | 0.359 |
| 8 | PP-OCRv6 (PaddleOCR, English) | local, offline | 0.427 | 0.620 |

**Findings.** (i) General vision-language models beat specialized OCR on real cursive by a wide margin —
the specialized engines (TrOCR-small, PP-OCRv6) produce confident wrong words, exactly the failure mode the
Week-7 report identified as most dangerous for downstream grading. (ii) Scale helps: 8B < 2B error within
one family; the hosted frontier model halves the best local error again. (iii) Full-page reading beats
line-level reading for every VLM, because sentence context disambiguates words (*rightly* vs *highly*);
the line-level pipeline keeps per-line confidence routing instead. (iv) A verification pass — feeding the
model its own draft plus the image — cut Gemini's WER from 7.0 % to 4.2 % for one extra call. (v) VLMs
misread handwritten digits (roll number J004 → "JODU"), so the glyph-CNN + roster check keeps owning
student-ID reading. Chandra and DeepSeek-OCR could not run on the available 4 GB GPU; MiniCPM-V was not
tested (disk budget).

## 4. Changes to the asset (Week 8)

1. **Swappable recognition backend.** `models/recognition.py` now provides a factory,
   `make_line_reader(name)`, selected by `EXAM_DIGITIZER_HTR`: `gemini` (default; key from
   `GEMINI_API_KEY` env or `./.env`, line crops JPEG-batched into one request per page, retries on
   network errors), any Hub id containing `-VL` or prefixed `vlm:` for a local open VLM
   (Qwen3-VL via transformers ≥ 4.57), `tesseract`, or any other Hub id for TrOCR. A backend that cannot
   load **warns and falls back to TrOCR**, so the pipeline never hard-fails on a missing key or old
   library.
2. **Layout fix.** A text-wide line written in the bottom page-number strip (the last line of a full
   page) is no longer dropped; the narrow printed page number still is. This had cost every recogniser
   the page's final line (−3 CER points for the integrated pipeline).
3. **Tests.** 11 new unit tests (backend selection, fallback, Gemini batching/parsing, default config,
   bottom-strip layout). Full suite: **52 passed, 1 skipped**, on both transformers 4.41 and 5.x; the
   suite pins the local TrOCR model so tests never call the network.

## 5. Responsible AI

The Gemini backend sends page images to Google. Assignment §8 and our own Week-7 commitments therefore
still stand: **real student scripts must never be sent to it** — the hosted backend is for synthetic or
consented material only (the test page was a group member's own assignment, used with consent), and the
fully-local Qwen path exists precisely so the privacy guarantee can be kept where it matters. The packaged
Platt calibration was fitted for TrOCR confidences; other backends' confidences are on different scales
(Gemini backend reports self-estimated confidence), so `assets/calibration.json` must be re-fitted per
backend before `needs_review` routing is trusted. The API key lives in an untracked `.env` and is never
logged.

## 6. Limitations and next steps

Single-page evidence: the bake-off used one real page and one writer; the frozen 40-booklet synthetic set
has not yet been re-run with the new backends (and IAM-derived data likely flatters VLMs, which saw IAM in
training). Struck-through text still leaks into transcripts occasionally; detection could move into
preprocessing. Local VLM speed on CPU (minutes/page) needs the quantized-GPU path productionised.
Next: re-run the 40-booklet benchmark, re-fit calibration per backend, OCR-tolerant key-point matching on
the grader side (unchanged from Week 7), and evaluate a page-level mode that keeps per-line confidence.
