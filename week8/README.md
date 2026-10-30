# Week 8 — final deliverables (Group 1 · Handwritten Examination Digitizer)

Everything for the Week-8 submission, in one folder. Headline: on a **real** handwritten assignment
page the recogniser went from **16.0 % CER / 35.9 % WER** (Week-7 TrOCR-small) to **2.5 % CER /
4.2 % WER** (Gemini + verification pass), with a fully-local option at **5.9 % CER** (Qwen3-VL-8B).
The recognition model is now a configuration switch with automatic fallback, covered by tests.

| deliverable | where | notes |
|---|---|---|
| **Final repository** | `handwritten_exam_digitizer/` and `handwritten_exam_digitizer_week8.zip` | asset v0.8.0: swappable recognition backends (`EXAM_DIGITIZER_HTR`: `gemini` default · local Qwen3-VL · TrOCR fallback), layout fix, updated README |
| **Final testing** | `testing/` | `pytest_full_report.txt` (**52 passed, 1 skipped**, incl. slow end-to-end model tests) · `bakeoff_results.md` (8-system benchmark on a real page) · `model_comparison.html` (word-level diffs) · `ground_truth_real_page.txt` |
| **Report** | `report/SNLP_Week8_Report.pdf` (+ `.md` source) | method, results, changes, responsible AI; Week-7 report included for reference |
| **README** | `handwritten_exam_digitizer/README.md` | project README with the new "Choosing the recognition model" section |
| **Presentation** | `presentation/SNLP_Week8_Presentation.pptx` | 10 slides: problem → method → results → errors → what we built → architecture → demo → responsible AI → close |
| **Demonstration** | `demo/` | step-by-step demo script (`README.md`), real test page, full 3-page Gemini transcript |
| **Architecture diagram** | `docs/architecture_week8.png` | pipeline with the three swappable backends and fallback path |

## One-line setup for the demo

```bash
cd handwritten_exam_digitizer && pip install -e ".[api,pdf]"
# Gemini backend: put GEMINI_API_KEY=... in a .env (never commit it); without a key the
# digitizer warns and falls back to the local TrOCR model.
```

## Known-open items (stated honestly)

* The frozen 40-booklet synthetic benchmark has not been re-run with the new backends; the bake-off
  evidence is one real page, one writer.
* Platt calibration (`assets/calibration.json`) is still fitted for TrOCR — re-fit per backend
  before trusting `needs_review` routing.
* Struck-through text occasionally leaks into transcripts; handwritten digits stay with the
  glyph-CNN + roster check.
