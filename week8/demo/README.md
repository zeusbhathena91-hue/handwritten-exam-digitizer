# Week 8 demonstration

What to show, in order, with the commands. All paths are inside `week8/handwritten_exam_digitizer`
unless noted. Install once: `pip install -e ".[api,pdf]"` (Python ≥ 3.10; the Gemini backend needs
`GEMINI_API_KEY` in the environment or a `.env` in the working directory).

## 1. The problem (30 s)

Open `../testing/model_comparison.html`. Scroll from the Gemini section (almost no marks) to the
TrOCR-small section (a sea of red): same ink, same scorer — only the model changed.

## 2. Digitize a real handwritten page with the new default backend (1 min)

```bash
python -m exam_digitizer.cli digitize ../demo/sample_real_page.png --booklet-id DEMO
```

Returns question-wise JSON (`Q1`, `ans` found from the margin) with per-line confidences via Gemini.
Point out `model: layout+gemini-3-flash-preview` in the output.

## 3. Same page, fully local — the privacy path (1 min)

```bash
EXAM_DIGITIZER_HTR=Qwen/Qwen3-VL-2B-Instruct python -m exam_digitizer.cli digitize ../demo/sample_real_page.png
# (Windows PowerShell: $env:EXAM_DIGITIZER_HTR = "Qwen/Qwen3-VL-2B-Instruct")
```

No network, no key. First run downloads the model (~4.4 GB); slow on CPU, fast on a GPU.
Also show the fallback: unset `GEMINI_API_KEY`, run with the default backend, and the digitizer
*warns and falls back to TrOCR* instead of failing.

## 4. The review page (1 min)

```bash
python -m exam_digitizer.cli serve --port 8000     # browser: http://127.0.0.1:8000/ui
```

Upload the page, show boxed lines, correct one word, approve, export the reviewed JSON.

## 5. The whole assignment (30 s)

`gemini_full_transcript.md` — all 3 pages of the real assignment transcribed (transcribe + verify
pass per page). Point at the roll number ("JODU") as the reason student IDs stay with the glyph CNN.

## Files here

| file | what it is |
|---|---|
| `sample_real_page.png` | the real handwritten test page (used with the writer's consent) |
| `gemini_full_transcript.md` | Gemini transcript of the full 3-page assignment |
| `../testing/model_comparison.html` | word-level diffs of every model vs the ink |
| `../testing/ground_truth_real_page.txt` | the hand-made ground truth for the test page |
