# API / module documentation — Handwritten Examination Digitizer (schema 1.0)

This page is everything another application needs to call the component. It is the answer to the assignment's
"reusable asset" test: *If another group wanted to use our NLP component tomorrow, what would they need?*

## 1. Interfaces

| Interface | How to call |
|---|---|
| Python module | `from exam_digitizer import Digitizer`; `Digitizer().digitize(pages, booklet_id="", roster=None, roll_policy="digits-exact", include_lines=False, review_threshold=0.5)` |
| Command line | `python -m exam_digitizer.cli digitize PAGE... [--booklet-id ID] [--roster FILE_OR_LIST] [--roll-policy P] [--include-lines] [--out result.json]` |
| HTTP (FastAPI) | `python -m exam_digitizer.cli serve --port 8000` then `POST /digitize` (multipart) or `POST /digitize-base64` (JSON) |
| Review page (browser) | `http://127.0.0.1:8000/ui` — upload a booklet, check every line against the scan, correct, approve, export (section 9) |

All three return the same JSON document (section 3). The HTTP service also exposes `GET /health`, `GET /schema`
(JSON schema + example), `GET /docs` (interactive OpenAPI UI), `GET /ui` (review page) and `POST /preview` (the deskewed
pages that the line boxes refer to, section 9).

## 2. Input

| Parameter | Type | Meaning |
|---|---|---|
| `pages` | list of files / bytes / arrays, or one PDF | the pages of **one** booklet, in reading order. PNG, JPEG, TIFF, BMP; any resolution (pages are rescaled to 1654 px width ≈ A4 at 200 dpi; 150–300 dpi scans and phone photos work). A PDF is rendered at 200 dpi. |
| `booklet_id` | string | echoed back; use your own identifier (scan file name, barcode, …) |
| `roster` | list of roll numbers, format `NN L NNN` (e.g. `23D001`) | the valid student IDs for this exam. Strongly recommended: without it the ID is returned unverified (`format-only`). HTTP: comma-separated string or JSON list; a server-wide default can be set with `EXAM_DIGITIZER_ROSTER=/path/roster.txt`. |
| `roll_policy` | `digits-exact` (default) · `strict` · `snap` · `format-only` | how the roll number read from the six boxes is matched to the roster (section 5) |
| `include_lines` | bool | also return every recognised line with its box and line confidence (for review UIs) |
| `review_threshold` | float, default 0.5 | `needs_review` is true below this calibrated confidence |

### HTTP examples

```bash
# multipart (files)
curl -s -F "pages=@T001_p1.png" -F "pages=@T001_p2.png" -F "booklet_id=T001" \
     -F "roster=23D001,23D019,23D046" -F "include_lines=false" http://127.0.0.1:8000/digitize

# JSON (base64 pages), e.g. from another service
python - <<'EOF'
import base64, json, requests
pages = [base64.b64encode(open(p, "rb").read()).decode() for p in ["T001_p1.png", "T001_p2.png"]]
r = requests.post("http://127.0.0.1:8000/digitize-base64",
                  json={"booklet_id": "T001", "pages": pages, "roster": ["23D001", "23D019"]})
print(json.dumps(r.json(), indent=1))
EOF
```

## 3. Output (schema 1.0) — field names are frozen

```json
{
  "schema_version": "1.0",
  "booklet_id": "T001",
  "student_id": "23D019",                 // null = could not be read with confidence; NEVER a guess
  "student_id_confidence": 0.999,
  "pages": 2,
  "answers": [
    {
      "student_id": "23D019",
      "question": "Q4",                   // "Q<n>"; "Q?" = text before the first question number
      "answer_text": "let us consider first what would appear to be ...",
      "page": 1,                          // page on which the answer starts (1-based)
      "confidence": 0.95,                 // calibrated P(transcript usable) — see section 4
      "needs_review": false,              // route to a human?
      "flags": [],                        // section 4
      "lines": [ {"page": 1, "bbox": [245, 402, 1570, 455], "text": "...", "confidence": 0.97} ]   // only with include_lines
    }
  ],
  "model": "layout+microsoft/trocr-small-handwritten",
  "confidence_kind": "calibrated",        // "raw" if the calibration file is missing
  "runtime_s": 4.45
}
```

Answers are in reading order. The same question label can appear twice (flag `duplicate_label`); consumers should
decide whether to merge or review. `answer_text` joins the lines of an answer with single spaces and keeps the
student's spelling — nothing is corrected or completed.

The machine-readable schema is `exam_digitizer.schema.DigitizeResult.model_json_schema()` or `GET /schema`.

## 4. Confidence, flags and human review

`confidence` is a **calibrated probability** that the transcript is usable (correct question label and character error
rate ≤ 10 %). It is Platt scaling of the pipeline's raw score (mean TrOCR line probability × √question-label posterior),
fitted in Week 5 on the frozen test set (out-of-fold AUROC 0.855, expected calibration error 0.061). Operating points
measured there:

| auto-accept the most confident … | unusable among accepted | sent to a human |
|---|---|---|
| 100 % | 19.5 % | 0 of 113 |
| 90 % | 14.7 % | 11 |
| 80 % | 11.1 % | 23 |
| 70 % | 5.1 % | 34 |

`needs_review = confidence < review_threshold OR flags non-empty OR student_id is null`.

| flag | meaning |
|---|---|
| `header_inferred` | a margin mark was found but could not be read; label = previous number + 1 |
| `header_low_confidence` | question-label posterior < 0.6 (e.g. looped `q`, `2`/`3`/`7` confusion) |
| `duplicate_label` | the label was already used in this booklet |
| `inline_header` | the booklet has no margin marks; the label came from a `Q<n>` at the start of the line text |
| `no_header` | text before the first question number (label `Q?`) |
| `low_recognition_confidence` | mean line confidence < 0.6: unreadable, non-Latin script, heavy blur/noise |
| `student_id_missing` | no roll number could be read with confidence (`student_id` is null) |

Known failure modes that confidence does **not** catch: fluent real-word substitutions (`dowel → cloud`) with high
confidence, and a confidently misread digit that happens to be another student's valid roll number.

## 5. Roll-number policies

| policy | behaviour | test-set accuracy / wrong student | off-roster student assigned to someone else (8 booklets) |
|---|---|---|---|
| `digits-exact` (default) | the five digits read must match a roster entry exactly; otherwise `null` | 0.95 / 0.00 | 1 of 8 |
| `strict` | the whole string read must be on the roster | 0.80 / 0.00 | 0 of 8 |
| `snap` | nearest roster entry, never abstains | 1.00 / 0.00 | 8 of 8 |
| `format-only` | no roster: the string read if it matches `NN L NNN` and the digit confidence ≥ 0.3 | — | — |

A `null` costs a reviewer seconds; a wrong ID costs a student their marks — hence the default.

## 6. Errors (HTTP)

| status | when |
|---|---|
| 400 | a page could not be decoded (not an image/PDF), or no pages |
| 413 | more than `EXAM_DIGITIZER_MAX_PAGES` files/pages (default 40) or more than `EXAM_DIGITIZER_MAX_MB` per request (default 60 MB); checked before any model runs |
| 422 | invalid parameter (unknown `roll_policy`, bad JSON body) |
| 500 | unexpected failure — please report with the booklet |

## 7. Performance and requirements

CPU (2 vCPU): ≈ 2.8 s per page with `trocr-small-handwritten`; GPU: well under 1 s/page. Memory ≈ 1.5 GB.
Models: TrOCR-small (62 M parameters, downloaded from the Hugging Face Hub on first use, then cached) and the packaged
glyph CNN (0.4 M). Set `EXAM_DIGITIZER_HTR=microsoft/trocr-base-handwritten` for the larger model (CER −0.007, 7× slower
on CPU). Everything runs locally; no network is needed after the first model download (checked in the Week-7 notebook:
the CLI runs with `HF_HUB_OFFLINE=1` and all proxies pointed at a closed port).

The service has **no authentication**: it binds to 127.0.0.1 by default. To expose it inside the university, put it
behind the exam cell's reverse proxy / single sign-on; never on the public internet.

## 8. Versioning

`schema_version` changes only if a field is renamed or removed (never planned); new optional fields may be added.
The package version (`exam_digitizer.__version__`) tracks the project week (0.7.0 = Week 7: review page, `/preview`,
request limits; the response is unchanged).

## 9. Review page and the reviewed document

`GET /ui` serves a single HTML page (no external scripts) that calls `/digitize` with `include_lines=true` and
`/preview`. A reviewer sees the deskewed scan with every recognised line boxed (green ≥ 0.9, amber 0.6–0.9, red < 0.6
line confidence), edits question labels, text and a missing student ID, approves answers and exports a JSON file. The
export is a valid schema-1.0 document plus two optional blocks that consumers may ignore:

```json
{ "...": "all schema-1.0 fields",
  "answers": [ { "...": "...", "needs_review": false,
                 "review": {"status": "corrected", "original": {"question": "Q2", "answer_text": "aware of the moving footage ..."}} } ],
  "review": {"reviewer": "exam-cell-01", "reviewed_at": "2026-10-23T10:15:02Z", "approved": 4, "corrected": 1, "student_id_changed": false} }
```

`review.status` is `approved` (checked, unchanged) or `corrected`; approved answers get `needs_review = false`. Answers
the reviewer did not approve keep their original `needs_review`. `demo/exam_pipeline.py` grades an approved answer even
if the digitizer had flagged it. `POST /preview` (multipart `pages`, optional `scale` 0.1–1, default 0.5) returns
`{"scale": 0.5, "pages": ["data:image/jpeg;base64,…"]}`; multiply a contract `bbox` by `scale` to draw it on the image.
