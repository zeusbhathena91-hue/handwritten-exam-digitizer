# Responsible-AI statement — Handwritten Examination Digitizer (Group 1)

**What it is.** A local OCR asset that turns scanned answer booklets into question-wise text for the university's
grading workflow. It **assists** the examination cell; it does not grade and it never decides on its own when it is
unsure. All numbers below come from the project notebook (frozen synthetic test set: 40 IAM writers, 114 answers).

**Privacy.** Real answer scripts are personal data. The asset runs entirely on university hardware: models are
downloaded once at installation (TrOCR) or shipped in the package (glyph CNN, calibration); afterwards it works with the
network blocked (checked: identical output). The service stores nothing — no pages, no transcripts, no logs of content;
the review page builds its export in the browser. The cloud LLM/VLM systems evaluated in Week 4 are **not** part of the
asset. Development used only public datasets (IAM, MNIST, EMNIST) and a dummy roster; no real student data was used.

**Accuracy and its consequence.** Answer-level CER 0.124, recognition-only CER 0.045, question segmentation F1 0.925,
student-ID accuracy 0.95 with **zero wrong students** (`null` instead of a guess). Fed to a strict keyword grader,
digitization costs 2.3 of 10 marks per answer if everything is graded automatically, **1.5 with the default review
policy** (17 % of answers re-read), 0.7 at threshold 0.8 (37 % re-read). Automatic marking from digitized handwriting
must therefore always be checked by staff, and graders must match rubric points tolerantly.

**Bias.** No significant relation between recognition error and handwriting width, stroke width or slant (all 95 % CIs
include zero, 40 writers). Error is concentrated: the worst 10 % of writers produce 28 % of misread characters; 62 % of
their answers are routed to a human (11 % for others). IAM writers are not our students; the check must be repeated on
real anonymised booklets, and per-group error rates (e.g. by medium of schooling, left-handedness) should be monitored
after deployment. English Latin-script prose only: other scripts are flagged (low confidence), not read.

**Hallucination.** Nothing is generated, summarised or "corrected"; the student's spelling is kept. But the recogniser's
language prior produces fluent wrong words (*footrope → footage*, *dowel → cloud*) — 2.0 per 100 words even on lines it
is ≥ 0.9 sure of; 60 % of these reach the grader without review. This is the main known failure and is stated in the API
documentation.

**Explainability.** Every answer carries a calibrated confidence (P(usable), ECE 0.061), named flags, and on request its
line boxes and line confidences; the review page shows each line on the deskewed scan next to its text.

**Security.** Student writing is untrusted input: an instruction written on the answer sheet is transcribed verbatim, so
every answer passes Group 12's guard before any LLM grader (enforced by `demo/exam_pipeline.py`). The API bounds request
size (413), rejects malformed input before any model runs, renders no model output as HTML, and has no authentication —
it must stay on 127.0.0.1 or behind the university's SSO.

**Human oversight.** `needs_review` routes low-confidence, flagged and unidentified answers to staff; no marks are
produced for a booklet without a student ID; a reviewer approves or corrects on the review page, and the export keeps
the original reading, the reviewer and the time. The review threshold is a per-call setting the exam board owns.

**Limitations.** Expects question numbers in the left margin of a ruled sheet (inline numbers: segmentation F1 0.31); no
diagrams, tables or equations; two-column pages are interleaved; CPU ≈ 3 s/page; evaluated on synthetic booklets built
from real handwriting, not on our university's scripts. Group 2's and Group 12's services were represented by
contract stand-ins.
