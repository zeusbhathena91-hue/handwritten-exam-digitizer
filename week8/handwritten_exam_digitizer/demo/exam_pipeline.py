"""End-to-end exam flow:  booklet -> Group 1 digitizer -> Group 2 grader (which screens with Group 12's PromptGuard itself).

    # pages straight from the scanner
    python demo/exam_pipeline.py demo/sample_booklet/T001_p1.png --answer-key demo/answer_key.json \
        --digitizer-url http://127.0.0.1:8002 --grader-url http://127.0.0.1:8000 --roster demo/roster.txt
    # or a booklet already corrected on the review page
    python demo/exam_pipeline.py T001_reviewed.json --answer-key demo/answer_key.json --grader-url http://127.0.0.1:8000

--grader-style group2   (CLI default) the REAL Group-2 API: POST {grader-url}/evaluate-answer?model=v3
--grader-style standin  our own stand-in from partner_standins.py (also supports --guard-url); default of run()

With the real Group-2 service do NOT pass --guard-url: their backend already calls PromptGuard (POST /v1/analyze) before
grading and returns grading_status="human_review" for blocked answers. Those go to our review queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def digitize(pages, url, roster=None, booklet_id=None):
    files = [("pages", (Path(p).name, Path(p).read_bytes(),
              "application/octet-stream")) for p in pages]
    r = requests.post(f"{url}/digitize", files=files, timeout=600,
                      data={"booklet_id": booklet_id or Path(pages[0]).stem, "roster": ",".join(roster or [])})
    r.raise_for_status()
    return r.json()


def _detail(r: requests.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))[:300]
    except Exception:
        return r.text[:300]


def grade_group2(base_url: str, a: dict, key: dict, model: str = "v3") -> dict:
    """One answer -> real Group-2 contract. Returns their JSON (score may be None when PromptGuard sends it to review)."""
    rubric = key.get("rubric") or []
    body = {
        # G2 only echoes this back; the model ignores it
        "question": a["question"],
        "reference_answer": key["reference_answer"],
        "marking_rubric": "; ".join(rubric) if isinstance(rubric, list) and rubric else (rubric or None),
        "student_answer": a["answer_text"],
        # G2 wants an int >= 1
        "maximum_marks": max(1, int(round(key.get("max_marks", 10)))),
    }
    r = requests.post(f"{base_url.rstrip('/')}/evaluate-answer",
                      params={"model": model}, json=body, timeout=120)
    if r.status_code in (502, 504):
        raise RuntimeError(f"Group-2 grader could not reach PromptGuard (HTTP {r.status_code}): {_detail(r)}. "
                           "Start PromptGuard on :8001 or set PROMPTGUARD_URL.")
    if r.status_code == 422:
        raise RuntimeError(
            f"Group-2 rejected the request body (422): {_detail(r)} | sent: {body}")
    r.raise_for_status()
    return r.json()


def grade_standin(url: str, a: dict, key: dict) -> dict:
    r = requests.post(url, timeout=120, json={
        "question": a["question"], "question_text": key.get("question_text", ""), "reference_answer": key["reference_answer"],
        "rubric": key.get("rubric", []), "student_answer": a["answer_text"], "max_marks": key.get("max_marks", 10)})
    r.raise_for_status()
    return r.json()


def run(booklet: dict, answer_key: dict, grader_url: str, guard_url: str | None = None,
        grader_style: str = "standin", model: str = "v3") -> dict:
    # NB: run() keeps the stand-in default (notebook + tests call it that way); the command line defaults to the REAL
    # Group 2 service (--grader-style group2) and library users pass grader_style="group2" explicitly.
    """booklet: digitizer JSON (schema 1.0); answer_key: {"Q1": {question_text, reference_answer, rubric, max_marks}}."""
    graded, queue = [], []
    sid = booklet.get("student_id")
    for a in booklet["answers"]:
        item = {"question": a["question"], "page": a["page"],
                "digitizer_confidence": a["confidence"]}
        approved = (a.get("review") or {}).get(
            "status") in ("approved", "corrected")
        if not sid:
            queue.append({**item, "reason": "student not identified"})
            continue
        if a["needs_review"] and not approved:
            queue.append({**item, "reason": "digitizer: " +
                         (", ".join(a["flags"]) or "low confidence")})
            continue
        # PromptGuard/G2 reject empty text
        if grader_style == "group2" and not a["answer_text"].strip():
            queue.append({**item, "reason": "empty answer text"})
            continue
        if guard_url:                                                                  # stand-in flow only
            g = requests.post(
                guard_url, json={"text": a["answer_text"]}, timeout=60).json()
            if g["risk_level"] != "SAFE":
                queue.append(
                    {**item, "reason": f"guard: {g['risk_level']} {g['attack_type']}", "suspicious_text": g["suspicious_text"]})
                continue
        key = answer_key.get(a["question"])
        if key is None:
            queue.append({**item, "reason": "question not in the answer key"})
            continue
        g = grade_group2(grader_url, a, key, model) if grader_style == "group2" else grade_standin(
            grader_url, a, key)
        # real G2 + PromptGuard said no
        if g.get("grading_status") == "human_review" or g.get("score") is None:
            pg = g.get("promptguard") or {}
            queue.append({**item, "reason": f"promptguard: {pg.get('recommended_action', 'HUMAN_REVIEW')} {pg.get('attack_type') or ''}".strip(),
                          "risk_level": pg.get("risk_level")})
            continue
        gc = g.get("confidence") if g.get("confidence") is not None else 1.0
        graded.append({**item, "score": g["score"], "maximum_score": g["maximum_score"], "feedback": g.get("feedback"),
                       "missing_key_points": g.get("missing_key_points", []), "grader_model": g.get("model"),
                       "grader_confidence": gc, "confidence": round(a["confidence"] * gc, 3), "human_reviewed": approved})
    missing = sorted(set(answer_key) - {a["question"]
                     for a in booklet["answers"]})
    return {"booklet_id": booklet["booklet_id"], "student_id": sid, "graded": graded, "review_queue": queue,
            "questions_not_found": missing, "auto_marks": sum(g["score"] for g in graded),
            "auto_max": sum(g["maximum_score"] for g in graded)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+",
                    help="page images / PDF of one booklet, or one digitizer JSON")
    ap.add_argument("--answer-key", required=True)
    ap.add_argument("--digitizer-url", default="http://127.0.0.1:8002")
    ap.add_argument("--grader-url", default="http://127.0.0.1:8000")
    ap.add_argument("--grader-style",
                    choices=["group2", "standin"], default="group2")
    ap.add_argument("--model", default="v3",
                    choices=["v3", "v2", "v1", "baseline"])
    ap.add_argument("--guard-url", default=None,
                    help="stand-in flow only; real Group 2 screens internally")
    ap.add_argument("--roster", default=None,
                    help="file with one roll number per line")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if len(a.inputs) == 1 and a.inputs[0].endswith(".json"):
        booklet = json.loads(Path(a.inputs[0]).read_text())
    else:
        roster = Path(a.roster).read_text().split() if a.roster else None
        booklet = digitize(a.inputs, a.digitizer_url, roster)
    sheet = run(booklet, json.loads(Path(a.answer_key).read_text()),
                a.grader_url, a.guard_url, a.grader_style, a.model)
    text = json.dumps(sheet, indent=1, ensure_ascii=False)
    Path(a.out).write_text(text) if a.out else print(text)
    print(f"{sheet['booklet_id']}: student {sheet['student_id']} · {len(sheet['graded'])} answers graded "
          f"({sheet['auto_marks']:g}/{sheet['auto_max']:g}) · {len(sheet['review_queue'])} to human review"
          + (f" · not found: {', '.join(sheet['questions_not_found'])}" if sheet["questions_not_found"] else ""))
