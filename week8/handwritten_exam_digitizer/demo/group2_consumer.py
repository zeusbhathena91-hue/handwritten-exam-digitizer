"""How Group 2 (AI Exam Grader) consumes the digitizer — 25 lines, no knowledge of our source code needed.

    python demo/group2_consumer.py T001_p1.png T001_p2.png --url http://127.0.0.1:8000 --roster demo/roster.txt

Reads one booklet through the HTTP API and turns it into grader inputs: one record per (student, question) with the
answer text, plus a review queue for answers the digitizer is not confident about.
"""
import argparse
import json
from pathlib import Path

import requests

ap = argparse.ArgumentParser()
ap.add_argument("pages", nargs="+")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--roster", default=None, help="file with one roll number per line")
a = ap.parse_args()

roster = Path(a.roster).read_text() if a.roster else ""
files = [("pages", (Path(p).name, open(p, "rb"), "image/png")) for p in a.pages]
r = requests.post(f"{a.url}/digitize", files=files, data={"booklet_id": Path(a.pages[0]).stem, "roster": roster})
r.raise_for_status()
booklet = r.json()                                     # schema 1.0, see docs/API.md

grader_inputs = [{"student_id": ans["student_id"], "question": ans["question"], "student_answer": ans["answer_text"],
                  "digitizer_confidence": ans["confidence"]}
                 for ans in booklet["answers"] if not ans["needs_review"]]
review_queue = [{"question": ans["question"], "page": ans["page"], "why": ans["flags"] or ["low confidence"],
                 "confidence": ans["confidence"]} for ans in booklet["answers"] if ans["needs_review"]]

print(f"booklet {booklet['booklet_id']}: student {booklet['student_id']} "
      f"(p={booklet['student_id_confidence']:.2f}), {booklet['pages']} pages, {len(booklet['answers'])} answers")
print(f"→ {len(grader_inputs)} answers go to the grader, {len(review_queue)} to human review")
print(json.dumps({"grader_inputs": grader_inputs, "review_queue": review_queue}, indent=1)[:2000])
