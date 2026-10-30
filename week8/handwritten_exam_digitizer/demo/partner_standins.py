"""Stand-ins for two partner services of the AI University platform, built ONLY from the contracts in the project
handout, so that the end-to-end exam pipeline can be run before the partner groups' services are deployed.
Replace them by pointing `exam_pipeline.py --grader-url/--guard-url` at the real services; nothing else changes.

    python demo/partner_standins.py --port 8100
    POST /evaluate-answer   Group 2 (AI Exam Answer Grader) contract — baseline "Approach A": lexical similarity to
                            the reference answer + rubric-term coverage. NOT Group 2's model.
    POST /check-input       Group 12 (Malicious Input Detection) contract — a keyword/regex baseline. NOT Group 12's model.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

STOP = set("a an the and or but if of to in on at by for with from as is are was were be been being it its this that these "
           "those there their they them he she his her we our you your i me my not no so than then too very can could "
           "would should will shall may might must do does did done have has had into out up down over under again "
           "which who whom what when where why how all any both each few more most other some such only own same".split())


def content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def cosine(a: Counter, b: Counter) -> float:
    num = sum(a[k] * b[k] for k in a.keys() & b.keys())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


class GradeRequest(BaseModel):
    question: str = Field(..., examples=["Q4"])
    question_text: str = ""
    reference_answer: str
    rubric: List[str] = Field(default_factory=list, description="key points/terms the answer must contain")
    student_answer: str
    max_marks: float = 10


class GradeResponse(BaseModel):
    question: str
    score: float
    maximum_score: float
    feedback: str
    confidence: float


def grade(req: GradeRequest) -> GradeResponse:
    stu = content_words(req.student_answer)
    sim = cosine(Counter(stu), Counter(content_words(req.reference_answer)))
    terms = [t.lower() for t in req.rubric]
    hit = [t for t in terms if all(w in stu for w in content_words(t))]
    cov = len(hit) / len(terms) if terms else sim
    score = round(2 * req.max_marks * (0.5 * sim + 0.5 * cov)) / 2          # nearest half mark
    missing = [t for t in terms if t not in hit]
    fb = (f"Covers {len(hit)}/{len(terms)} rubric points" + (f"; missing: {', '.join(missing)}" if missing else "") +
          f". Similarity to the reference answer {sim:.2f}.") if terms else f"Similarity to the reference answer {sim:.2f}."
    return GradeResponse(question=req.question, score=score, maximum_score=req.max_marks, feedback=fb,
                         confidence=round(1 - abs(sim - cov), 3))


INJECTION = {   # attack_type -> patterns (Group 12's handout lists instruction override, prompt leaking, …)
    "instruction_override": r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rubric|rules|above|previous)\b",
    "grade_manipulation": r"\b(award|give|assign)\b.{0,30}\b(full|maximum|max|10|ten|all)\b.{0,15}\b(marks?|points?|score)\b",
    "role_play": r"\byou are (now )?(an?|the)\b.{0,40}\b(grader|examiner|assistant|ai|model)\b",
    "prompt_leak": r"\b(system prompt|reveal (your|the) (prompt|instructions))\b",
}


class CheckRequest(BaseModel):
    text: str


def check(req: CheckRequest) -> dict:
    found = [(k, m.group(0)) for k, p in INJECTION.items() for m in re.finditer(p, req.text, re.I)]
    level = "HIGH_RISK" if len({k for k, _ in found}) > 1 or any(k == "grade_manipulation" for k, _ in found) else \
        "SUSPICIOUS" if found else "SAFE"
    return {"attack_detected": bool(found), "risk_level": level, "attack_type": found[0][0] if found else None,
            "confidence_score": 0.9 if level == "HIGH_RISK" else 0.6 if found else 0.8,
            "suspicious_text": [s for _, s in found],
            "recommended_action": {"SAFE": "ALLOW", "SUSPICIOUS": "HUMAN_REVIEW", "HIGH_RISK": "BLOCK"}[level]}


app = FastAPI(title="Partner-service stand-ins (Group 2 grader, Group 12 guard)")
app.post("/evaluate-answer", response_model=GradeResponse)(grade)
app.post("/check-input")(check)

if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    uvicorn.run(app, host="127.0.0.1", port=ap.parse_args().port)
