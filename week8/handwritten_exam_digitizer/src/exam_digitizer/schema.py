"""The output contract (schema version 1.0) consumed by Group 2's AI Exam Grader. Field names are frozen."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class LineRead(BaseModel):
    """One recognised text line (returned only with include_lines=True; lets a reviewer see what was read where)."""
    page: int = Field(..., ge=1)
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x0, y0, x1, y1] in the deskewed 1654-px-wide page")
    text: str
    confidence: float = Field(..., ge=0, le=1, description="recogniser line confidence (uncalibrated)")


class Answer(BaseModel):
    student_id: Optional[str] = Field(None, description="roll number, or null if it could not be read with confidence")
    question: str = Field(..., description="'Q<n>'; 'Q?' for text before the first question number", examples=["Q3"])
    answer_text: str
    page: int = Field(..., ge=1, description="page on which the answer starts (1-based)")
    confidence: float = Field(..., ge=0, le=1, description="P(transcript is usable: right label and CER ≤ 10 %), calibrated")
    needs_review: bool = Field(..., description="route to a human: low confidence, any flag, or unknown student")
    flags: List[str] = Field(default_factory=list, description="see docs/API.md for the flag vocabulary")
    lines: Optional[List[LineRead]] = Field(None, description="per-line detail, only with include_lines=True")


class DigitizeResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    booklet_id: str
    student_id: Optional[str] = None
    student_id_confidence: float = Field(..., ge=0, le=1)
    pages: int = Field(..., ge=0)
    answers: List[Answer]
    model: str = Field(..., description="recogniser used, e.g. 'layout+microsoft/trocr-small-handwritten'")
    confidence_kind: str = Field("calibrated", description="'calibrated' (Platt) or 'raw' if no calibration file was loaded")
    runtime_s: float = Field(..., ge=0)


class DigitizeOptions(BaseModel):
    """Options shared by the module, the CLI and the API."""
    booklet_id: str = ""
    roster: Optional[List[str]] = Field(None, description="valid roll numbers (NN L NNN); null = no roster check")
    roll_policy: str = Field("digits-exact", description="digits-exact | strict | snap | format-only")
    include_lines: bool = False
    review_threshold: float = Field(0.5, ge=0, le=1)


class DigitizeBase64Request(DigitizeOptions):
    pages: List[str] = Field(..., min_length=1, description="page images (PNG/JPEG) or one PDF, base64-encoded, in order")


EXAMPLE_RESULT = {
    "schema_version": "1.0",
    "booklet_id": "T001",
    "student_id": "23D019",
    "student_id_confidence": 0.999,
    "pages": 1,
    "answers": [
        {"student_id": "23D019", "question": "Q4", "page": 1, "confidence": 0.95, "needs_review": False, "flags": [],
         "answer_text": "let us consider first what would appear to be the simplest case - that of flat fish on the bottom ."},
        {"student_id": "23D019", "question": "Q2", "page": 1, "confidence": 0.41, "needs_review": True,
         "flags": ["header_low_confidence"], "answer_text": "aware of the moving footage before it reaches them ."},
    ],
    "model": "layout+microsoft/trocr-small-handwritten",
    "confidence_kind": "calibrated",
    "runtime_s": 4.45,
}
