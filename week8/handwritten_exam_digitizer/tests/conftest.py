"""Shared fixtures: a synthetic answer-sheet page drawn with a printed font (no datasets, no downloads needed)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

P_W, P_H = 1654, 2339
MARGIN_X, BODY_TOP, BODY_BOTTOM = 215, 300, 2215
ROLL_X0, ROLL_Y0, ROLL_W, ROLL_H, ROLL_STEP = 330, 128, 70, 82, 84
FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
LINES = ["the quick brown fox jumps over the lazy dog", "a university generates enormous amounts of text",
         "handwritten answers must become machine readable", "before automated assessment can happen",
         "confidence scores route answers to a human reviewer", "every answer carries its page and its flags"]


def make_page(n_answers: int = 2, lines_per_answer: int = 2, roll: str = "23D007", page_no: int = 1, pitch: int = 94,
              blank: bool = False, marks: bool = True):
    """Grey uint8 page in the answer-sheet template, printed text, question numbers in the margin."""
    pil = Image.new("L", (P_W, P_H), 250)
    d = ImageDraw.Draw(pil)
    f = lambda s: ImageFont.truetype(FONT, s)
    d.text((70, 52), "ANSWER BOOKLET  ·  Natural Language Processing  ·  End-Semester Examination", font=f(30), fill=55)
    d.text((70, 150), "Roll No.", font=f(32), fill=55)
    for i in range(6):
        x = ROLL_X0 + i * ROLL_STEP
        d.rectangle([x, ROLL_Y0, x + ROLL_W, ROLL_Y0 + ROLL_H], outline=70, width=3)
        if not blank:
            d.text((x + 18, ROLL_Y0 + 14), roll[i], font=f(48), fill=30)
    d.line([(60, 262), (P_W - 60, 262)], fill=80, width=4)
    rules = list(range(BODY_TOP + pitch, BODY_BOTTOM, pitch))
    for y in rules:
        d.line([(60, y), (P_W - 60, y)], fill=205, width=2)
    d.line([(MARGIN_X, 270), (MARGIN_X, P_H - 95)], fill=150, width=3)
    d.text((P_W // 2 - 45, P_H - 78), f"Page {page_no}", font=f(28), fill=70)
    if not blank:
        r = 0
        for q in range(n_answers):
            for j in range(lines_per_answer):
                y = rules[r] - 48
                if j == 0 and marks:
                    d.text((110, y), f"Q{q + 1}", font=f(40), fill=30)
                d.text((MARGIN_X + 40, y), LINES[(q * lines_per_answer + j) % len(LINES)], font=f(36), fill=30)
                r += 1
            r += 1
    return np.asarray(pil, np.uint8).copy()


@pytest.fixture(scope="session")
def page():
    return make_page()


@pytest.fixture(scope="session")
def page_file(tmp_path_factory, page):
    p = tmp_path_factory.mktemp("pages") / "T001_p1.png"
    Image.fromarray(page).save(p)
    return p


def _models_available() -> bool:
    if os.environ.get("EXAM_DIGITIZER_SKIP_SLOW"):
        return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    from exam_digitizer.models.glyphs import DEFAULT_WEIGHTS
    return DEFAULT_WEIGHTS.exists()


slow = pytest.mark.skipif(not _models_available(), reason="needs torch/transformers, packaged glyph weights and TrOCR "
                                                            "(set EXAM_DIGITIZER_SKIP_SLOW=1 to skip explicitly)")


@pytest.fixture(scope="session")
def digitizer():
    from exam_digitizer import Config, Digitizer
    # pinned to the local TrOCR model: tests must run offline and never call the Gemini API
    return Digitizer(Config(device="cpu", htr_model="microsoft/trocr-small-handwritten"))
