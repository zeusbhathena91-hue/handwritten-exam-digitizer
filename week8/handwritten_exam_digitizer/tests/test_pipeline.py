"""End-to-end tests of the asset (need TrOCR + the packaged glyph weights; skipped otherwise)."""
import io

import numpy as np
from PIL import Image

from exam_digitizer.schema import DigitizeResult
from tests.conftest import make_page, slow

ROSTER = [f"23D{i:03d}" for i in range(1, 31)]


@slow
def test_end_to_end_contract(digitizer, page_file):
    res = digitizer.digitize([page_file], booklet_id="T001", roster=ROSTER)
    DigitizeResult(**res)                                   # validates
    assert res["schema_version"] == "1.0" and res["booklet_id"] == "T001" and res["pages"] == 1
    assert res["confidence_kind"] == "calibrated"
    assert len(res["answers"]) >= 1
    for a in res["answers"]:
        assert 0 <= a["confidence"] <= 1 and isinstance(a["flags"], list) and a["question"].startswith("Q")
        assert a["page"] == 1 and len(a["answer_text"]) > 0 and "lines" not in a
    assert res["student_id"] in ROSTER or res["student_id"] is None        # never a guess outside the roster
    if res["student_id"] is None:
        assert all("student_id_missing" in a["flags"] and a["needs_review"] for a in res["answers"])


@slow
def test_include_lines_and_bytes_input(digitizer, page):
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, "PNG")
    res = digitizer.digitize([buf.getvalue()], include_lines=True)
    assert all("lines" in a and len(a["lines"]) >= 1 for a in res["answers"])
    line = res["answers"][0]["lines"][0]
    assert len(line["bbox"]) == 4 and 0 <= line["confidence"] <= 1


@slow
def test_blank_page_returns_no_answers(digitizer):
    res = digitizer.digitize([make_page(blank=True)], booklet_id="BLANK")
    assert res["answers"] == [] and res["student_id"] is None and res["pages"] == 1


@slow
def test_two_pages_are_read_in_order(digitizer):
    res = digitizer.digitize([make_page(page_no=1), make_page(page_no=2, n_answers=1)], booklet_id="TWO")
    assert res["pages"] == 2
    assert [a["page"] for a in res["answers"]] == sorted(a["page"] for a in res["answers"])


@slow
def test_raw_output_matches_contract(digitizer, page_file):
    raw = digitizer.digitize_raw([page_file], booklet_id="T001", roster=ROSTER)
    con = digitizer.to_contract(raw)
    assert [a["question"] for a in raw["answers"]] == [a["question"] for a in con["answers"]]
    assert all("_lines" in a and "_features" in a for a in raw["answers"])


@slow
def test_pdf_input(digitizer, page, tmp_path):
    import pytest
    pytest.importorskip("pypdfium2")
    pdf = tmp_path / "booklet.pdf"
    Image.fromarray(page).save(pdf, "PDF", resolution=200)
    res = digitizer.digitize(pdf, booklet_id="PDF")
    assert res["pages"] == 1 and len(res["answers"]) >= 1
