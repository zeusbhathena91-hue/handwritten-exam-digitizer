import pytest
from pydantic import ValidationError

from exam_digitizer.schema import EXAMPLE_RESULT, SCHEMA_VERSION, Answer, DigitizeResult

FROZEN_ANSWER_FIELDS = {"student_id", "question", "answer_text", "page", "confidence", "needs_review", "flags"}
FROZEN_TOP_FIELDS = {"booklet_id", "student_id", "student_id_confidence", "answers", "pages", "model", "runtime_s"}


def test_example_validates():
    r = DigitizeResult(**EXAMPLE_RESULT)
    assert r.schema_version == SCHEMA_VERSION == "1.0"
    assert len(r.answers) == 2 and r.answers[1].needs_review


def test_contract_fields_are_present():
    assert FROZEN_ANSWER_FIELDS <= set(Answer.model_fields)
    assert FROZEN_TOP_FIELDS <= set(DigitizeResult.model_fields)


def test_confidence_bounds_enforced():
    bad = dict(EXAMPLE_RESULT, answers=[dict(EXAMPLE_RESULT["answers"][0], confidence=1.7)])
    with pytest.raises(ValidationError):
        DigitizeResult(**bad)


def test_json_schema_is_exportable():
    s = DigitizeResult.model_json_schema()
    assert "answers" in s["properties"] and "Answer" in s["$defs"]
