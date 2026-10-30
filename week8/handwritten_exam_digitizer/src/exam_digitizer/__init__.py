"""Handwritten Examination Digitizer — Group 1, NLP Course Project 2026 (AI University platform).

    from exam_digitizer import Digitizer
    result = Digitizer().digitize(["page1.png", "page2.png"], booklet_id="T001", roster=["23D001", "23D002"])

`result` follows the frozen output contract (schema 1.0, see `exam_digitizer.schema.DigitizeResult`).
"""
from .config import Config
from .digitizer import Digitizer
from .schema import SCHEMA_VERSION, Answer, DigitizeOptions, DigitizeResult

__version__ = "0.7.0"
__all__ = ["Config", "Digitizer", "DigitizeResult", "DigitizeOptions", "Answer", "SCHEMA_VERSION", "__version__"]
