"""THE ASSET: `Digitizer.digitize(pages) -> contract JSON` (scanned answer booklet → question-wise answer text)."""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .config import Config
from .models.confidence import DEFAULT_CALIBRATION, PlattCalibrator, review_policy
from .models.glyphs import GlyphReader
from .models.layout import crop_line, layout_page
from .models.recognition import make_line_reader
from .models.structuring import structure
from .preprocessing.page import load_page, preprocess, render_pdf
from .schema import SCHEMA_VERSION, DigitizeResult

PageInput = Union[str, Path, bytes, np.ndarray]


def expand_pages(pages: Union[PageInput, Sequence[PageInput]]) -> List[Union[bytes, Path, np.ndarray]]:
    """Accept one page, a list of pages, or PDF(s); return a flat list of page images in order."""
    if isinstance(pages, (str, Path, bytes, bytearray, np.ndarray)):
        pages = [pages]
    out = []
    for p in pages:
        if isinstance(p, (str, Path)) and str(p).lower().endswith(".pdf"):
            out += render_pdf(Path(p).read_bytes())
        elif isinstance(p, (bytes, bytearray)) and bytes(p[:5]) == b"%PDF-":
            out += render_pdf(bytes(p))
        else:
            out.append(p)
    return out


class Digitizer:
    """Loads the models once; call `digitize` per booklet. Thread-safe for reads (no state is mutated per call)."""

    def __init__(self, config: Optional[Config] = None, recognizer=None, glyphs: Optional[GlyphReader] = None,
                 calibrator: Optional[PlattCalibrator] = None):
        self.config = config or Config()
        self.device = self.config.resolved_device()
        self.glyphs = glyphs or GlyphReader(self.config.glyph_weights, device=self.device)
        self.recognizer = recognizer or make_line_reader(self.config.htr_model, device=self.device,
                                                         batch_size=self.config.batch_size, cache_dir=self.config.cache_dir)
        cal_path = Path(self.config.calibration or DEFAULT_CALIBRATION)
        if calibrator is not None:
            self.calibrator = calibrator
        elif cal_path.exists():
            self.calibrator = PlattCalibrator.load(cal_path)
        else:
            warnings.warn(f"no calibration file at {cal_path}: confidences are raw (uncalibrated)")
            self.calibrator = None

    @property
    def model_name(self) -> str:
        return f"layout+{getattr(self.recognizer, 'name', 'unknown')}"

    # ------------------------------------------------------------------ internal (notebook-compatible) output
    def analyse(self, pages: Sequence[PageInput]):
        """Preprocessing + layout for all pages."""
        out, bands = [], []
        for pno, p in enumerate(pages, 1):
            pp = preprocess(p)
            b, marks, info = layout_page(pp, pno)
            out.append(dict(pp=pp, info=info, marks=marks))
            bands += b
        return out, bands

    def digitize_raw(self, pages, booklet_id: str = "", roster: Optional[Sequence[str]] = None,
                     roll_policy: Optional[str] = None) -> dict:
        """Full-detail output (with _features, _lines, _header, _pages) — identical to the notebook's `digitize()`."""
        pages = expand_pages(pages)
        roster = self.config.roster if roster is None else roster
        roll_policy = roll_policy or self.config.roll_policy
        t0 = time.time()
        analysed, bands = self.analyse(pages)
        crops = [crop_line(analysed[b.page - 1]["pp"], b) for b in bands]
        t_layout = time.time() - t0
        t1 = time.time()
        reads = self.recognizer.read(crops) if crops else []
        t_rec = getattr(self.recognizer, "last_seconds", time.time() - t1) if crops else 0.0
        t2 = time.time()
        out = structure(analysed, bands, reads, self.glyphs, booklet_id, roster, roll_policy)
        out["runtime_s"] = t_layout + t_rec + (time.time() - t2)
        out["model"] = self.model_name
        out["_pages"] = [dict(angle=pg["pp"].angle, margin=pg["info"]["margin"], virtual=pg["info"]["virtual"],
                              n_marks=len(pg["marks"])) for pg in analysed]
        return out

    # ------------------------------------------------------------------ the contract
    def to_contract(self, raw: dict, include_lines: bool = False, review_threshold: Optional[float] = None) -> dict:
        thr = self.config.review_threshold if review_threshold is None else review_threshold
        out = {"schema_version": SCHEMA_VERSION, "booklet_id": raw["booklet_id"], "student_id": raw["student_id"],
               "student_id_confidence": round(float(raw["student_id_confidence"]), 4), "pages": raw["pages"], "answers": []}
        for a in raw["answers"]:
            p = float(self.calibrator(a["confidence"])) if self.calibrator else float(a["confidence"])
            flags = list(a["flags"]) + ([] if a["student_id"] else ["student_id_missing"])
            rec = dict(student_id=a["student_id"], question=a["question"], answer_text=a["answer_text"], page=a["page"],
                       confidence=round(p, 3), needs_review=review_policy(p, flags, a["student_id"], thr), flags=flags)
            if include_lines:
                rec["lines"] = [dict(page=l["page"], bbox=[l["x0"], l["y0"], l["x1"], l["y1"]], text=l["text"],
                                     confidence=round(float(l["conf"]), 3)) for l in a["_lines"]]
            out["answers"].append(rec)
        out.update(model=raw["model"], confidence_kind="calibrated" if self.calibrator else "raw",
                   runtime_s=round(float(raw["runtime_s"]), 3))
        result = DigitizeResult(**out)      # validates the contract before anything leaves the asset
        return result.model_dump(exclude=None if include_lines else {"answers": {"__all__": {"lines"}}})

    def digitize(self, pages, booklet_id: str = "", roster: Optional[Sequence[str]] = None,
                 roll_policy: Optional[str] = None, include_lines: bool = False,
                 review_threshold: Optional[float] = None) -> dict:
        """pages: image path(s) / bytes / arrays, or a PDF → the contract JSON (schema 1.0) as a dict."""
        raw = self.digitize_raw(pages, booklet_id, roster, roll_policy)
        return self.to_contract(raw, include_lines, review_threshold)
