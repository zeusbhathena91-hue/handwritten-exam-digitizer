"""Question segmentation: recognised lines + margin marks → answers with labels, raw confidence and flags."""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

import numpy as np

from .glyphs import GlyphReader
from .layout import Band

INLINE_RE = re.compile(r"^\s*[Qq]\s*[\.\-]?\s*(\d{1,2})\s*[\.\):,]?\s+(.*)$")

# flags an answer can carry (documented in docs/API.md)
FLAGS = {
    "header_inferred": "margin mark present but unreadable; label = previous number + 1",
    "header_low_confidence": "question label posterior < 0.6",
    "duplicate_label": "the same question label was already used in this booklet",
    "inline_header": "no margin marks in the booklet; label taken from a 'Q<n>' at the start of the line text",
    "no_header": "text before the first question number (label 'Q?')",
    "low_recognition_confidence": "mean line confidence < 0.6 (unreadable, non-Latin or very poor scan)",
    "student_id_missing": "no roll number could be read with confidence",
}


def structure(pages: Sequence[dict], bands: Sequence[Band], line_reads: Sequence[tuple], glyphs: GlyphReader,
              booklet_id: str = "", roster: Optional[Sequence[str]] = None, roll_policy: str = "digits-exact") -> dict:
    """pages: [{"pp": Prepped, ...}] ; bands: detected lines in reading order ; line_reads: [(text, conf)] per band."""
    sid, sid_conf, sid_info = glyphs.decode_roll(pages[0]["pp"], roster=roster, policy=roll_policy)
    any_marks = any(b.header is not None for b in bands)
    answers, cur, prev_n = [], None, 0
    for b, (text, conf) in zip(bands, line_reads):
        label, hconf, flags = None, None, []
        if b.header is not None:
            pp = pages[b.page - 1]["pp"]
            x0, y0, x1, y1 = b.header["bbox"]
            x0, y0, x1, y1 = max(0, x0 - 6), max(0, y0 - 6), x1 + 6, y1 + 6
            used = {int(a["question"][1:]) for a in answers if a["question"][1:].isdigit()}
            label, hconf = glyphs.decode_header(pp.gray[y0:y1, x0:x1], prev_n, used)
            if label is None:
                label, hconf, flags = f"Q{prev_n + 1}", 0.3, ["header_inferred"]
            elif hconf < 0.6:
                flags = ["header_low_confidence"]
        elif not any_marks:
            m = INLINE_RE.match(text)
            if m:
                label, hconf, flags, text = f"Q{int(m.group(1))}", 0.5, ["inline_header"], m.group(2)
        if label is not None:
            if any(a["question"] == label for a in answers):
                flags.append("duplicate_label")
            cur = dict(question=label, page=b.page, lines=[], confs=[], hconf=hconf, flags=flags, bands=[],
                       hbox=dict(page=b.page, bbox=[int(v) for v in b.header["bbox"]]) if b.header is not None else None)
            answers.append(cur)
            prev_n = int(label[1:])
        elif cur is None:
            cur = dict(question="Q?", page=b.page, lines=[], confs=[], hconf=0.1, flags=["no_header"], bands=[], hbox=None)
            answers.append(cur)
        cur["lines"].append(text)
        cur["confs"].append(conf)
        cur["bands"].append(b)
    out = []
    for a in answers:
        rc = float(np.mean(a["confs"])) if a["confs"] else 0.0
        raw = rc * math.sqrt(max(a["hconf"], 1e-6))
        flags = a["flags"] + (["low_recognition_confidence"] if rc < 0.6 else [])
        out.append(dict(student_id=sid, question=a["question"], answer_text=" ".join(t for t in a["lines"] if t),
                        page=a["page"], confidence=round(raw, 4), needs_review=bool(raw < 0.6 or flags), flags=flags,
                        _features=dict(mean_conf=rc, min_conf=float(np.min(a["confs"])) if a["confs"] else 0.0,
                                       header_conf=float(a["hconf"]), n_lines=len(a["lines"]),
                                       n_chars=sum(len(t) for t in a["lines"]),
                                       frac_low=float(np.mean(np.array(a["confs"]) < 0.6)) if a["confs"] else 1.0,
                                       inferred=float("header_inferred" in a["flags"])),
                        _header=a["hbox"],
                        _lines=[dict(page=b.page, y0=b.y0, y1=b.y1, x0=b.x0, x1=b.x1, text=t, conf=c)
                                for b, t, c in zip(a["bands"], a["lines"], a["confs"])]))
    return dict(booklet_id=booklet_id, student_id=sid, student_id_confidence=round(sid_conf, 4), student_id_info=sid_info,
                answers=out, pages=len(pages))
