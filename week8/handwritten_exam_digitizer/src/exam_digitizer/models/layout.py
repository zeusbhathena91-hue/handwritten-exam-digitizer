"""Layout analysis: text lines (horizontal projection of the ink mask) and margin marks (question numbers)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..preprocessing.page import Prepped


@dataclass
class Band:
    """One detected text line."""
    page: int
    y0: int
    y1: int
    x0: int
    x1: int
    header: object = None   # dict with bbox when a margin mark is attached


def find_bands(ink: np.ndarray, x_lo: int, x_hi: int, y_lo: int, y_hi: int):
    """Horizontal projection → runs above an adaptive threshold; tall runs (touching lines) are split at the
    projection minimum. Returns (bands, median line height)."""
    region = ink[y_lo:y_hi, x_lo:x_hi]
    prof = np.convolve(region.sum(1).astype(float), np.ones(9) / 9, "same")
    if prof.max() <= 0:
        return [], 0
    thr = max(3.0, 0.07 * np.percentile(prof[prof > 0], 95))
    on = prof > thr
    runs, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:                                  # close tiny gaps (dots, i-dots)
        if merged and r[0] - merged[-1][1] < 10:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= 18 and region[r[0]:r[1]].sum() > 400]
    if not runs:
        return [], 0
    h_med = float(np.median([r[1] - r[0] for r in runs]))
    out = []

    def split(a, b):
        if b - a > 1.65 * h_med:
            lo, hi = a + int(0.35 * h_med), b - int(0.35 * h_med)
            if hi > lo:
                c = lo + int(np.argmin(prof[lo:hi]))
                split(a, c)
                split(c, b)
                return
        out.append((a, b))

    for a, b in runs:
        split(a, b)
    bands = []
    for a, b in out:
        cols = np.where(np.convolve(region[a:b].sum(0).astype(float), np.ones(5), "same") > 1)[0]
        if len(cols) == 0:
            continue
        bands.append((y_lo + a, y_lo + b, x_lo + int(cols[0]), x_lo + int(cols[-1])))
    return bands, h_med


def group_marks(ink: np.ndarray, x_lo: int, x_hi: int, y_lo: int, y_hi: int, h_med: float):
    """Ink components left of the margin, grouped vertically into marks."""
    if x_hi - x_lo < 20:
        return []
    sub = ink[y_lo:y_hi, x_lo:x_hi]
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
    comps = [(st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3], st[i, 4]) for i in range(1, n) if st[i, 4] >= 25]
    comps.sort(key=lambda c: (c[1] + c[3]) / 2)
    marks = []
    for c in comps:
        cy = (c[1] + c[3]) / 2
        if marks and abs(cy - marks[-1]["cy"]) < 0.45 * max(h_med, 40):
            m = marks[-1]
            m["box"] = [min(m["box"][0], c[0]), min(m["box"][1], c[1]), max(m["box"][2], c[2]), max(m["box"][3], c[3])]
            m["area"] += c[4]
            m["cy"] = (m["box"][1] + m["box"][3]) / 2
        else:
            marks.append({"box": list(c[:4]), "area": c[4], "cy": cy})
    out = []
    for m in marks:
        x0, y0, x1, y1 = m["box"]
        if y1 - y0 >= 15 and m["area"] >= 120:
            out.append({"bbox": [int(x_lo + x0), int(y_lo + y0), int(x_lo + x1), int(y_lo + y1)], "cy": y_lo + m["cy"]})
    return out


def layout_page(pp: Prepped, page_no: int):
    """Text lines + margin marks of one page. If no printed margin exists, a virtual margin is placed just left of
    the median line start. Returns (bands, marks, info)."""
    H, W = pp.ink.shape
    body_top = (pp.header_bottom + 10) if pp.header_bottom is not None else 30
    body_bot = int(0.962 * H)                       # above the printed page number
    x_right = W - 40
    if pp.margin_x is not None:
        margin, virtual = pp.margin_x, False
        raw, h_med = find_bands(pp.ink, margin + 8, x_right, body_top, body_bot)
    else:                                            # virtual margin from the median line start
        raw, h_med = find_bands(pp.ink, 15, x_right, body_top, body_bot)
        starts = np.array([b[2] for b in raw]) if raw else np.array([])
        x_body = float(np.median(starts)) if len(starts) else 15
        consistent = len(starts) and np.mean(np.abs(starts - x_body) < 45) >= 0.5
        margin, virtual = (int(x_body - 12) if consistent and x_body > 120 else None), True
        if margin is not None:
            raw = [(a, b, max(x0, margin + 4), x1) for a, b, x0, x1 in raw]
            raw2 = []
            for a, b, x0, x1 in raw:                 # re-trim line start to the first ink right of the virtual margin
                cols = np.where(pp.ink[a:b, margin + 4:x1 + 1].sum(0) > 1)[0]
                if len(cols):
                    raw2.append((a, b, margin + 4 + int(cols[0]), x1))
            raw = raw2
    # the bottom strip holds the printed page number, but a full page's last written line can end up
    # there too: keep bottom-strip bands that are wide like a text line (a page number is short)
    x_lo = (margin + 8) if margin is not None else 15
    tail, _ = find_bands(pp.ink, x_lo, x_right, body_bot, H - 4)
    raw += [(a, b, x0, x1) for a, b, x0, x1 in tail if x1 - x0 >= 0.25 * (x_right - x_lo)]
    bands = [Band(page_no, a, b, x0, x1) for a, b, x0, x1 in raw]
    marks = group_marks(pp.ink, 15, (margin - (6 if not virtual else 2)), body_top, body_bot, h_med) if margin is not None else []
    for mk in marks:
        if not bands:
            break
        d = [abs(mk["cy"] - (b.y0 + b.y1) / 2) for b in bands]
        j = int(np.argmin(d))
        if d[j] < 0.8 * max(h_med, 40) and bands[j].header is None:
            bands[j].header = mk
    return bands, marks, dict(margin=margin, virtual=virtual, h_med=h_med, body=(body_top, body_bot))


def crop_line(pp: Prepped, b: Band, pad_x: int = 12, pad_y: int = 8) -> np.ndarray:
    H, W = pp.clean.shape
    return pp.clean[max(0, b.y0 - pad_y):min(H, b.y1 + pad_y), max(0, b.x0 - pad_x):min(W, b.x1 + pad_x)]
