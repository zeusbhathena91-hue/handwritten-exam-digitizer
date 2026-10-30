"""Page preprocessing: load → crop to paper → background normalisation → deskew → rule/margin detection → ink mask.

Every page is brought to a standard width (1654 px ≈ A4 at 200 dpi) so that the pixel thresholds used by the layout
analysis mean the same physical size on every input. All functions are pure (no module state).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

STD_W = 1654  # standard page width in pixels (A4 at 200 dpi)

PageSource = Union[str, Path, bytes, np.ndarray]


def decode_image_bytes(data: bytes) -> np.ndarray:
    """PNG/JPEG/TIFF bytes -> grey uint8 array (raises ValueError if the bytes are not an image)."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("could not decode image bytes (expected PNG/JPEG/TIFF/BMP)")
    return arr


def render_pdf(data: bytes, dpi: int = 200) -> list[np.ndarray]:
    """PDF bytes -> one grey page image per PDF page (needs the optional dependency pypdfium2)."""
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover
        raise ImportError("PDF input needs `pip install pypdfium2`") from e
    pdf = pdfium.PdfDocument(io.BytesIO(data))
    pages = []
    for i in range(len(pdf)):
        bitmap = pdf[i].render(scale=dpi / 72.0, grayscale=True)
        pages.append(np.asarray(bitmap.to_pil().convert("L")))
    return pages


def load_page(src: PageSource) -> np.ndarray:
    """Path / bytes / array -> grey uint8 page resized to the standard width."""
    if isinstance(src, (str, Path)):
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cannot read image {src}")
    elif isinstance(src, (bytes, bytearray)):
        img = decode_image_bytes(bytes(src))
    else:
        img = np.asarray(src)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[1] != STD_W:
        s = STD_W / img.shape[1]
        img = cv2.resize(img, (STD_W, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return img


def rotate(img: np.ndarray, angle: float, border: float = 255) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


def crop_to_paper(g: np.ndarray) -> np.ndarray:
    """If the page lies on a darker background (crooked scan, phone photo on a table), rotate the paper upright and
    crop to it. Pages that already fill the image are returned unchanged."""
    h, w = g.shape
    fr = max(4, min(h, w) // 60)
    borders = [g[:fr], g[-fr:], g[:, :fr], g[:, -fr:]]
    frame = np.concatenate([b.ravel() for b in borders])
    if sum(np.median(b) < 170 for b in borders) < 3:      # background must be visible on at least three sides
        return g
    small = cv2.GaussianBlur(cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), (5, 5), 0)
    _, m = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 0.3 * m.size:
        return g
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    if cv2.contourArea(c) < 0.92 * rect[1][0] * rect[1][1]:     # paper outline is not a clean rectangle: do not crop
        return g
    pts = cv2.boxPoints(rect)
    angs = [math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(pts, np.roll(pts, -1, 0))]
    angs = [((x + 45) % 90) - 45 for x in angs]
    ang = float(np.median(angs))
    bed = float(np.median(frame))
    g = rotate(g, ang, border=bed)
    small = cv2.GaussianBlur(cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), (5, 5), 0)
    _, m = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, bw, bh = [4 * v for v in cv2.boundingRect(max(cnts, key=cv2.contourArea))]
    return load_page(g[y + 6:y + bh - 6, x + 6:x + bw - 6])


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Divide by a median-filtered paper estimate (removes shadows / uneven light), denoise if the sensor noise is
    strong, then stretch contrast so the darkest ink is 0 (restores faded pages)."""
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 21), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0
    resid = norm - cv2.medianBlur(np.clip(norm, 0, 255).astype(np.uint8), 3)
    sigma = 1.4826 * np.median(np.abs(resid[norm > 200])) if (norm > 200).any() else 0.0
    if sigma > 8:                                               # strong sensor noise: edge-preserving smoothing
        norm = cv2.bilateralFilter(np.clip(norm, 0, 255).astype(np.uint8), 7, float(2 * sigma), 5).astype(np.float32)
    lo = np.percentile(norm, 0.3)
    norm = (norm - lo) / max(255.0 - lo, 1.0) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def estimate_skew(norm: np.ndarray, max_angle: float = 10.0) -> float:
    """Projection-profile deskew: the angle (±max_angle°, 0.25° then 0.05° steps) that maximises row-sum variance."""
    small = cv2.resize(norm, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    ink = (cv2.medianBlur(small, 3) < 225).astype(np.float32) * 255

    def score(a):
        return np.var(rotate(ink, -a, border=0).sum(1))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.25)
    a0 = coarse[int(np.argmax([score(a) for a in coarse]))]
    fine = np.arange(a0 - 0.25, a0 + 0.25 + 1e-6, 0.05)
    return float(fine[int(np.argmax([score(a) for a in fine]))])


@dataclass
class Prepped:
    """A preprocessed page."""
    gray: np.ndarray        # deskewed, background-normalised grey (for recognition)
    clean: np.ndarray       # same with ruled lines / margin removed
    ink: np.ndarray         # binary ink mask without rules (for layout)
    angle: float            # estimated skew that was removed (degrees)
    margin_x: object        # x of the printed margin line, or None
    rules_y: list           # y of the printed ruled lines
    header_bottom: object   # y of the printed header separator, or None


def preprocess(src: PageSource) -> Prepped:
    g = crop_to_paper(load_page(src))
    norm = normalize_background(g)
    angle = estimate_skew(norm)
    norm = rotate(norm, -angle)
    H, W = norm.shape
    light = (norm < 235).astype(np.uint8)
    hl = cv2.morphologyEx(light, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W // 12, 1)))
    vl = cv2.morphologyEx(light, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 10)))
    # rule rows (long horizontal runs) and the printed header separator (dark, in the top 20 %)
    row_frac = hl.sum(1) / W
    rows = np.where(row_frac > 0.45)[0]
    groups = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1) if len(rows) else []
    lines_y = [(int(gp.mean()), float(norm[gp.min():gp.max() + 1][hl[gp.min():gp.max() + 1] > 0].mean()))
               for gp in groups if len(gp)]
    header_bottom = max([y for y, v in lines_y if y < 0.2 * H and v < 160], default=None)
    rules_y = [y for y, v in lines_y if v >= 160 and (header_bottom is None or y > header_bottom)]
    # margin = strongest long vertical line in the left third
    col = vl.sum(0)
    x_lo, x_hi = int(0.04 * W), int(0.35 * W)
    xm = x_lo + int(np.argmax(col[x_lo:x_hi]))
    margin_x = xm if col[xm] > 0.30 * H else None
    # remove rules/margin where they are lighter than ink
    hmask = cv2.dilate(hl, np.ones((5, 3), np.uint8)) > 0
    vmask = cv2.dilate(vl, np.ones((3, 7), np.uint8)) > 0
    rule_level = float(np.median(norm[(hl > 0)])) if hl.any() else 200.0
    clean = norm.copy()
    clean[hmask & (norm > rule_level - 45)] = 255      # light rules: remove, keep ink that crosses them
    clean[vmask] = 255                                   # printed margin / box sides
    if header_bottom is not None:                        # dark printed separator line
        band = slice(max(0, header_bottom - 6), header_bottom + 7)
        clean[band][hmask[band]] = 255
    t, _ = cv2.threshold(cv2.medianBlur(clean, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (cv2.medianBlur(clean, 3) < min(max(t, 90), 185)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    small_cc = np.where(st[:, cv2.CC_STAT_AREA] < 12)[0]
    ink[np.isin(lab, small_cc)] = 0
    return Prepped(norm, clean, ink, angle, margin_x, rules_y, header_bottom)
