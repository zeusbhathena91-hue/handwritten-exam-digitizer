"""Glyph classifier (36 classes: 0-9, A-Z) and the two constrained decoders built on it:

* `GlyphReader.decode_header`  – margin mark → question label "Q<n>" + posterior (lattice over segmentations);
* `GlyphReader.decode_roll`    – six roll-number boxes → student ID under the format grammar `d d L d d d`
                                 and a roster policy (strict / digits-exact / snap / format-only).

The CNN weights ship with the package (assets/glyph_cnn.pt). If they are missing, `train_glyph_cnn` re-trains the
network from MNIST + EMNIST-letters (downloaded from the Hugging Face Hub) exactly as in the project notebook.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch

from ..preprocessing.page import Prepped

CLASSES = [str(d) for d in range(10)] + [chr(65 + i) for i in range(26)]
DIGITS, LETTERS = list(range(10)), list(range(10, 36))
MARKER = [CLASSES.index(c) for c in "Q9GAO0"]   # classes a handwritten "Q"/"q" is read as
ROLL_FORMAT = ["d", "d", "L", "d", "d", "d"]
ROLL_RE = re.compile(r"^\d{2}[A-Z]\d{3}$")
ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_WEIGHTS = ASSETS / "glyph_cnn.pt"
ROLL_POLICIES = ("digits-exact", "strict", "snap", "format-only")


def to_mnist(crop: np.ndarray) -> np.ndarray:
    """crop: grey (paper=255) or ink-bright array -> 28x28 MNIST-style (ink bright, 20-px box, CoM-centred)."""
    a = crop.astype(np.float32)
    if a.mean() > 127:
        a = 255 - a
    a[a < 40] = 0
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), np.uint8)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20.0 / max(h, w)
    a = cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    h, w = a.shape
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    m = out.sum()
    if m > 0:
        cy, cx = (np.arange(28)[:, None] * out).sum() / m, (np.arange(28)[None, :] * out).sum() / m
        out = cv2.warpAffine(out, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))
    return np.clip(out / max(out.max(), 1) * 255, 0, 255).astype(np.uint8)


class GlyphCNN(torch.nn.Module):
    """≈0.4 M parameter CNN, 28×28 input, 36 classes."""

    def __init__(self, n: int = 36):
        super().__init__()
        c = torch.nn
        self.f = c.Sequential(c.Conv2d(1, 32, 3, padding=1), c.BatchNorm2d(32), c.ReLU(), c.MaxPool2d(2),
                              c.Conv2d(32, 64, 3, padding=1), c.BatchNorm2d(64), c.ReLU(), c.MaxPool2d(2),
                              c.Conv2d(64, 128, 3, padding=1), c.BatchNorm2d(128), c.ReLU(), c.MaxPool2d(2),
                              c.Flatten(), c.Dropout(0.3), c.Linear(128 * 3 * 3, 256), c.ReLU(), c.Dropout(0.3), c.Linear(256, n))

    def forward(self, x):
        return self.f(x)


def local_ink(gray_crop: np.ndarray) -> np.ndarray:
    """Ink mask thresholded on the crop itself (Otsu): faint question numbers survive, noise is closed over."""
    g = cv2.GaussianBlur(gray_crop, (3, 3), 0)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g < min(t, 225)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def mark_components(gray_crop: np.ndarray):
    """Connected components of a margin mark, left to right, without dots and noise specks."""
    ink = local_ink(gray_crop)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = [dict(x0=st[i, 0], y0=st[i, 1], x1=st[i, 0] + st[i, 2], y1=st[i, 1] + st[i, 3], area=st[i, 4], ids=[i])
             for i in range(1, n) if st[i, 4] >= 20]
    if not comps:
        return [], lab
    hmax, amax = max(c["y1"] - c["y0"] for c in comps), max(c["area"] for c in comps)
    keep = [c for c in comps if c["area"] >= 0.08 * amax and not (c["y1"] - c["y0"] < 0.42 * hmax and c["x1"] - c["x0"] < 0.42 * hmax)]
    keep.sort(key=lambda c: (c["x0"] + c["x1"]) / 2)
    merged = []                               # pieces almost fully stacked on top of each other are one glyph
    for c in keep:
        if merged:
            m = merged[-1]
            ov = min(m["x1"], c["x1"]) - max(m["x0"], c["x0"])
            if ov > 0.7 * min(m["x1"] - m["x0"], c["x1"] - c["x0"]):
                m.update(x0=min(m["x0"], c["x0"]), y0=min(m["y0"], c["y0"]), x1=max(m["x1"], c["x1"]),
                         y1=max(m["y1"], c["y1"]), area=m["area"] + c["area"], ids=m["ids"] + c["ids"])
                continue
        merged.append(c)
    return merged, lab


def group_glyph(gray_crop: np.ndarray, lab: np.ndarray, comps) -> np.ndarray:
    x0, y0 = min(c["x0"] for c in comps), min(c["y0"] for c in comps)
    x1, y1 = max(c["x1"] for c in comps), max(c["y1"] for c in comps)
    m = np.isin(lab[y0:y1, x0:x1], [i for c in comps for i in c["ids"]])
    return to_mnist(np.where(m, 255 - gray_crop[y0:y1, x0:x1].astype(np.float32), 0))


def find_roll_boxes(pp: Prepped):
    """The row of ≥4 printed roll-number boxes in the page header (left→right)."""
    y_max = pp.header_bottom or int(0.15 * pp.gray.shape[0])
    bw = (pp.gray[:y_max] < 170).astype(np.uint8)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = [cv2.boundingRect(c) for c in cnts]
    cands = [(x, y, w, h) for x, y, w, h in cands if 50 <= w <= 120 and 55 <= h <= 125 and 0.55 <= w / h <= 1.35]
    cands = [c for c in cands if not any(o != c and o[0] <= c[0] + 3 and o[1] <= c[1] + 3 and o[0] + o[2] >= c[0] + c[2] - 3
                                         and o[1] + o[3] >= c[1] + c[3] - 3 and o[2] * o[3] > c[2] * c[3] for o in cands)]
    if not cands:
        return []
    ys = np.array([c[1] for c in cands])
    best = max(cands, key=lambda c: (np.abs(ys - c[1]) < 15).sum())
    row = sorted([c for c in cands if abs(c[1] - best[1]) < 15], key=lambda c: c[0])
    return row if len(row) >= 4 else []


def clean_roster(roster: Optional[Iterable[str]]) -> list[str]:
    """Keep roster entries that match the roll-number grammar `NN L NNN`; warn about the rest."""
    if not roster:
        return []
    out, bad = [], []
    for r in roster:
        r = str(r).strip().upper()
        (out if ROLL_RE.match(r) else bad).append(r)
    if bad:
        warnings.warn(f"{len(bad)} roster entries ignored (not of the form 23D001): {bad[:5]}")
    return sorted(set(out))


class GlyphReader:
    """Holds the glyph CNN and exposes the two constrained decoders."""

    def __init__(self, weights: Optional[Path] = None, device: str = "cpu", auto_train: bool = True):
        self.device = device
        self.net = GlyphCNN().to(device)
        path = Path(weights or DEFAULT_WEIGHTS)
        if not path.exists():
            if not auto_train:
                raise FileNotFoundError(f"glyph CNN weights not found at {path}")
            warnings.warn(f"glyph CNN weights not found at {path}; training from MNIST/EMNIST (needs internet, a few minutes)")
            train_glyph_cnn(path, device=device)
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()
        self.weights_path = path

    @torch.inference_mode()
    def probs(self, arrs28: Sequence[np.ndarray]) -> np.ndarray:
        if len(arrs28) == 0:
            return np.zeros((0, 36))
        x = torch.tensor(np.stack(arrs28), dtype=torch.float32).unsqueeze(1).to(self.device) / 255.0
        return torch.softmax(self.net(x), 1).cpu().numpy()

    # ----------------------------------------------------------------- question numbers
    def decode_header(self, gray_crop: np.ndarray, prev_n: int = 0, used: Iterable[int] = (), max_q: int = 20,
                      merge_penalty: float = np.log(0.3)):
        """Lattice decoding of a margin mark into a label Q<n> (n = 1..max_q).
        Segmentations: every way of grouping adjacent ink components into at most 3 glyphs. Per segmentation,
        candidate labels are 'Q'+digits or digits only; score = glyph log-probs + log prior(form) +
        log prior(n | previous answer) + penalty per merge. The label posterior is marginalised over segmentations.
        Returns (label or None, posterior)."""
        used = set(used)
        comps, lab = mark_components(gray_crop)
        m = len(comps)
        if m == 0 or m > 5:
            return None, 0.0
        groups, segs = {}, []
        for mask in range(2 ** (m - 1)):
            gs, cur = [], [0]
            for i in range(m - 1):
                if mask >> i & 1:
                    cur.append(i + 1)
                else:
                    gs.append(tuple(cur))
                    cur = [i + 1]
            gs.append(tuple(cur))
            if len(gs) <= 3:
                segs.append((gs, bin(mask).count("1")))
                for g in gs:
                    groups.setdefault(g, None)
        keys = list(groups)
        probs = self.probs([group_glyph(gray_crop, lab, [comps[i] for i in g]) for g in keys]) + 1e-9
        P = dict(zip(keys, probs))

        def prior_n(n):
            if n in used:
                return np.log(0.02)
            return np.log(0.5) if n == prev_n + 1 else np.log(0.5 / max_q)

        cands = {}
        for gs, n_merge in segs:
            k = len(gs)
            for n in range(1, max_q + 1):
                ds = [int(c) for c in str(n)]
                base = n_merge * merge_penalty + prior_n(n)
                if k == len(ds) + 1:
                    sc = np.log(0.85) + np.log(P[gs[0]][MARKER].sum()) + sum(np.log(P[gs[i + 1]][d]) for i, d in enumerate(ds))
                    cands[f"Q{n}"] = np.logaddexp(cands.get(f"Q{n}", -np.inf), sc + base)
                if k == len(ds):
                    sc = np.log(0.15) + sum(np.log(P[gs[i]][d]) for i, d in enumerate(ds))
                    cands[f"Q{n}"] = np.logaddexp(cands.get(f"Q{n}", -np.inf), sc + base)
        if not cands:
            return None, 0.0
        labs = list(cands)
        sc = np.array([cands[l] for l in labs])
        post = np.exp(sc - sc.max())
        post /= post.sum()
        j = int(np.argmax(post))
        return labs[j], float(post[j])

    # ----------------------------------------------------------------- roll numbers
    def decode_roll(self, pp: Prepped, roster: Optional[Sequence[str]] = None, policy: str = "digits-exact", tau: float = 0.30):
        """Student-ID extraction from the six header boxes. Policies:
           strict        – the string read must itself be on the roster (and confidence >= tau);
           digits-exact  – the five digits must match a roster entry exactly; the letter (same for the whole class) is
                           taken from the roster; confidence = product of digit probabilities >= tau;   (default)
           snap          – always return the roster entry with the highest joint likelihood (never abstains);
           format-only   – no roster: return the string read if it matches `NN L NNN` and confidence >= tau.
        Without a roster every policy falls back to format-only. Returns (student_id or None, confidence, info)."""
        if policy not in ROLL_POLICIES:
            raise ValueError(f"unknown roll policy {policy!r}; choose from {ROLL_POLICIES}")
        roster = clean_roster(roster)
        if not roster:
            policy = "format-only"
        boxes = find_roll_boxes(pp)
        if len(boxes) != 6:
            return None, 0.0, {"reason": f"{len(boxes)} boxes found", "policy": policy}
        glyphs, empty = [], 0
        for x, y, w, h in boxes:
            inner = pp.gray[y + 8:y + h - 8, x + 8:x + w - 8]
            ink = local_ink(inner) if inner.min() < 140 else np.zeros_like(inner)
            if ink.sum() < 40:
                empty += 1
                glyphs.append(np.zeros((28, 28), np.uint8))
                continue
            glyphs.append(to_mnist(np.where(ink > 0, 255 - inner.astype(np.float32), 0)))
        if empty >= 2:
            return None, 0.0, {"reason": "boxes empty", "policy": policy}
        P = self.probs(glyphs)
        chars, probs, logp = [], [], np.zeros(len(roster))
        for i, kind in enumerate(ROLL_FORMAT):
            allowed = DIGITS if kind == "d" else LETTERS
            sub = P[i, allowed] / P[i, allowed].sum()
            chars.append(CLASSES[allowed[int(np.argmax(sub))]])
            probs.append(float(sub.max()))
            if roster:
                logp += np.log(np.array([sub[allowed.index(CLASSES.index(r[i]))] for r in roster]) + 1e-12)
        sid = "".join(chars)
        info = {"read": sid, "policy": policy}
        if policy == "format-only":
            conf = float(np.prod(probs))
            return (sid if conf >= tau else None), conf, {**info, **({} if conf >= tau else {"reason": "low confidence"})}
        if policy == "snap":
            j = int(np.argmax(logp))
            return roster[j], float(np.exp(logp[j])), info
        if policy == "strict":
            conf = float(np.prod(probs))
            ok = sid in roster and conf >= tau
            return (sid if ok else None), conf, {**info, **({} if ok else {"reason": "not on roster" if sid not in roster else "low confidence"})}
        digits = sid[:2] + sid[3:]
        match = [r for r in roster if r[:2] + r[3:] == digits]
        conf = float(np.prod([p for p, k in zip(probs, ROLL_FORMAT) if k == "d"]))
        if match and conf >= tau:
            return match[0], conf, info
        return None, conf, {**info, "reason": "digits not on roster" if not match else "low confidence"}


# --------------------------------------------------------------------------- training fallback
def train_glyph_cnn(out_path: Path, device: str = "cpu", seed: int = 7, per_letter: int = 1200, n_digits: int = 30000,
                    epochs: int = 4) -> Path:
    """Re-train the glyph CNN from MNIST + EMNIST-letters *train* splits (Hugging Face Hub), as in the notebook."""
    import io as _io
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    def dec(img_field):
        return np.array(Image.open(_io.BytesIO(img_field["bytes"])).convert("L"))

    mnist = pd.read_parquet(hf_hub_download("ylecun/mnist", "mnist/train-00000-of-00001.parquet", repo_type="dataset"))
    emnist = pd.read_parquet(hf_hub_download("tanganke/emnist_letters", "emnist-letters/train-00000-of-00001.parquet", repo_type="dataset"))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, y = [], []
    for img, lab in zip(*[mnist.sample(n=n_digits, random_state=seed)[c] for c in ("image", "label")]):
        X.append(to_mnist(dec(img)))
        y.append(int(lab))
    for lab in range(26):
        for img in emnist[emnist.label == lab].sample(n=per_letter, random_state=seed).image:
            X.append(to_mnist(dec(img).T))          # EMNIST is stored transposed
            y.append(10 + lab)
    Xt = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(1) / 255.0
    yt = torch.tensor(np.array(y))
    net = GlyphCNN().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 256
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(Xt) // bs + 1))

    def augment(xb):
        n = xb.shape[0]
        ang = (torch.rand(n) - 0.5) * 0.35
        sc = 0.85 + torch.rand(n) * 0.3
        tx, ty = (torch.rand(n) - 0.5) * 0.15, (torch.rand(n) - 0.5) * 0.15
        th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                          torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
        grid = torch.nn.functional.affine_grid(th, xb.shape, align_corners=False)
        return torch.nn.functional.grid_sample(xb, grid, align_corners=False)

    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(net(augment(Xt[idx]).to(device)), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return out_path
