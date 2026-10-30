"""Calibrated confidence and the human-review policy.

The pipeline's raw answer confidence (mean line probability × √header posterior) ranks answers well but is not a
probability. Platt scaling — a one-feature logistic regression on logit(raw) — maps it to P(answer is usable),
where *usable* = correct question label and answer CER ≤ 10 %. The parameters shipped in assets/calibration.json were
fitted in Week 5 on the 113 test-set answers of the core system (out-of-fold AUROC 0.855, ECE 0.061).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_CALIBRATION = ASSETS / "calibration.json"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """p = sigmoid(coef · (logit(raw) − mean) / scale + intercept)."""

    def __init__(self, mean: float, scale: float, coef: float, intercept: float, meta: Optional[dict] = None):
        self.mean, self.scale, self.coef, self.intercept = float(mean), float(scale), float(coef), float(intercept)
        self.meta = meta or {}

    def __call__(self, raw):
        z = self.coef * (_logit(raw) - self.mean) / self.scale + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def from_sklearn(cls, pipeline, meta: Optional[dict] = None) -> "PlattCalibrator":
        """From a fitted `make_pipeline(StandardScaler(), LogisticRegression())` on the single feature logit(raw)."""
        sc, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
        return cls(sc.mean_[0], sc.scale_[0], lr.coef_[0][0], lr.intercept_[0], meta)

    def to_json(self) -> dict:
        return {"kind": "platt", "feature": "logit(raw_confidence)", "mean": self.mean, "scale": self.scale,
                "coef": self.coef, "intercept": self.intercept, **self.meta}

    def save(self, path: Path = DEFAULT_CALIBRATION) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATION) -> "PlattCalibrator":
        d = json.loads(Path(path).read_text())
        meta = {k: v for k, v in d.items() if k not in ("kind", "feature", "mean", "scale", "coef", "intercept")}
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], meta)


def review_policy(confidence: float, flags: list, student_id, threshold: float = 0.5) -> bool:
    """An answer goes to a human if its calibrated confidence is below the threshold, it carries any flag, or the
    student could not be identified. Week-5 risk–coverage: accepting the 70 % most confident answers automatically
    leaves 5 % unusable among them (vs 19.5 % without review)."""
    return bool(confidence < threshold or flags or student_id is None)
