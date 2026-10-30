"""Runtime configuration of the asset (everything that was a notebook global in Weeks 1–5)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env_roster() -> Optional[List[str]]:
    """EXAM_DIGITIZER_ROSTER=/path/to/roster.txt (one roll number per line) or a comma-separated list."""
    v = os.environ.get("EXAM_DIGITIZER_ROSTER")
    if not v:
        return None
    if Path(v).exists():
        return [ln.strip() for ln in Path(v).read_text().splitlines() if ln.strip()]
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Config:
    htr_model: str = os.environ.get("EXAM_DIGITIZER_HTR", "gemini")
    device: Optional[str] = os.environ.get("EXAM_DIGITIZER_DEVICE")      # None = cuda if available else cpu
    glyph_weights: Optional[Path] = None                                  # None = packaged assets/glyph_cnn.pt
    calibration: Optional[Path] = None                                    # None = packaged assets/calibration.json
    cache_dir: Optional[Path] = None                                      # optional on-disk recognition cache
    roster: Optional[List[str]] = field(default_factory=_env_roster)      # default roster (per-call roster overrides)
    roll_policy: str = os.environ.get("EXAM_DIGITIZER_ROLL_POLICY", "digits-exact")
    review_threshold: float = float(os.environ.get("EXAM_DIGITIZER_REVIEW_THRESHOLD", "0.5"))
    batch_size: Optional[int] = None

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"
