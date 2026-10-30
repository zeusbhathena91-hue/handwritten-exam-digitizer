"""Line recognisers. Every recogniser has `name` and `read(crops) -> list[(text, confidence)]`.

* `TrOCRLineReader`     – TrOCR (ViT encoder + autoregressive text decoder), the core handwriting model.
                          Line confidence = exp(mean token log-probability) of the greedy decode.
* `QwenVLLineReader`    – an open vision-language model (Qwen2/2.5/3-VL or any image-text-to-text Hub
                          model) prompted to transcribe one line per call; needs transformers >= 4.57.
* `TesseractLineReader` – Tesseract LSTM OCR per line (printed text; the Week-2 baseline recogniser).

`make_line_reader(name)` picks the class from the model name (see its docstring) and falls back to
TrOCR when a VLM cannot be loaded, so `EXAM_DIGITIZER_HTR=Qwen/Qwen3-VL-2B-Instruct` just works.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_HTR_MODEL = "microsoft/trocr-small-handwritten"


def _crop_key(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()


class TrOCRLineReader:
    def __init__(self, name: str = DEFAULT_HTR_MODEL, device: str = "cpu", batch_size: Optional[int] = None,
                 cache_dir: Optional[Path] = None, max_new_tokens: int = 72):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        try:                                   # silence the "unauthenticated requests" notice of the Hub client
            from huggingface_hub.utils import logging as hub_logging
            hub_logging.set_verbosity_error()
        except Exception:  # pragma: no cover
            pass
        try:
            import transformers.modeling_utils as mu
            mu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass
        self.name, self.device, self.max_new_tokens = name, device, max_new_tokens
        try:
            self.proc = TrOCRProcessor.from_pretrained(name, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(name, local_files_only=True).to(device).eval()
        except Exception:
            self.proc = TrOCRProcessor.from_pretrained(name)
            self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
        # compat: newer transformers read this
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.bs = batch_size or (32 if device == "cuda" else 8)
        self.torch = torch
        # optional on-disk cache keyed by the crop pixels
        self.cache_path = Path(
            cache_dir) / f"htr_{name.split('/')[-1]}.json" if cache_dir else None
        self.cache = json.loads(self.cache_path.read_text(
        )) if self.cache_path and self.cache_path.exists() else {}
        self.last_seconds = 0.0

    _key = staticmethod(_crop_key)

    def read(self, crops: Sequence[np.ndarray]):
        """crops: list of grey uint8 arrays -> list of (text, confidence). `last_seconds` holds the recognition time."""
        from PIL import Image
        torch = self.torch
        keys = [self._key(c) for c in crops]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        with torch.inference_mode():
            for s in range(0, len(todo), self.bs):
                idx = todo[s:s + self.bs]
                t0 = time.time()
                ims = [Image.fromarray(crops[i]).convert("RGB") for i in idx]
                pv = self.proc(images=ims, return_tensors="pt").pixel_values.to(
                    self.device)
                g = self.model.generate(pv, max_new_tokens=self.max_new_tokens, num_beams=1, output_scores=True,
                                        return_dict_in_generate=True)
                texts = self.proc.batch_decode(
                    g.sequences, skip_special_tokens=True)
                ts = self.model.compute_transition_scores(
                    g.sequences, g.scores, normalize_logits=True).cpu().numpy()
                gen = g.sequences[:, 1:].cpu().numpy()
                pad = self.model.generation_config.pad_token_id
                dt = (time.time() - t0) / len(idx)
                for j, i in enumerate(idx):
                    valid = gen[j][: ts.shape[1]] != pad
                    lp = ts[j][valid] if valid.any() else ts[j]
                    self.cache[keys[i]] = [texts[j].strip(), float(
                        np.exp(lp.mean())) if len(lp) else 0.0, dt]
        if todo and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        self.last_seconds = float(sum(self.cache[k][2] for k in keys))
        out = [tuple(self.cache[k][:2]) for k in keys]
        if not self.cache_path:            # no persistent cache: do not let the in-memory one grow unboundedly
            self.cache = {}
        return out


class QwenVLLineReader:
    """An image-text-to-text VLM from the Hub (Qwen2/2.5/3-VL and similar) prompted to transcribe one
    line per call. Line confidence = exp(mean token log-probability), as for TrOCR, but on a different
    scale: the packaged Platt calibration was fitted for TrOCR and should be re-fitted per VLM."""

    PROMPT = ("Transcribe the handwritten text in this image exactly as written. Do not correct "
              "spelling or grammar. Output only the transcription, nothing else.")

    def __init__(self, name: str, device: str = "cpu", batch_size: Optional[int] = None,
                 cache_dir: Optional[Path] = None, max_new_tokens: int = 72):
        self.name, self.device, self.max_new_tokens = name, device, max_new_tokens
        self.cache_path = Path(
            cache_dir) / f"htr_{name.split('/')[-1]}.json" if cache_dir else None
        self.cache = json.loads(self.cache_path.read_text(
        )) if self.cache_path and self.cache_path.exists() else {}
        self.last_seconds = 0.0
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        dtype = torch.bfloat16 if self.device == "cpu" else "auto"
        try:
            self.proc = AutoProcessor.from_pretrained(self.name, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.name, dtype=dtype, local_files_only=True).to(self.device).eval()
        except Exception:
            self.proc = AutoProcessor.from_pretrained(self.name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.name, dtype=dtype).to(self.device).eval()
        self.torch = torch

    _key = staticmethod(_crop_key)

    def read(self, crops: Sequence[np.ndarray]):
        """crops: list of grey uint8 arrays -> list of (text, confidence). `last_seconds` holds the recognition time."""
        from PIL import Image
        torch = self.torch
        keys = [self._key(c) for c in crops]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        with torch.inference_mode():
            for i in todo:
                t0 = time.time()
                im = Image.fromarray(crops[i]).convert("RGB")
                if im.height < 56:                  # the vision encoder needs a few patch rows per line
                    im = im.resize((max(28, round(im.width * 56 / im.height)), 56))
                messages = [{"role": "user", "content": [{"type": "image", "image": im},
                                                         {"type": "text", "text": self.PROMPT}]}]
                inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                       return_dict=True, return_tensors="pt").to(self.device)
                g = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                        output_scores=True, return_dict_in_generate=True)
                text = self.proc.decode(g.sequences[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
                ts = self.model.compute_transition_scores(
                    g.sequences, g.scores, normalize_logits=True)[0].cpu().numpy()
                conf = float(np.exp(ts.mean())) if len(ts) else 0.0
                self.cache[keys[i]] = [" ".join(text.split()), conf, time.time() - t0]
        if todo and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        self.last_seconds = float(sum(self.cache[k][2] for k in keys))
        out = [tuple(self.cache[k][:2]) for k in keys]
        if not self.cache_path:
            self.cache = {}
        return out


class TesseractLineReader:
    """Tesseract per line (`--psm 7`). Needs the `tesseract` binary and `pytesseract`."""
    name = "tesseract-line"

    def __init__(self):
        import pytesseract  # noqa: F401  (fail early if missing)
        self.last_seconds = 0.0

    def read(self, crops: Sequence[np.ndarray]):
        import pytesseract
        t0 = time.time()
        out = []
        for c in crops:
            d = pytesseract.image_to_data(
                c, config="--psm 7", output_type=pytesseract.Output.DATAFRAME)
            d = d[(d.conf > -1) & d.text.notna()]
            d = d[d.text.astype(str).str.strip() != ""]
            out.append((" ".join(d.text.astype(str)), float(
                d.conf.mean() / 100) if len(d) else 0.0))
        self.last_seconds = time.time() - t0
        return out


class GeminiLineReader:
    """Hosted Gemini API as the line recogniser. This sends images over the network: never use it for
    real student scripts (assignment §8) — synthetic/public pages or explicit consent only. Line crops
    are batched into one request per call; `confidence` is the model's own 0-1 estimate (uncalibrated),
    and the key comes from `GEMINI_API_KEY` (environment, or a `.env` in the working directory)."""

    DEFAULT_MODEL = "gemini-3-flash-preview"
    API = "https://generativelanguage.googleapis.com/v1beta/models"
    PROMPT = ("Each image is one handwritten line from an exam page, in order. Transcribe every image "
              "exactly as written, without correcting spelling or grammar; omit words that are crossed "
              'out. Reply with only a JSON array, one object per image, in order: '
              '{"text": "<the line>", "confidence": <0..1, how certain the transcription is exact>}.')

    def __init__(self, name: str = "gemini", device: str = "cpu", batch_size: Optional[int] = None,
                 cache_dir: Optional[Path] = None, api_key: Optional[str] = None, timeout: int = 180):
        import os
        self.model = self.DEFAULT_MODEL if name == "gemini" else name
        self.name = self.model
        self.key = api_key or os.environ.get("GEMINI_API_KEY") or self._dotenv_key()
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set (environment or ./.env)")
        self.bs = batch_size or 12
        self.timeout = timeout
        self.last_seconds = 0.0

    @staticmethod
    def _dotenv_key() -> Optional[str]:
        import re
        p = Path(".env")
        if not p.exists():
            return None
        m = re.search(r"^\s*GEMINI_API_KEY\s*=\s*['\"]?([^'\"\s]+)", p.read_text(encoding="utf-8-sig"), re.M)
        return m.group(1) if m else None

    def read(self, crops: Sequence[np.ndarray]):
        """crops: list of grey uint8 arrays -> list of (text, confidence). `last_seconds` holds the API time."""
        import base64
        import io

        import requests
        from PIL import Image
        out, t0 = [], time.time()
        for s in range(0, len(crops), self.bs):
            batch = crops[s:s + self.bs]
            parts = []
            for c in batch:
                buf = io.BytesIO()
                Image.fromarray(c).convert("RGB").save(buf, "JPEG", quality=90)
                parts.append({"inline_data": {"mime_type": "image/jpeg",
                                              "data": base64.b64encode(buf.getvalue()).decode()}})
            parts.append({"text": self.PROMPT})
            last_err = None
            for _ in range(3):                       # flaky uplinks: retry timeouts/drops
                try:
                    r = requests.post(f"{self.API}/{self.model}:generateContent",
                                      headers={"x-goog-api-key": self.key},
                                      json={"contents": [{"parts": parts}],
                                            "generationConfig": {"temperature": 0}},
                                      timeout=self.timeout)
                    break
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_err = e
            else:
                raise RuntimeError(f"Gemini API unreachable after 3 attempts: {last_err}")
            if r.status_code != 200:
                raise RuntimeError(f"Gemini API: HTTP {r.status_code}: {r.text[:200]}")
            text = "".join(p.get("text", "") for cand in r.json().get("candidates", [])
                           for p in cand.get("content", {}).get("parts", []))
            t = text.strip()
            if t.startswith("```"):
                t = t.split("\n", 1)[1].rsplit("```", 1)[0]
            items = json.loads(t)
            for it in items[:len(batch)]:
                out.append((str(it.get("text", "")).strip(), float(it.get("confidence", 0.0))))
            out += [("", 0.0)] * max(0, len(batch) - len(items))
        self.last_seconds = time.time() - t0
        return out


def make_line_reader(name: str = DEFAULT_HTR_MODEL, device: str = "cpu", batch_size: Optional[int] = None,
                     cache_dir: Optional[Path] = None):
    """Pick the recogniser from the model name: `tesseract` -> TesseractLineReader; `vlm:<hub id>`, or a
    Hub id containing `-VL` (Qwen2/2.5/3-VL, ...) -> QwenVLLineReader; anything else -> TrOCRLineReader.
    A VLM that cannot be loaded (old transformers, no weights, OOM) falls back to the default TrOCR."""
    n = name.lower()
    if n in ("tesseract", "tesseract-line"):
        return TesseractLineReader()
    if n.startswith("gemini"):
        try:
            return GeminiLineReader(name, device=device, batch_size=batch_size, cache_dir=cache_dir)
        except Exception as e:
            warnings.warn(f"could not initialise Gemini recogniser ({e}); falling back to {DEFAULT_HTR_MODEL!r}")
            return TrOCRLineReader(DEFAULT_HTR_MODEL, device=device, batch_size=batch_size, cache_dir=cache_dir)
    is_vlm = n.startswith("vlm:") or "-vl-" in n or n.endswith("-vl")
    if is_vlm:
        hub_id = name[4:] if n.startswith("vlm:") else name
        try:
            return QwenVLLineReader(hub_id, device=device, batch_size=batch_size, cache_dir=cache_dir)
        except Exception as e:
            warnings.warn(f"could not load VLM recogniser {hub_id!r} ({e}); falling back to {DEFAULT_HTR_MODEL!r}")
    return TrOCRLineReader(DEFAULT_HTR_MODEL if is_vlm else name, device=device,
                           batch_size=batch_size, cache_dir=cache_dir)
