"""Backend selection: `make_line_reader` picks the recogniser from the model name (VLMs such as
Qwen3-VL vs TrOCR) and falls back to TrOCR when the VLM cannot be loaded. No models are loaded."""
from __future__ import annotations

import types

import pytest

from exam_digitizer.models import recognition as R


def _stub_trocr_init(self, name=R.DEFAULT_HTR_MODEL, device="cpu", batch_size=None,
                     cache_dir=None, max_new_tokens=72):
    self.name, self.device, self.last_seconds = name, device, 0.0


@pytest.fixture
def no_model_loading(monkeypatch):
    monkeypatch.setattr(R.QwenVLLineReader, "_load_model", lambda self: None)
    monkeypatch.setattr(R.TrOCRLineReader, "__init__", _stub_trocr_init)


@pytest.mark.parametrize("name", ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct",
                                  "vlm:openbmb/MiniCPM-V-2_6"])
def test_vl_model_names_select_the_vlm_reader(no_model_loading, name):
    r = R.make_line_reader(name, device="cpu")
    assert isinstance(r, R.QwenVLLineReader)
    assert "vlm:" not in r.name


def test_trocr_names_select_the_trocr_reader(no_model_loading):
    r = R.make_line_reader("microsoft/trocr-small-handwritten", device="cpu")
    assert isinstance(r, R.TrOCRLineReader)
    assert r.name == "microsoft/trocr-small-handwritten"


def test_vlm_load_failure_falls_back_to_trocr(monkeypatch):
    def boom(self):
        raise RuntimeError("transformers too old")
    monkeypatch.setattr(R.QwenVLLineReader, "_load_model", boom)
    monkeypatch.setattr(R.TrOCRLineReader, "__init__", _stub_trocr_init)
    with pytest.warns(UserWarning, match="falling back"):
        r = R.make_line_reader("Qwen/Qwen3-VL-2B-Instruct", device="cpu")
    assert isinstance(r, R.TrOCRLineReader)
    assert r.name == R.DEFAULT_HTR_MODEL


def test_gemini_names_select_the_gemini_reader(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    r = R.make_line_reader("gemini", device="cpu")
    assert isinstance(r, R.GeminiLineReader)
    assert r.model == R.GeminiLineReader.DEFAULT_MODEL
    r2 = R.make_line_reader("gemini-3-flash-preview", device="cpu")
    assert r2.model == "gemini-3-flash-preview"


def test_gemini_without_key_falls_back_to_trocr(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)                                        # a cwd without a .env
    monkeypatch.setattr(R.TrOCRLineReader, "__init__", _stub_trocr_init)
    with pytest.warns(UserWarning, match="falling back"):
        r = R.make_line_reader("gemini", device="cpu")
    assert isinstance(r, R.TrOCRLineReader)


def test_gemini_reader_batches_crops_and_parses_the_reply(monkeypatch):
    import json

    import numpy as np
    import requests

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    reader = R.GeminiLineReader("gemini")
    sent = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            n = len(sent["body"]["contents"][0]["parts"]) - 1          # images sent, minus the prompt
            arr = [{"text": f"line {i}", "confidence": 0.9} for i in range(n)]
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(arr)}]}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"], sent["body"] = url, json
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    crops = [np.full((40, 200), 200, np.uint8) for _ in range(3)]
    out = reader.read(crops)
    assert out == [("line 0", 0.9), ("line 1", 0.9), ("line 2", 0.9)]
    assert "test-key-not-real" not in sent["url"]                      # key travels in the header, not the URL
    assert len(sent["body"]["contents"][0]["parts"]) == 4              # 3 images + 1 prompt


def test_the_default_recogniser_is_gemini(monkeypatch):
    """The packaged default is the Gemini backend; the factory falls back to TrOCR without an API key."""
    import importlib

    import exam_digitizer.config as C
    monkeypatch.delenv("EXAM_DIGITIZER_HTR", raising=False)
    assert importlib.reload(C).Config().htr_model == "gemini"


def test_digitizer_builds_its_recogniser_through_the_factory(monkeypatch):
    from exam_digitizer import Config
    from exam_digitizer import digitizer as D
    sentinel = types.SimpleNamespace(name="stub-reader", read=lambda crops: [], last_seconds=0.0)
    seen = {}

    def fake_factory(name, **kw):
        seen["name"] = name
        return sentinel

    monkeypatch.setattr(D, "make_line_reader", fake_factory)
    dz = D.Digitizer(Config(device="cpu", htr_model="Qwen/Qwen3-VL-2B-Instruct"),
                     glyphs=types.SimpleNamespace(), calibrator=types.SimpleNamespace())
    assert dz.recognizer is sentinel
    assert seen["name"] == "Qwen/Qwen3-VL-2B-Instruct"
