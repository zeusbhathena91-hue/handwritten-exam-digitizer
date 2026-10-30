import numpy as np
import pytest

from exam_digitizer.models.glyphs import CLASSES, DEFAULT_WEIGHTS, GlyphReader, clean_roster, to_mnist


def test_to_mnist_shape_and_normalisation():
    blank = np.full((40, 30), 255, np.uint8)
    assert to_mnist(blank).sum() == 0
    crop = np.full((60, 40), 255, np.uint8)
    crop[10:50, 15:25] = 0                      # a vertical bar ("1")
    g = to_mnist(crop)
    assert g.shape == (28, 28) and g.dtype == np.uint8 and g.max() == 255
    ys, xs = np.where(g > 0)
    assert abs(ys.mean() - 13.5) < 1.5 and abs(xs.mean() - 13.5) < 1.5     # centre-of-mass centred


def test_clean_roster_filters_bad_entries():
    with pytest.warns(UserWarning):
        r = clean_roster(["23d001", " 23D002 ", "ABC", "23D0001", "23D002"])
    assert r == ["23D001", "23D002"]
    assert clean_roster(None) == []


def test_classes_vocabulary():
    assert len(CLASSES) == 36 and CLASSES[:10] == list("0123456789") and CLASSES[10] == "A" and CLASSES[-1] == "Z"


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="packaged glyph weights missing")
def test_decode_header_blank_and_probs():
    gr = GlyphReader(device="cpu")
    assert gr.decode_header(np.full((60, 80), 255, np.uint8)) == (None, 0.0)
    p = gr.probs([np.zeros((28, 28), np.uint8)])
    assert p.shape == (1, 36) and abs(p.sum() - 1) < 1e-5
