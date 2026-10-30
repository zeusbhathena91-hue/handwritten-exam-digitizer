import cv2
import numpy as np

from exam_digitizer.preprocessing import STD_W, crop_to_paper, estimate_skew, load_page, normalize_background, preprocess, rotate
from tests.conftest import MARGIN_X, make_page


def test_load_page_resizes_to_standard_width(page):
    small = cv2.resize(page, None, fx=0.5, fy=0.5)
    assert load_page(small).shape[1] == STD_W
    assert load_page(page).shape == page.shape


def test_preprocess_finds_template_elements(page):
    pp = preprocess(page)
    assert pp.margin_x is not None and abs(pp.margin_x - MARGIN_X) <= 6
    assert pp.header_bottom is not None and abs(pp.header_bottom - 262) <= 6
    assert len(pp.rules_y) >= 15
    assert abs(pp.angle) < 0.3
    assert pp.ink.dtype == np.uint8 and pp.ink.max() == 1          # binary ink mask
    assert pp.clean[pp.rules_y[5], 800] > 240                      # ruled line removed from the clean image


def test_deskew_recovers_rotation(page):
    rotated = rotate(page, 1.5, border=250)
    assert abs(estimate_skew(normalize_background(rotated)) - 1.5) <= 0.3


def test_crop_to_paper_handles_dark_background(page):
    pad = 120
    on_table = cv2.copyMakeBorder(page, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=90)
    g = crop_to_paper(load_page(on_table))
    assert g.shape[1] == STD_W
    assert np.median(g[:5]) > 200 and np.median(g[:, :5]) > 200   # dark bed removed


def test_blank_page_does_not_crash():
    pp = preprocess(make_page(blank=True))
    assert pp.ink.sum() < 0.01 * pp.ink.size
