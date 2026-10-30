from exam_digitizer.models.layout import crop_line, layout_page
from exam_digitizer.preprocessing import preprocess
from tests.conftest import make_page


def test_lines_and_margin_marks_are_found():
    n_answers, per = 3, 2
    pp = preprocess(make_page(n_answers=n_answers, lines_per_answer=per))
    bands, marks, info = layout_page(pp, 1)
    assert len(bands) == n_answers * per
    assert len(marks) == n_answers
    assert sum(b.header is not None for b in bands) == n_answers
    assert bands == sorted(bands, key=lambda b: b.y0)             # reading order
    crop = crop_line(pp, bands[0])
    assert crop.ndim == 2 and crop.shape[0] > 20 and crop.shape[1] > 200


def test_without_marks_no_headers_are_attached():
    pp = preprocess(make_page(marks=False))
    bands, marks, info = layout_page(pp, 1)
    assert len(bands) == 4 and marks == [] and all(b.header is None for b in bands)


def test_wide_text_line_in_the_bottom_margin_is_kept():
    """A line written below the page-number margin (e.g. the last line of a full page) is a real text
    line and must be found; the narrow printed page number alone must not become a band (tests above)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from tests.conftest import FONT, MARGIN_X, P_H

    pil = Image.fromarray(make_page(n_answers=2, lines_per_answer=2))
    d = ImageDraw.Draw(pil)
    d.text((MARGIN_X + 40, P_H - 75), "values form the basis of our thoughts and behaviour",
           font=ImageFont.truetype(FONT, 36), fill=30)
    bands, marks, info = layout_page(preprocess(np.asarray(pil, np.uint8)), 1)
    assert len(bands) == 5
    assert bands[-1].y1 > info["body"][1]                        # it lies below the body window
    assert bands == sorted(bands, key=lambda b: b.y0)            # reading order is kept
