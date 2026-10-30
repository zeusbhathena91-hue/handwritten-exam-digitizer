from exam_digitizer.evaluation import cer_counts, evaluate, norm_text, wer_counts


def test_normalisation_makes_tokenised_and_plain_text_equal():
    assert norm_text("it 's a start .") == norm_text("it's a start.")
    assert norm_text("“quoted” — text") == '" quoted " - text'


def test_cer_wer_counts():
    assert cer_counts("abc", "abc") == (0, 3)
    assert cer_counts("abc", "") == (3, 3)
    assert cer_counts("abc", "abd")[0] == 1
    assert wer_counts("the cat sat", "the hat sat") == (1, 3)


def test_evaluate_segmentation_and_id_metrics():
    gt = {"booklet_id": "X", "student_id": "23D001", "n_pages": 1,
          "questions": [{"question": "Q1", "answer_text": "hello world", "page": 1, "lines": [0]},
                        {"question": "Q2", "answer_text": "good bye", "page": 1, "lines": [1]}]}
    pr = {"student_id": None, "answers": [{"question": "Q1", "answer_text": "hello world", "page": 1, "confidence": 0.9},
                                          {"question": "Q7", "answer_text": "good bye", "page": 1, "confidence": 0.5}]}
    m, q, b = evaluate([gt], {"X": pr})
    assert m["Seg P"] == 0.5 and m["Seg R"] == 0.5 and m["label-free CER"] == 0.0 and m["ID null"] == 1.0
    expected = len(norm_text("good bye")) / (len(norm_text("hello world")) + len(norm_text("good bye")))
    assert abs(m["CER"] - expected) < 1e-9
