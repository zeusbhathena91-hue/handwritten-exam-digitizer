import numpy as np

from exam_digitizer.models.confidence import PlattCalibrator, review_policy


def test_platt_is_monotone_and_bounded(tmp_path):
    cal = PlattCalibrator(mean=1.5, scale=1.2, coef=1.8, intercept=0.9, meta={"fitted_on": "unit test"})
    xs = np.linspace(0.01, 0.99, 50)
    ps = cal(xs)
    assert np.all(np.diff(ps) > 0) and ps.min() > 0 and ps.max() < 1
    p = tmp_path / "cal.json"
    cal.save(p)
    cal2 = PlattCalibrator.load(p)
    assert np.allclose(cal2(xs), ps) and cal2.meta["fitted_on"] == "unit test"


def test_platt_from_sklearn_matches_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.2, 0.99, 200)
    y = (raw + rng.normal(0, 0.15, 200) > 0.7).astype(int)
    X = np.log(raw / (1 - raw))[:, None]
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    cal = PlattCalibrator.from_sklearn(pipe)
    assert np.allclose(cal(raw), pipe.predict_proba(X)[:, 1], atol=1e-6)


def test_review_policy():
    assert review_policy(0.9, [], "23D001") is False
    assert review_policy(0.3, [], "23D001") is True
    assert review_policy(0.9, ["header_low_confidence"], "23D001") is True
    assert review_policy(0.9, [], None) is True
    assert review_policy(0.6, [], "23D001", threshold=0.7) is True
