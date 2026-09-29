import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from sklearn_decision import ChoiceClassifier, FakeModel, NoulClassifier, calibrate_zero_shot


def data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = [f"row {i} {rng.integers(1000)}" for i in range(n)]
    y = np.array(["a", "b", "c", "d"] * (n // 4))
    return X, y


def overconfident_classifier():
    # temperature 0.1: near one-hot answers that know nothing about y
    return ChoiceClassifier("Which?", dict.fromkeys("abcd"), model=FakeModel(temperature=0.1))


def test_calibration_fixes_overconfidence_without_refitting():
    X, y = data()
    clf = overconfident_classifier().fit(X)
    raw = log_loss(y, clf.predict_proba(X), labels=clf.classes_)
    cal = calibrate_zero_shot(clf, X, y)
    assert isinstance(cal, CalibratedClassifierCV)
    P = cal.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert log_loss(y, P, labels=cal.classes_) < min(raw, np.log(4) + 0.1)
    assert cal.estimator.estimator is clf  # frozen: neither refitted nor cloned


def test_works_with_two_labels_per_class_and_unfitted_estimators():
    X, y = data(8)
    cal = calibrate_zero_shot(overconfident_classifier(), X, y)
    assert list(cal.classes_) == ["a", "b", "c", "d"]


def test_binary_noul_classifier():
    X, _ = data()
    y = np.array([0, 1] * 20)
    clf = NoulClassifier("Is it positive?", model=FakeModel(temperature=0.1)).fit(X, y)
    cal = calibrate_zero_shot(clf, X, y)
    assert cal.predict_proba(X).shape == (40, 2)
