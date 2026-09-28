"""Recalibrating zero-shot estimators.

Decision models' raw probabilities are often overconfident: on 20 Newsgroups,
Jev's zero-shot log-loss was 1.73 at 77% accuracy (benchmarks/RESULTS.md).
Calibrate on your own labels before thresholding or combining them.

Measured recommendation (log-loss on held-out data, benchmarks/calibration.py):

* **A few labels (≈ 4-15 per class):** :func:`calibrate_zero_shot`, i.e. sigmoid
  (Platt) scaling on the frozen classifier. Jev: 1.73 -> 0.71 with 16 labels.
* **More labels (≈ 16+ per class):** a logistic head on the answers' log-odds,
  ``make_pipeline(QuestionFeaturizer({...}, link="logit"), LogisticRegression())``.
  Jev: 0.60 with 400 labels.
* Avoid ``method="isotonic"`` with few labels: it made log-loss worse than raw.
"""
from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.utils.validation import check_is_fitted

__all__ = ["calibrate_zero_shot"]


def calibrate_zero_shot(estimator, X, y, *, method: str = "sigmoid", cv: int = 2) -> CalibratedClassifierCV:
    """Calibrate a zero-shot classifier's probabilities on labelled data.

    The classifier is frozen: calibration never refits it, so its answers
    (and cache) are reused as they are. Because a frozen classifier gives the
    same predictions on every fold, ``cv`` only sets how many labels each
    class needs (``cv=2``: at least two), not the result.

    Parameters
    ----------
    estimator : NoulClassifier, ChoiceClassifier or any classifier
        Fitted or not; an unfitted one is fitted on ``X, y`` first (for the
        zero-shot estimators that only records the label set).
    X, y : calibration data, with every class present at least ``cv`` times.
    method : {"sigmoid", "isotonic", "temperature"}
        "sigmoid" is the robust choice with few labels. "temperature" keeps the
        predicted class unchanged but needs scikit-learn >= 1.8.
    cv : int

    Returns
    -------
    CalibratedClassifierCV
        Fitted; use its ``predict_proba`` / ``predict``.
    """
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        estimator = clone(estimator).fit(X, y)
    return CalibratedClassifierCV(FrozenEstimator(estimator), method=method, cv=cv).fit(X, y)
