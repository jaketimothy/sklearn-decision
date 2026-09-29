"""Decision-primitive estimators: one question, scikit-learn's predict API.

Each wraps a single question in a :class:`QuestionFeaturizer`, so they share
its cache, retries and version pinning. ``fit`` learns nothing from y beyond
the label set: these are zero-shot models that plug into sklearn's
evaluation, calibration (``CalibratedClassifierCV``), thresholding
(``TunedThresholdClassifierCV``) and stacking tools.

Their probabilities are the model's raw answers, which are often overconfident
(Jev's zero-shot log-loss was 1.73 at 77% accuracy on 20 Newsgroups).
Calibrate them on a few labels with :func:`sklearn_decision.calibrate_zero_shot`
before thresholding or combining them.

The decision model is the ``model`` parameter, and ``cache_path``,
``state_columns``, ``state_fn``, ``on_error`` and ``verbose`` work as on
:class:`~sklearn_decision.QuestionFeaturizer`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, column_or_1d

from .featurizer import QuestionFeaturizer, UsageMixin, _input_tags
from .models import DecisionModel
from .models.base import model_capabilities
from .questions import choice, noul, score

__all__ = ["NoulClassifier", "ChoiceClassifier", "ScoreRegressor"]

_QNAME = "q"


class _SingleQuestionEstimator(UsageMixin, BaseEstimator):
    """Shared plumbing: build, fit and query a one-question featurizer."""

    def _fit_featurizer(self, question: dict, X) -> None:
        caps = model_capabilities(self.model)
        confidence = question["type"] != "noul" and caps is not None and caps.native_confidence
        feat = QuestionFeaturizer(
            {_QNAME: question}, model=self.model, link="identity", score_repr="probs",
            include_confidence=confidence, cache_path=self.cache_path, state_columns=self.state_columns,
            state_fn=self.state_fn, on_error=self.on_error, verbose=self.verbose)
        self.featurizer_ = feat.fit(X)
        self.model_ = feat.model_
        for attr in ("n_features_in_", "feature_names_in_"):
            if hasattr(feat, attr):
                setattr(self, attr, getattr(feat, attr))
            else:
                self.__dict__.pop(attr, None)

    def _answers(self, X) -> np.ndarray:
        check_is_fitted(self, "featurizer_")
        return self.featurizer_.transform(X)

    def _n_dist_cols(self) -> int:
        return len(self.featurizer_.feature_names_out_) - int(self.featurizer_.include_confidence)

    def __sklearn_tags__(self):
        return _input_tags(super().__sklearn_tags__(), self.model)


class _ConfidenceMixin:
    """``predict_confidence`` for choice and score questions, when the model
    reports a confidence of its own (Jev does; local logit readouts don't)."""

    def _has_confidence(self) -> bool:
        if hasattr(self, "featurizer_"):
            return bool(self.featurizer_.include_confidence)
        caps = model_capabilities(self.model)
        return caps is not None and caps.native_confidence

    @available_if(_has_confidence)
    def predict_confidence(self, X) -> np.ndarray:
        """The model's own confidence in each answer, e.g. to abstain below a threshold.

        Only available when the model reports one (``capabilities().native_confidence``).

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame
            Rows, as for :meth:`predict`.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return self._answers(X)[:, -1]


def _check_y_for_x(X, y):
    if X is not None and y is not None and hasattr(X, "__len__"):
        check_consistent_length(X, y)


class NoulClassifier(ClassifierMixin, _SingleQuestionEstimator):
    """Zero-shot binary classifier from one noul (yes/no) statement.

    Parameters
    ----------
    instructions : str
        A statement that should be true exactly for ``positive_label``.
    positive_label : optional
        The label for which ``instructions`` is true. Required unless the
        labels are 0/1, -1/1 or booleans, where it defaults to 1 (True):
        with labels such as {"refund", "spam"}, guessing would silently
        invert the classifier.
    model : str or DecisionModel
        The decision model (required), as for :class:`QuestionFeaturizer`.
    cache_path, state_columns, state_fn, on_error, verbose
        As for :class:`QuestionFeaturizer`.

    ``predict_proba`` returns [P(negative), P(positive)] in ``classes_`` order.
    Noul answers carry no separate confidence, so there is no
    ``predict_confidence``.
    """

    def __init__(self, instructions: str = "", positive_label=None, *,
                 model: str | DecisionModel | None = None, cache_path: str | None = None,
                 state_columns: Sequence[str] | None = None, state_fn=None, on_error: str = "raise",
                 verbose: bool = False):
        self.instructions = instructions
        self.positive_label = positive_label
        self.model = model
        self.cache_path = cache_path
        self.state_columns = state_columns
        self.state_fn = state_fn
        self.on_error = on_error
        self.verbose = verbose

    def fit(self, X, y=None):
        """Record the label set; no model calls. ``y`` may be None, giving
        classes [False, True]."""
        if y is None:
            classes = np.array([False, True])
        else:
            y = column_or_1d(y, warn=True)
            y_type = type_of_target(y, input_name="y", raise_unknown=True)
            if y_type not in ("binary", "multiclass"):
                raise ValueError(f"Unknown label type: {y_type}")
            _check_y_for_x(X, y)
            classes = np.unique(y)
            if len(classes) > 2:
                raise ValueError(f"Only binary classification is supported. The type of the target is {y_type}.")
            if len(classes) < 2:
                raise ValueError(f"NoulClassifier needs 2 classes in y, got {len(classes)} class")
        if self.positive_label is not None:
            pos = self.positive_label
        elif set(classes.tolist()) in ({0, 1}, {-1, 1}):
            pos = classes[1]
        else:
            raise ValueError(f"Set positive_label: y's labels are {classes.tolist()}, and nothing says which "
                             "one the statement is true for. (It defaults to 1 only for 0/1, -1/1 or boolean "
                             "labels.)")
        if pos not in classes:
            raise ValueError(f"positive_label {pos!r} not in classes {classes.tolist()}")
        self.classes_ = classes
        self._pos_idx_ = int(np.flatnonzero(classes == pos)[0])
        self._fit_featurizer(noul(self.instructions), X)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities from the model's P(true), in ``classes_`` order.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame
            Rows: strings, JSON-able records, or a DataFrame / 2-D array.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Raw answers; see :func:`~sklearn_decision.calibrate_zero_shot`.
        """
        p = self._answers(X)[:, 0]
        P = np.column_stack([1 - p, p])
        return P if self._pos_idx_ == 1 else P[:, ::-1]

    def predict(self, X) -> np.ndarray:
        """The positive label where P(true) >= 0.5, else the negative one.

        Tune the threshold with ``TunedThresholdClassifierCV`` if 0.5 isn't right.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        p = self.predict_proba(X)[:, self._pos_idx_]
        if np.isnan(p).any():
            raise ValueError("Some rows have no answer (on_error='nan'); use predict_proba")
        idx = np.where(p >= 0.5, self._pos_idx_, 1 - self._pos_idx_)
        return self.classes_[idx]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True  # zero-shot: does not learn from y
        tags.classifier_tags.multi_class = False
        return tags


class ChoiceClassifier(ClassifierMixin, _ConfidenceMixin, _SingleQuestionEstimator):
    """Zero-shot classifier: the classes are the options of one choice question.

    Parameters
    ----------
    instructions : str
    criteria : dict or None
        {class label: description or None}. None uses the labels seen in y,
        with no descriptions. Keys are the real class labels (ints are fine);
        they are sent to the model as strings.
    model : str or DecisionModel
        The decision model (required), as for :class:`QuestionFeaturizer`.
    cache_path, state_columns, state_fn, on_error, verbose
        As for :class:`QuestionFeaturizer`.
    """

    def __init__(self, instructions: str = "", criteria: Mapping | None = None, *,
                 model: str | DecisionModel | None = None, cache_path: str | None = None,
                 state_columns: Sequence[str] | None = None, state_fn=None, on_error: str = "raise",
                 verbose: bool = False):
        self.instructions = instructions
        self.criteria = criteria
        self.model = model
        self.cache_path = cache_path
        self.state_columns = state_columns
        self.state_fn = state_fn
        self.on_error = on_error
        self.verbose = verbose

    def fit(self, X, y=None):
        """Record the label set; no model calls."""
        if y is not None:
            y = column_or_1d(y, warn=True)
            check_classification_targets(y)
            _check_y_for_x(X, y)
        if self.criteria is not None:
            labels = list(self.criteria)
            if y is not None:
                unknown = set(np.unique(y).tolist()) - set(labels)
                if unknown:
                    raise ValueError(f"y has labels missing from criteria: {sorted(unknown, key=str)}")
        elif y is not None:
            labels = np.unique(y).tolist()
        else:
            raise ValueError("ChoiceClassifier requires y to be passed, but the target y is None; "
                             "pass criteria for a y-free fit.")
        if len({str(c) for c in labels}) != len(set(labels)):
            raise ValueError("Class labels collide when converted to strings")
        classes = np.unique(np.asarray(labels))  # np.unique order, as sklearn expects
        if len(classes) < 2:
            raise ValueError(f"ChoiceClassifier needs at least 2 classes, got {len(classes)} class")
        desc = self.criteria or {}
        self.classes_ = classes
        self._fit_featurizer(choice(self.instructions, {str(c): desc.get(c) for c in classes}), X)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """The model's distribution over the options, in ``classes_`` order.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame
            Rows: strings, JSON-able records, or a DataFrame / 2-D array.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Raw answers; see :func:`~sklearn_decision.calibrate_zero_shot`.
        """
        return self._answers(X)[:, : self._n_dist_cols()]  # columns follow classes_

    def predict(self, X) -> np.ndarray:
        """The most probable class for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        P = self.predict_proba(X)
        if np.isnan(P).any():
            raise ValueError("Some rows have no answer (on_error='nan'); use predict_proba")
        return self.classes_[P.argmax(axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True  # zero-shot: does not learn from y
        return tags


class ScoreRegressor(RegressorMixin, _ConfidenceMixin, _SingleQuestionEstimator):
    """Zero-shot ordinal regressor from one score rubric.

    Parameters
    ----------
    instructions : str
    levels : sequence of str
        Rubric levels, ordered low -> high.
    level_values : sequence of float or None
        Numeric value of each level (default 0..L-1).
    model : str or DecisionModel
        The decision model (required), as for :class:`QuestionFeaturizer`.
    cache_path, state_columns, state_fn, on_error, verbose
        As for :class:`QuestionFeaturizer`.

    ``predict`` returns the expected value sum_i p_i * level_values[i];
    ``predict_levels_proba`` gives the full distribution over levels.
    """

    def __init__(self, instructions: str = "", levels: Sequence[str] = (),
                 level_values: Sequence[float] | None = None, *,
                 model: str | DecisionModel | None = None, cache_path: str | None = None,
                 state_columns: Sequence[str] | None = None, state_fn=None, on_error: str = "raise",
                 verbose: bool = False):
        self.instructions = instructions
        self.levels = levels
        self.level_values = level_values
        self.model = model
        self.cache_path = cache_path
        self.state_columns = state_columns
        self.state_fn = state_fn
        self.on_error = on_error
        self.verbose = verbose

    def fit(self, X, y=None):
        """Validate the rubric; no model calls. ``y`` is only checked."""
        if y is not None:
            y = check_array(y, ensure_2d=False, dtype="numeric", input_name="y")
            y = column_or_1d(y, warn=True)
            _check_y_for_x(X, y)
        L = len(self.levels)
        vals = np.arange(L, dtype=float) if self.level_values is None else np.asarray(self.level_values, float)
        if len(vals) != L:
            raise ValueError(f"level_values has {len(vals)} entries but there are {L} levels")
        self.level_values_ = vals
        self._fit_featurizer(score(self.instructions, self.levels), X)
        return self

    def predict_levels_proba(self, X) -> np.ndarray:
        """The model's distribution over the rubric's levels, low to high.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame

        Returns
        -------
        ndarray of shape (n_samples, n_levels)
        """
        return self._answers(X)[:, : self._n_dist_cols()]

    def predict(self, X) -> np.ndarray:
        """The expected value, ``sum_i p_i * level_values[i]``, for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or DataFrame

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return self.predict_levels_proba(X) @ self.level_values_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True  # zero-shot: does not learn from y
        return tags
