"""Turn estimator input into the per-row ``state`` values sent to a model.

Accepted input, following ``CountVectorizer`` for raw text:

* 1-D (list, Series, ndarray) of strings or JSON-able records: one state per
  element. ``n_features_in_`` is not set, as for text vectorizers.
* DataFrame: one JSON record per row (``state_columns`` picks the columns).
  Sets ``feature_names_in_`` / ``n_features_in_`` and checks them on reuse.
* 2-D array: one list record per row, or the bare value for a single column.
  Sets ``n_features_in_``.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

__all__ = ["prepare_states", "states_only"]


def _is_frame(X) -> bool:
    return hasattr(X, "columns") and hasattr(X, "to_json")


def _clean(v):
    """JSON has no NaN/inf: map them to null, recursively."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def _forget_features(est) -> None:
    for attr in ("n_features_in_", "feature_names_in_"):
        est.__dict__.pop(attr, None)


def prepare_states(est, X, *, reset: bool, state_columns=None, state_fn=None) -> list:
    """Validate ``X`` against ``est`` (setting or checking its input-feature
    attributes) and return one state per row."""
    if X is None:
        raise ValueError(f"{type(est).__name__} needs X; got None")
    if _is_frame(X):
        validate_data(est, X, reset=reset, skip_check_array=True)
        if len(X) == 0:
            raise ValueError("Found array with 0 sample(s) while a minimum of 1 is required.")
        cols = list(state_columns) if state_columns is not None else list(X.columns)
        rows = json.loads(X[cols].to_json(orient="records", date_format="iso"))
    else:
        if state_columns is not None:
            raise ValueError("state_columns only applies to DataFrame input")
        arr = X if isinstance(X, np.ndarray) else None
        if arr is None and not sparse.issparse(X):
            arr = np.asarray(X, dtype=object)
        if arr is None or arr.ndim == 2:
            # sparse input lands here too, and validate_data rejects it with sklearn's message
            arr = validate_data(est, X, reset=reset, dtype=None, ensure_all_finite=False)
            rows = arr[:, 0].tolist() if arr.shape[1] == 1 else arr.tolist()
            rows = [_clean(r) for r in rows]
        elif arr.ndim == 1:
            if reset:
                _forget_features(est)
            elif hasattr(est, "n_features_in_"):
                raise ValueError(
                    f"Expected 2D array, got 1D array instead: {type(est).__name__} was fitted on "
                    f"{est.n_features_in_} feature(s). Reshape your data either using array.reshape(-1, 1) "
                    "if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
                )
            if len(arr) == 0:
                raise ValueError("Found array with 0 sample(s) while a minimum of 1 is required.")
            rows = [_clean(r) for r in arr.tolist()]
        else:
            raise ValueError(f"Expected 1-D or 2-D input, got {arr.ndim}-D")
    if state_fn is not None:
        rows = [state_fn(r) for r in rows]
    return rows


class _Scratch(BaseEstimator):
    pass


def states_only(X, *, state_columns=None, state_fn=None) -> list:
    """States for X without touching any estimator's fitted attributes."""
    return prepare_states(_Scratch(), X, reset=True, state_columns=state_columns, state_fn=state_fn)
