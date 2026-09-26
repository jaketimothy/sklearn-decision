"""Question-level importance, for pruning a question bank."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["grouped_permutation_importance"]


def grouped_permutation_importance(estimator, X_feat, y, groups: Mapping[str, list[int]], scoring=None,
                                   n_repeats: int = 5, random_state: int = 0):
    """Permutation importance where all columns of one question move together.

    ``estimator`` is the downstream model, already fitted on the features;
    ``X_feat`` is a held-out feature matrix (ndarray or DataFrame) and
    ``groups`` is ``featurizer.feature_groups_``. Returns a list of
    (question, mean score drop, std) sorted by importance.
    """
    from sklearn.metrics import check_scoring

    scorer = check_scoring(estimator, scoring=scoring)
    rng = np.random.default_rng(random_state)
    base = scorer(estimator, X_feat, y)
    is_df = hasattr(X_feat, "iloc")
    out = []
    for g, idx in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X_feat))
            Xp = X_feat.copy()
            if is_df:
                Xp.iloc[:, idx] = X_feat.iloc[perm, idx].to_numpy()
            else:
                Xp[:, idx] = X_feat[perm][:, idx]
            drops.append(base - scorer(estimator, Xp, y))
        out.append((g, float(np.mean(drops)), float(np.std(drops))))
    return sorted(out, key=lambda t: -t[1])
