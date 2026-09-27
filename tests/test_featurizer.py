import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn_decision import (
    DecisionModelError,
    FakeModel,
    QuestionFeaturizer,
    Response,
    choice,
    noul,
    score,
)

BANK = {
    "refund": noul("The customer asks for money back."),
    "product": choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency": score("How urgent is the request?", ["can wait", "this week", "today"]),
}


def feat(**kw):
    kw.setdefault("model", FakeModel())
    kw.setdefault("cache_path", None)
    return QuestionFeaturizer(BANK, **kw)


@pytest.mark.parametrize("kw, names", [
    ({}, ["refund", "product__app", "product__api", "product__billing", "urgency__ev"]),
    ({"score_repr": "probs"},
     ["refund", "product__app", "product__api", "product__billing", "urgency__L0", "urgency__L1", "urgency__L2"]),
    ({"score_repr": "both", "drop_redundant": True},
     ["refund", "product__app", "product__api", "urgency__L0", "urgency__L1", "urgency__ev"]),
    ({"include_confidence": True},
     ["refund", "product__app", "product__api", "product__billing", "product__confidence",
      "urgency__ev", "urgency__confidence"]),
])
def test_column_layout(texts, kw, names):
    f = feat(**kw).fit(texts)
    X = f.transform(texts)
    assert list(f.get_feature_names_out()) == names
    assert X.shape == (len(texts), len(names))
    assert sum(len(v) for v in f.feature_groups_.values()) == len(names)


def test_identity_values_are_probabilities(texts):
    f = feat(score_repr="both").fit(texts)
    X = f.transform(texts)
    g = f.feature_groups_
    assert np.all((X[:, g["refund"]] > 0) & (X[:, g["refund"]] < 1))
    np.testing.assert_allclose(X[:, g["product"]].sum(axis=1), 1)
    probs, ev = X[:, g["urgency"][:3]], X[:, g["urgency"][3]]
    np.testing.assert_allclose(ev, probs @ [0, 1, 2])


def test_links(texts):
    P = feat(score_repr="probs").fit_transform(texts)
    L = feat(score_repr="probs", link="logit").fit_transform(texts)
    np.testing.assert_allclose(L, np.log(P / (1 - P)), rtol=1e-6)
    f = feat(score_repr="probs", link="clr").fit(texts)
    C = f.transform(texts)
    for q in ("product", "urgency"):
        cols = f.feature_groups_[q]
        np.testing.assert_allclose(C[:, cols].sum(axis=1), 0, atol=1e-9)  # clr is centred
        lp = np.log(P[:, cols])
        np.testing.assert_allclose(C[:, cols], lp - lp.mean(axis=1, keepdims=True))
    r = f.feature_groups_["refund"][0]
    np.testing.assert_allclose(C[:, r], L[:, r])  # noul columns still get logit


def test_clr_rejects_drop_redundant(texts):
    with pytest.raises(ValueError, match="clr"):
        feat(link="clr", drop_redundant=True).fit(texts)


def test_fit_makes_no_model_calls(texts):
    f = feat().fit(texts)
    assert f.model_.usage["calls"] == 0


def test_cache_hits_on_rerun_and_dedupes_identical_rows(tmp_path, texts):
    f = feat(cache_path=str(tmp_path / "c.sqlite")).fit(texts)
    rows = texts + texts[:2]  # two duplicates
    X1 = f.transform(rows)
    assert f.model_.usage["calls"] == len(texts)
    X2 = f.transform(rows)
    assert f.model_.usage["calls"] == len(texts)
    np.testing.assert_array_equal(X1, X2)
    # a fresh estimator on the same file reuses answers too
    g = clone(f).fit(texts)
    np.testing.assert_array_equal(g.transform(rows), X1)
    assert g.model_.usage["calls"] == 0


def test_adding_a_question_only_fetches_that_column(texts):
    f = feat(cache_path=None).fit(texts)
    f.transform(texts)
    shared = f.cache_
    bank2 = dict(BANK, extra=noul("Mentions a date."))
    g = QuestionFeaturizer(bank2, model=FakeModel(max_questions_per_call=10), cache_path=None).fit(texts)
    g.cache_ = shared
    g.transform(texts)
    assert g.model_.usage["answers_fetched"] == len(texts)


def test_chunking_follows_model_capability(texts):
    f = feat(model=FakeModel(max_questions_per_call=1)).fit(texts)
    f.transform(texts)
    assert f.model_.usage["calls"] == len(texts) * len(BANK)


def test_transform_leaves_estimator_dict_unchanged(texts):
    f = feat().fit(texts)
    before = dict(f.__dict__)
    f.transform(texts)
    assert f.__dict__.keys() == before.keys()
    assert all(f.__dict__[k] is before[k] for k in before)


def test_dataframe_state_columns_and_state_fn():
    df = pd.DataFrame({"text": ["a", "b"], "noise": [1, 2]})
    f = feat(state_columns=["text"]).fit(df)
    assert list(f.feature_names_in_) == ["text", "noise"]
    seen = []

    class Spy(FakeModel):
        def answer(self, items):
            seen.extend(s for s, _ in items)
            return super().answer(items)

    f = feat(model=Spy(), state_columns=["text"]).fit(df)
    f.transform(df)
    assert seen == [{"text": "a"}, {"text": "b"}]
    seen.clear()
    f = feat(model=Spy(), state_fn=str.upper).fit(["a"])
    f.transform(["a"])
    assert seen == ["A"]


def test_1d_input_has_no_n_features_in(texts):
    f = feat().fit(texts)
    assert not hasattr(f, "n_features_in_")
    f.fit(np.array(texts).reshape(-1, 1))
    assert f.n_features_in_ == 1
    with pytest.raises(ValueError, match="Reshape your data"):
        f.transform(texts)


def test_set_output_pandas(texts):
    out = feat().set_output(transform="pandas").fit_transform(texts)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["refund", "product__app", "product__api", "product__billing", "urgency__ev"]


class Flaky(FakeModel):
    """Fails every request whose state contains 'fail'."""

    def __init__(self, seed=0, *, fatal=False):
        super().__init__(seed)
        self.fatal = fatal

    def answer(self, items):
        good = super().answer(items)
        return [DecisionModelError("boom", fatal=self.fatal) if "fail" in s else r
                for (s, _), r in zip(items, good)]


def test_on_error_nan_and_successes_are_cached():
    rows = ["ok one", "please fail", "ok two"]
    f = feat(model=Flaky(), on_error="nan").fit(rows)
    with pytest.warns(UserWarning, match="1 of 3 model requests failed"):
        X = f.transform(rows)
    assert np.isnan(X[1]).all() and not np.isnan(X[[0, 2]]).any()

    f = feat(model=Flaky()).fit(rows)
    with pytest.raises(DecisionModelError, match="boom"):
        f.transform(rows)
    calls = f.model_.usage["calls"]
    f.transform(["ok one", "ok two"])  # cached before the raise
    assert f.model_.usage["calls"] == calls


def test_fatal_errors_raise_even_with_nan():
    f = feat(model=Flaky(fatal=True), on_error="nan").fit(["please fail"])
    with pytest.raises(DecisionModelError):
        f.transform(["please fail"])


def test_mixed_versions_warn(texts):
    f = feat().fit(texts)
    f.transform(texts[:3])
    # same cache, same namespace, different reported version
    g = feat(model=FakeModel(version="fake-1")).fit(texts)
    g.cache_ = f.cache_

    class Newer(FakeModel):
        def answer(self, items):
            return [Response(r.answers, "fake-2") for r in super().answer(items)]

    g.model_ = Newer().resolve()
    with pytest.warns(UserWarning, match="several model versions"):
        g.transform(texts)


def test_pickle_roundtrip_with_sqlite(tmp_path, texts):
    f = feat(cache_path=str(tmp_path / "c.sqlite")).fit(texts)
    X = f.transform(texts)
    g = pickle.loads(pickle.dumps(f))
    np.testing.assert_array_equal(g.transform(texts), X)


def test_grid_search_over_nested_params(texts):
    y = [1, 0, 1, 0, 1, 0]
    pipe = make_pipeline(feat(), LogisticRegression())
    gs = GridSearchCV(pipe, {"questionfeaturizer__link": ["identity", "logit"],
                             "questionfeaturizer__model__seed": [0, 1]}, cv=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(texts, y)
    assert set(gs.best_params_) == {"questionfeaturizer__link", "questionfeaturizer__model__seed"}


def test_estimate_cost_delegates_to_model(texts):
    assert feat().estimate_cost(texts) == {"rows": len(texts), "requests": len(texts)}
    est = QuestionFeaturizer(BANK, model="jev-1.13", cache_path=None).estimate_cost(texts)
    assert est["requests"] == len(texts) and est["est_cost_usd"] >= 0


def test_model_limits_checked_at_fit(texts):
    with pytest.raises(ValueError, match="limit of 2"):
        feat(model=FakeModel(max_choice_options=2)).fit(texts)


def test_reordered_choice_options_are_a_different_question(texts):
    """Regression: answers are positional, so option order must be part of the cache key."""
    a = QuestionFeaturizer({"q": choice("Which?", ["x", "y", "z"])}, model=FakeModel(temperature=0.3)).fit(texts)
    b = QuestionFeaturizer({"q": choice("Which?", ["z", "x", "y"])}, model=FakeModel(temperature=0.3)).fit(texts)
    Pa = a.transform(texts)
    Pb = b.transform(texts)  # would hit a's cached answers, misaligned, if order were ignored
    assert b.model_.usage["calls"] == len(texts)
    # FakeModel's option scores don't depend on position, so realigned columns must match
    cols = [list(b.get_feature_names_out()).index(f"q__{o}") for o in "xyz"]
    np.testing.assert_allclose(Pb[:, cols], Pa, rtol=1e-12)
