"""Small public-API behaviours shared by every estimator."""
import pickle
import warnings

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from sklearn_decision import (
    ChoiceClassifier,
    ChoiceEncoder,
    FakeModel,
    NoulClassifier,
    QuestionFeaturizer,
    ScoreRegressor,
    noul,
)

FEAT = QuestionFeaturizer(cache_path=None)


def estimators():
    return [
        QuestionFeaturizer({"a": noul("x"), "b": noul("y")}, model=FakeModel(), cache_path=None),
        NoulClassifier("x", model=FakeModel(), featurizer=FEAT),
        ChoiceClassifier("x", dict.fromkeys("abc"), model=FakeModel(), featurizer=FEAT),
        ScoreRegressor("x", ["lo", "hi"], model=FakeModel(), featurizer=FEAT),
        ChoiceEncoder(["p", "q", "r"], model=FakeModel(), featurizer=FEAT),
    ]


@pytest.mark.parametrize("est", estimators(), ids=lambda e: type(e).__name__)
def test_usage_reports_calls_hits_and_versions(est, texts):
    with pytest.raises(NotFittedError):
        est.usage()
    est.fit(texts)
    assert est.usage()["calls"] == 0  # fit never calls the model
    run = est.transform if hasattr(est, "transform") else est.predict
    run(texts)
    first = est.usage()
    assert first["calls"] == len(texts) and first["versions"] == ["fake-1"]
    run(texts)
    second = est.usage()
    assert second["calls"] == first["calls"] and second["cache_hits"] > first["cache_hits"]
    second["calls"] = -1  # a copy: callers can't corrupt the counters
    assert est.usage()["calls"] == first["calls"]


def upper(text):
    return text.upper()


def test_unpicklable_state_fn_warns_at_fit(texts):
    with pytest.warns(UserWarning, match="can't be pickled"):
        QuestionFeaturizer({"a": noul("x")}, model=FakeModel(), state_fn=lambda t: t.upper()).fit(texts)
    with pytest.warns(UserWarning, match="can't be pickled"):
        ChoiceEncoder(["p", "q"], model=FakeModel(), featurizer=QuestionFeaturizer(state_fn=lambda t: t)).fit(texts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = QuestionFeaturizer({"a": noul("x")}, model=FakeModel(), state_fn=upper).fit(texts)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(f)).transform(texts), f.transform(texts))
