import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from sklearn_decision import (
    ChoiceClassifier,
    FakeModel,
    JevModel,
    NoulClassifier,
    QuestionFeaturizer,
    ScoreRegressor,
)

FEAT = QuestionFeaturizer(cache_path=None)


def test_choice_classifier_classes_follow_np_unique_with_int_labels(texts):
    clf = ChoiceClassifier("Which team?", {2: "billing", 0: "support", 1: "sales"}, model=FakeModel(),
                           featurizer=FEAT).fit(texts)
    np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
    q = clf.featurizer_.questions["q"]
    assert q["criteria"] == {"0": "support", "1": "sales", "2": "billing"}
    P = clf.predict_proba(texts)
    assert P.shape == (len(texts), 3)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    np.testing.assert_array_equal(clf.predict(texts), clf.classes_[P.argmax(axis=1)])
    assert clf.predict_confidence(texts).shape == (len(texts),)


def test_choice_classifier_labels_from_y(texts):
    y = ["a", "b", "a", "c", "b", "c"]
    clf = ChoiceClassifier("Which?", model=FakeModel(), featurizer=FEAT).fit(texts, y)
    np.testing.assert_array_equal(clf.classes_, ["a", "b", "c"])
    with pytest.raises(ValueError, match="missing from criteria"):
        ChoiceClassifier("Which?", {"a": None, "b": None}, model=FakeModel(), featurizer=FEAT).fit(texts, y)
    with pytest.raises(ValueError, match="requires y"):
        ChoiceClassifier("Which?", model=FakeModel(), featurizer=FEAT).fit(texts)


def test_choice_classifier_string_collision():
    with pytest.raises(ValueError, match="collide"):
        ChoiceClassifier("x", {1: None, "1": None}, model=FakeModel(), featurizer=FEAT).fit(["a"])


def test_choice_classifier_respects_model_option_limit(texts):
    with pytest.raises(ValueError, match="limit of 2"):
        ChoiceClassifier("x", list("abc") and dict.fromkeys("abc"), model=FakeModel(max_choice_options=2),
                         featurizer=FEAT).fit(texts)


def test_noul_classifier_positive_label_orientation(texts):
    y = np.array(["no", "yes"] * 3)
    a = NoulClassifier("Asks for a refund.", model=FakeModel(), featurizer=FEAT).fit(texts, y)
    b = NoulClassifier("Asks for a refund.", positive_label="no", model=FakeModel(), featurizer=FEAT).fit(texts, y)
    Pa, Pb = a.predict_proba(texts), b.predict_proba(texts)
    np.testing.assert_allclose(Pa, Pb[:, ::-1])  # same P(true), attached to the other class
    p_true = a.featurizer_.transform(texts)[:, 0]
    np.testing.assert_allclose(Pa[:, 1], p_true)
    np.testing.assert_array_equal(a.predict(texts), np.where(p_true >= 0.5, "yes", "no"))
    np.testing.assert_array_equal(b.predict(texts), np.where(p_true >= 0.5, "no", "yes"))


def test_noul_classifier_without_y(texts):
    clf = NoulClassifier("x", model=FakeModel(), featurizer=FEAT).fit(texts)
    np.testing.assert_array_equal(clf.classes_, [False, True])
    assert not hasattr(clf, "predict_confidence")
    with pytest.raises(ValueError, match="Only binary"):
        NoulClassifier("x", model=FakeModel(), featurizer=FEAT).fit(texts, [0, 1, 2, 0, 1, 2])


def test_score_regressor(texts):
    reg = ScoreRegressor("How urgent?", ["low", "mid", "high"], [0, 5, 10], model=FakeModel(),
                         featurizer=FEAT).fit(texts)
    P = reg.predict_levels_proba(texts)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    np.testing.assert_allclose(reg.predict(texts), P @ [0, 5, 10])
    with pytest.raises(ValueError, match="level_values"):
        ScoreRegressor("x", ["a", "b"], [1], model=FakeModel(), featurizer=FEAT).fit(texts)


def test_confidence_hidden_when_model_has_none(texts):
    clf = ChoiceClassifier("x", dict.fromkeys("ab"), model=FakeModel(native_confidence=False), featurizer=FEAT)
    assert not hasattr(clf, "predict_confidence")
    clf.fit(texts)
    assert not hasattr(clf, "predict_confidence")
    assert clf.predict_proba(texts).shape == (len(texts), 2)
    assert hasattr(ChoiceClassifier("x", model="jev-1.13"), "predict_confidence")


def test_model_param_overrides_template_and_is_not_mutated(texts):
    m = FakeModel(seed=3)
    clf = NoulClassifier("x", model=m, featurizer=QuestionFeaturizer(model="jev-1.13", cache_path=None))
    clf.fit(texts)
    assert isinstance(clf.model_, FakeModel) and clf.model_ is not m
    assert clf.model is m and m.usage["calls"] == 0
    clf.predict(texts)
    assert m.usage["calls"] == 0 and clf.model_.usage["calls"] == len(texts)


def test_nan_rows_block_predict_but_not_predict_proba():
    class Down(FakeModel):
        def answer(self, items):
            from sklearn_decision import DecisionModelError
            return [DecisionModelError("down") for _ in items]

    clf = NoulClassifier("x", model=Down(), featurizer=QuestionFeaturizer(cache_path=None, on_error="nan"))
    clf.fit(["a"])
    with pytest.warns(UserWarning):
        assert np.isnan(clf.predict_proba(["a"])).all()
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="predict_proba"):
        clf.predict(["a"])


def test_plugs_into_calibration_and_grid_search(texts):
    X = texts * 4
    y = np.array([1, 0, 1, 0, 1, 0] * 4)
    base = NoulClassifier("Asks for a refund.", model=FakeModel(), featurizer=FEAT)
    cal = CalibratedClassifierCV(base, cv=2).fit(X, y)
    assert cal.predict_proba(X).shape == (len(X), 2)
    gs = GridSearchCV(base, {"instructions": ["Asks for a refund.", "Is unhappy."], "model__seed": [0, 1]},
                      cv=2).fit(X, y)
    assert gs.best_params_["instructions"] in {"Asks for a refund.", "Is unhappy."}


def test_jev_model_instance_as_param(texts):
    clf = ChoiceClassifier("x", dict.fromkeys("ab"), model=JevModel("jev-1.13", timeout=60), featurizer=FEAT)
    assert clf.get_params()["model__timeout"] == 60
    clf.fit(texts)  # no credentials needed to fit
    assert clf.model_.cache_namespace() == "jev:jev-1.13"
