"""scikit-learn's estimator checks, run offline against FakeModel."""
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklearn_decision import (
    ChoiceClassifier,
    ChoiceEncoder,
    FakeModel,
    NoulClassifier,
    QuestionFeaturizer,
    ScoreRegressor,
    choice,
    noul,
    score,
)

ESTIMATORS = [
    QuestionFeaturizer(
        {
            "a": noul("The row describes something large."),
            "b": choice("Which kind is it?", ["x", "y", "z"]),
            "c": score("How strong is it?", ["low", "mid", "high"]),
        },
        model=FakeModel(),
        cache_path=None,
        score_repr="both",
        include_confidence=True,
    ),
    QuestionFeaturizer({"a": noul("Large."), "b": choice("Kind?", ["x", "y"])}, model=FakeModel(),
                       cache_path=None, link="clr"),
    NoulClassifier("The row is positive.", model=FakeModel()),
    ChoiceClassifier("Which class is this?", model=FakeModel()),
    ScoreRegressor("How big is it?", ["small", "medium", "large"], model=FakeModel()),
    ChoiceEncoder(["alpha", "beta", "gamma", "delta", "epsilon"], model=FakeModel(max_choice_options=3),
                  anchor=("none", "No option fits")),
    ChoiceEncoder("exemplars", n_exemplars=8, model=FakeModel(), random_state=0),
]


def _expected_failures(est):
    xfail = {
        "check_fit1d": "1-D input is a list of documents (one state per element), as for CountVectorizer",
    }
    if not isinstance(est, QuestionFeaturizer):
        # DataFrame rows are sent as {column: value} records and ndarray rows as
        # lists, so the model sees different states and gives different answers.
        xfail["check_classifier_data_not_an_array"] = "DataFrame states carry column names"
        xfail["check_regressor_data_not_an_array"] = "DataFrame states carry column names"
    if isinstance(est, NoulClassifier):
        # these checks train on labels such as [1, 2] or ["one", "two"]; which one the
        # statement is true for can't be guessed, so NoulClassifier asks for positive_label
        why = "labels other than 0/1, -1/1 or booleans need positive_label"
        for check in ("check_classifiers_classes", "check_estimators_dtypes", "check_fit2d_1feature"):
            xfail[check] = why
    return xfail


@parametrize_with_checks(ESTIMATORS, expected_failed_checks=_expected_failures)
def test_sklearn_compatible(estimator, check):
    check(estimator)
