"""scikit-learn's estimator checks with the local TransformersModel backend
(tiny in-memory LM; the answers are meaningless, the API contract is not)."""
from sklearn.utils.estimator_checks import parametrize_with_checks

import tiny_lm
from sklearn_decision import ChoiceClassifier, QuestionFeaturizer, TransformersModel, choice, noul
from sklearn_decision.models import transformers as tm
from test_sklearn_checks import _expected_failures

# Seed the weight registry directly: the checks clone estimators many times,
# and every clone must find the tiny model without going to the Hub.
_tok, _lm = tiny_lm.build()
tm._LOADED[(tiny_lm.NAME, "main", "cpu", "float32", False)] = (_tok, _lm, tiny_lm.COMMIT)
MODEL = TransformersModel(tiny_lm.NAME, device="cpu")


@parametrize_with_checks(
    [
        QuestionFeaturizer({"a": noul("Large."), "b": choice("Kind?", ["x", "y", "z"])}, model=MODEL,
                           link="logit"),
        ChoiceClassifier("Which class is this?", model=MODEL),
    ],
    expected_failed_checks=_expected_failures,
)
def test_sklearn_compatible_local(estimator, check):
    check(estimator)
