"""sklearn-decision: decision models (Jev and friends) as scikit-learn estimators.

Decision primitives, one question each:
    NoulClassifier, ChoiceClassifier, ScoreRegressor
Encoders:
    QuestionFeaturizer (a bank of noul/choice/score questions -> features)
    ChoiceEncoder      (a codebook of options -> simplex embeddings)

Every estimator takes ``model=`` : a registered name such as "jev-1.13", or a
DecisionModel instance such as ``JevModel("jev-1.13", timeout=60)``.
"""
from ._answers import DecisionModelError, DistAnswer, NoulAnswer, Response
from .choice import ChoiceEncoder, choice_bank, exemplar_options, stitch_blocks
from .estimators import ChoiceClassifier, NoulClassifier, ScoreRegressor
from .featurizer import QuestionFeaturizer
from .inspection import grouped_permutation_importance
from .models import (
    DEFAULT_MODEL,
    Capabilities,
    DecisionModel,
    FakeModel,
    JevAPIError,
    JevModel,
    register_model,
    resolve_model,
)
from .questions import choice, load_bank, noul, save_bank, score

__version__ = "0.1.0.dev0"

__all__ = [
    "QuestionFeaturizer",
    "ChoiceEncoder",
    "NoulClassifier",
    "ChoiceClassifier",
    "ScoreRegressor",
    "noul",
    "choice",
    "score",
    "load_bank",
    "save_bank",
    "choice_bank",
    "exemplar_options",
    "stitch_blocks",
    "grouped_permutation_importance",
    "DecisionModel",
    "Capabilities",
    "JevModel",
    "FakeModel",
    "DEFAULT_MODEL",
    "register_model",
    "resolve_model",
    "DecisionModelError",
    "JevAPIError",
    "NoulAnswer",
    "DistAnswer",
    "Response",
]
