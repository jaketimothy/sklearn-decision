"""The decision-model protocol and the name registry behind ``model="..."``."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator, clone

from .._answers import Response

__all__ = ["Capabilities", "DecisionModel", "register_model", "resolve_model", "registered_prefixes"]


@dataclass(frozen=True)
class Capabilities:
    """What a decision model can do. Estimators read limits from here rather
    than hard-coding one vendor's numbers.

    max_choice_options, max_score_levels : int or None
        Largest choice/score question the model accepts (None = no limit).
    max_questions_per_call : int or None
        Questions sent per request; each request re-sends the state
        (None = the whole bank in one request).
    native_confidence : bool
        Whether choice/score answers carry the model's own confidence.
    deterministic : bool
        Whether identical requests return identical answers.
    probability_resolution : float or None
        Step to which the model rounds reported probabilities (Jev: 0.01),
        or None for full precision. Log-odds features clip at half a step,
        so a rounded 0.0 doesn't become an outlier.
    """

    max_choice_options: int | None = None
    max_score_levels: int | None = None
    max_questions_per_call: int | None = None
    native_confidence: bool = False
    deterministic: bool = False
    probability_resolution: float | None = None


class DecisionModel(BaseEstimator, ABC):
    """Base class for decision-model backends.

    A backend is a scikit-learn estimator in the ``get_params`` sense only, so
    it can be a nested parameter (``model__name``, ``model__timeout``) that
    ``clone`` and grid search understand. Estimators never mutate the instance
    passed as their ``model`` parameter: they call :func:`resolve_model`, which
    returns a fresh, validated copy stored as ``model_``.

    Runtime state (usage counters, versions seen) lives on that copy.
    """

    def resolve(self) -> DecisionModel:
        """Validate parameters and pin anything that must not drift, such as a
        model revision. Called once per owning estimator's ``fit``. Must not
        need credentials or network access."""
        return self

    @abstractmethod
    def cache_namespace(self) -> str:
        """Everything that changes this model's answers (name, pinned revision,
        prompt template...). Part of every cache key."""

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What this model can do: option and level limits, questions per
        request, native confidence, determinism, probability rounding."""

    @abstractmethod
    def answer(self, items: Sequence[tuple[Any, Mapping[str, dict]]]) -> list[Response | BaseException]:
        """Answer a batch of ``(state, {question name: spec})`` requests.

        Returns one entry per item, in order: a :class:`Response`, or the
        exception that item failed with. Raise instead of returning only for
        errors that doom the whole batch (e.g. missing credentials).
        """

    def estimate_cost(self, states: Sequence, requests: Sequence[Mapping[str, dict]]) -> dict:
        """Rough cost of answering every ``requests`` chunk for every state
        with an empty cache. Backends with a price override this."""
        return {"rows": len(states), "requests": len(states) * len(requests)}

    def cost_usd(self) -> float | None:
        """What this instance's requests have cost so far, or None for models
        without a price."""
        return None

    #: Requests per call to :meth:`answer`. Estimators cache each chunk's
    #: answers and report progress as it lands, so a smaller chunk loses less
    #: to an interruption; a larger one lets a backend overlap more requests.
    chunk_size: int = 32

    @property
    def usage(self) -> dict:
        """Running totals: calls, answers fetched, cache hits, tokens."""
        return self.__dict__.setdefault(
            "_usage",
            {"calls": 0, "cache_hits": 0, "answers_fetched": 0, "input_tokens": 0, "output_tokens": 0},
        )

    @property
    def versions_seen(self) -> set[str]:
        """Concrete model versions behind the answers this instance served."""
        return self.__dict__.setdefault("_versions_seen", set())


_REGISTRY: dict[str, Callable[[str], DecisionModel]] = {}


def register_model(prefix: str, factory: Callable[[str], DecisionModel]) -> None:
    """Make ``model="<prefix>..."`` resolve to ``factory(name)``."""
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    _REGISTRY[prefix] = factory


def registered_prefixes() -> list[str]:
    return sorted(_REGISTRY)


def resolve_model(model: str | DecisionModel | None) -> DecisionModel:
    """A fresh, resolved backend for an estimator's ``model`` parameter."""
    if model is None:
        raise ValueError(
            "Choose a decision model, e.g. model='hf:<repo id>' (local open weights; nothing leaves "
            "the machine) or model='jev-latest' (TypeSafe's hosted API: rows are sent to TypeSafe), or pass "
            f"a DecisionModel instance. Registered prefixes: {registered_prefixes()}"
        )
    if isinstance(model, DecisionModel):
        return clone(model).resolve()
    if isinstance(model, str):
        matches = [p for p in _REGISTRY if model.startswith(p)]
        if not matches:
            raise ValueError(
                f"Unknown model {model!r}. Registered prefixes: {registered_prefixes()}. "
                "Pass a DecisionModel instance or call register_model()."
            )
        return _REGISTRY[max(matches, key=len)](model).resolve()
    raise TypeError(f"model must be a string or a DecisionModel, got {type(model).__name__}")


def model_capabilities(model: str | DecisionModel | None) -> Capabilities | None:
    """Capabilities for an unresolved ``model`` parameter, or None if it is invalid.
    Used by ``__sklearn_tags__``, which must not raise."""
    try:
        return resolve_model(model).capabilities()
    except Exception:
        return None
