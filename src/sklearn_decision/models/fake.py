"""A deterministic, offline decision model for tests, docs and dry runs."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .._answers import DistAnswer, NoulAnswer, Response
from .._cache import canon
from .base import Capabilities, DecisionModel

__all__ = ["FakeModel"]


class FakeModel(DecisionModel):
    """Answers derived from a hash of (seed, question, state), no network.

    Each choice option and score level gets an independent logistic score
    from ``hash(seed, instructions, state, label)``; the distribution is their
    softmax. Because an option's score ignores the other options, the model
    satisfies independence of irrelevant alternatives exactly, which makes it
    a clean reference for testing blocking and stitching.

    Parameters
    ----------
    seed : int
    version : str
        Reported as the concrete model version of every answer.
    temperature : float
        Softmax temperature; lower is more peaked.
    max_choice_options, max_score_levels, max_questions_per_call : int or None
        Limits to report through ``capabilities()``, to exercise blocking and
        chunking.
    native_confidence : bool
        Whether answers carry a confidence (the top probability).
    """

    def __init__(self, seed: int = 0, *, version: str = "fake-1", temperature: float = 1.0,
                 max_choice_options: int | None = None, max_score_levels: int | None = None,
                 max_questions_per_call: int | None = None, native_confidence: bool = True):
        self.seed = seed
        self.version = version
        self.temperature = temperature
        self.max_choice_options = max_choice_options
        self.max_score_levels = max_score_levels
        self.max_questions_per_call = max_questions_per_call
        self.native_confidence = native_confidence

    def resolve(self) -> FakeModel:
        if not self.temperature or self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        return self

    def cache_namespace(self) -> str:
        return f"fake:{self.version}:{self.seed}:{self.temperature}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            max_choice_options=self.max_choice_options,
            max_score_levels=self.max_score_levels,
            max_questions_per_call=self.max_questions_per_call,
            native_confidence=self.native_confidence,
            deterministic=True,
        )

    def answer(self, items: Sequence[tuple[Any, Mapping[str, dict]]]) -> list[Response | BaseException]:
        out = []
        for state, questions in items:
            s = canon(state)
            answers = {q: self._answer_one(spec, s) for q, spec in questions.items()}
            self.usage["calls"] += 1
            self.usage["answers_fetched"] += len(answers)
            out.append(Response(answers, self.version))
        return out

    def _z(self, *parts: str) -> float:
        """A standard-logistic draw determined by ``parts``."""
        h = hashlib.sha256("\x1f".join((str(self.seed), *parts)).encode()).digest()
        u = (int.from_bytes(h[:8], "big") + 0.5) / 2**64
        return math.log(u / (1 - u))

    def _answer_one(self, spec: Mapping, state: str):
        instr = spec["instructions"]
        if spec["type"] == "noul":
            return NoulAnswer(1 / (1 + math.exp(-self._z("noul", instr, state) / self.temperature)))
        labels = list(spec["criteria"]) if spec["type"] == "choice" else [str(lv) for lv in spec["criteria"]]
        z = [self._z(spec["type"], instr, state, str(lb)) / self.temperature for lb in labels]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        tot = sum(e)
        probs = tuple(v / tot for v in e)
        return DistAnswer(probs, max(probs) if self.native_confidence else None)
