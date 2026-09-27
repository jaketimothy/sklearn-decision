"""The answer schema every decision model normalizes to."""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["NoulAnswer", "DistAnswer", "Answer", "Response", "DecisionModelError",
           "answer_to_json", "answer_from_json"]


@dataclass(frozen=True)
class NoulAnswer:
    """P(true) for a noul question."""

    p: float


@dataclass(frozen=True)
class DistAnswer:
    """A distribution over a choice question's options or a score question's
    levels, in the order the question spec lists them."""

    probs: tuple[float, ...]
    confidence: float | None = None


Answer = NoulAnswer | DistAnswer


@dataclass(frozen=True)
class Response:
    """One model call's answers ({question name: answer}) and the concrete
    model version that produced them."""

    answers: dict[str, Answer]
    version: str


class DecisionModelError(RuntimeError):
    """A decision model failed to answer. ``fatal`` errors (bad credentials,
    misconfiguration) are raised even when ``on_error="nan"``."""

    def __init__(self, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def answer_to_json(ans: Answer) -> dict:
    if isinstance(ans, NoulAnswer):
        return {"noul": ans.p}
    return {"probs": list(ans.probs), "confidence": ans.confidence}


def answer_from_json(d: dict) -> Answer:
    if "noul" in d:
        return NoulAnswer(float(d["noul"]))
    conf = d.get("confidence")
    return DistAnswer(tuple(float(p) for p in d["probs"]), None if conf is None else float(conf))


def opt_float(v) -> float | None:
    """float(v), or None when v is missing or not finite."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None
