"""Question constructors and validation.

Questions are plain dicts in the JSON shape the decision-model APIs use, so a
question bank can live in a version-controlled ``.json`` file.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["noul", "choice", "score", "validate_question", "load_bank", "save_bank", "QUESTION_TYPES"]

QUESTION_TYPES = ("noul", "choice", "score")


def noul(instructions: str) -> dict:
    """Yes/no question. Produces one column: P(true)."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, options: Sequence[str] | Mapping[str, str | None]) -> dict:
    """Pick-one question. ``options`` is a list of labels or {label: description}.
    Produces one probability column per option."""
    criteria = dict(options) if isinstance(options, Mapping) else {str(o): None for o in options}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: Sequence[str]) -> dict:
    """Ordinal rubric, levels ordered low -> high. Produces the expected level
    and/or one probability column per level (see ``score_repr``)."""
    return {"type": "score", "instructions": instructions, "criteria": [str(lv) for lv in levels]}


def validate_question(name: str, q: Mapping, *, max_choice_options: int | None = None,
                      max_score_levels: int | None = None) -> None:
    """Check one question's structure, and its size against a model's limits."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Question names must be non-empty strings, got {name!r}")
    if not isinstance(q, Mapping):
        raise TypeError(f"{name}: a question must be a dict, got {type(q).__name__}")
    t = q.get("type")
    if t not in QUESTION_TYPES:
        raise ValueError(f"{name}: type must be one of {QUESTION_TYPES}, got {t!r}")
    if not q.get("instructions"):
        raise ValueError(f"{name}: missing instructions")
    if t == "choice":
        c = q.get("criteria")
        if not isinstance(c, Mapping) or len(c) < 2:
            raise ValueError(f"{name}: choice needs at least 2 options as a dict")
        if max_choice_options is not None and len(c) > max_choice_options:
            raise ValueError(f"{name}: {len(c)} options exceeds this model's limit of {max_choice_options}")
    if t == "score":
        c = q.get("criteria")
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            raise ValueError(f"{name}: score needs at least 2 levels as a list")
        if max_score_levels is not None and len(c) > max_score_levels:
            raise ValueError(f"{name}: {len(c)} levels exceeds this model's limit of {max_score_levels}")


def load_bank(path: str | Path) -> dict[str, dict]:
    """Read a question bank ({name: spec}) from a JSON file and validate it."""
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(bank, dict):
        raise ValueError(f"{path}: a question bank must be a JSON object")
    for name, q in bank.items():
        validate_question(name, q)
    return bank


def save_bank(bank: Mapping[str, Mapping], path: str | Path) -> None:
    """Write a question bank to a JSON file (validated first)."""
    for name, q in bank.items():
        validate_question(name, q)
    Path(path).write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
