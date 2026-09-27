import pytest

from sklearn_decision import choice, load_bank, noul, save_bank, score
from sklearn_decision.questions import validate_question


def test_constructors_match_wire_shape():
    assert noul("x") == {"type": "noul", "instructions": "x"}
    assert choice("x", ["a", "b"]) == {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}
    assert choice("x", {"a": "desc", "b": None})["criteria"] == {"a": "desc", "b": None}
    assert score("x", ["lo", "hi"]) == {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]}


@pytest.mark.parametrize("q, msg", [
    ({"type": "maybe", "instructions": "x"}, "type must be"),
    ({"type": "noul"}, "missing instructions"),
    (choice("x", ["only"]), "at least 2 options"),
    (score("x", ["only"]), "at least 2 levels"),
])
def test_validate_rejects_malformed(q, msg):
    with pytest.raises(ValueError, match=msg):
        validate_question("q", q)


def test_validate_applies_model_limits():
    validate_question("q", choice("x", list("abc")), max_choice_options=3)
    with pytest.raises(ValueError, match="limit of 2"):
        validate_question("q", choice("x", list("abc")), max_choice_options=2)
    with pytest.raises(ValueError, match="limit of 2"):
        validate_question("q", score("x", list("abc")), max_score_levels=2)


def test_bank_roundtrip(tmp_path):
    bank = {"a": noul("x"), "b": choice("y", ["p", "q"])}
    save_bank(bank, tmp_path / "bank.json")
    assert load_bank(tmp_path / "bank.json") == bank
