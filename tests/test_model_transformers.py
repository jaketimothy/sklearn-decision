"""TransformersModel against a tiny in-memory LM (plumbing), plus an opt-in
check against a real Hub model (``pytest -m hub``)."""
import pickle

import numpy as np
import pytest

import tiny_lm
from sklearn_decision import (
    ChoiceClassifier,
    ChoiceEncoder,
    NoulClassifier,
    QuestionFeaturizer,
    TransformersModel,
    choice,
    noul,
    resolve_model,
    score,
)
from sklearn_decision.models import transformers as tm

BANK = {
    "yes": noul("The text is about space."),
    "kind": choice("Which kind of text is it?", {"a": "first kind", "b": None, "c": "third"}),
    "size": score("How long is it?", ["short", "medium", "long"]),
}


@pytest.fixture
def load_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(tm, "load_pretrained", tiny_lm.install(calls))
    monkeypatch.setattr(tm, "_LOADED", {})
    return calls


def model(**kw):
    kw.setdefault("device", "cpu")
    return TransformersModel(tiny_lm.NAME, **kw)


def test_answers_are_valid_distributions(load_calls):
    X = tiny_lm.texts(5)
    f = QuestionFeaturizer(BANK, model=model(), score_repr="probs").fit(X)
    Z = f.transform(X)
    g = f.feature_groups_
    assert np.all((Z[:, g["yes"]] > 0) & (Z[:, g["yes"]] < 1))
    np.testing.assert_allclose(Z[:, g["kind"]].sum(axis=1), 1, rtol=1e-6)
    np.testing.assert_allclose(Z[:, g["size"]].sum(axis=1), 1, rtol=1e-6)
    assert f.model_.versions_seen == {f"{tiny_lm.NAME}@{tiny_lm.COMMIT}"}
    assert f.model_.usage["calls"] == 5 and f.model_.usage["input_tokens"] > 0


def test_batching_does_not_change_answers(load_calls):
    X = tiny_lm.texts(7)  # rows of different lengths, so batches are left-padded
    one = QuestionFeaturizer(BANK, model=model(batch_size=1), score_repr="probs").fit(X).transform(X)
    tm._LOADED.clear()
    from sklearn_decision import clear_memory_cache
    clear_memory_cache()
    many = QuestionFeaturizer(BANK, model=model(batch_size=16), score_repr="probs").fit(X).transform(X)
    np.testing.assert_allclose(one, many, atol=1e-5)


def test_readout_matches_manual_forward_pass(load_calls):
    import torch

    m = model().resolve()
    [res] = m.answer([("w1 w2 w3", {"q": BANK["yes"]})])
    tok, lm, _ = tm._LOADED[(tiny_lm.NAME, "main", "cpu", "auto", False)]
    with torch.no_grad():
        logits = lm(torch.tensor([tok.encode(m._prompt(tok, "w1 w2 w3", BANK["yes"]))])).logits[0, -1]
    ids = dict(zip(["Yes", "yes", "No", "no"], tok.convert_tokens_to_ids(["Yes", "yes", "No", "no"])))
    z_yes = torch.logsumexp(logits[[ids["Yes"], ids["yes"]]], 0)
    z_no = torch.logsumexp(logits[[ids["No"], ids["no"]]], 0)
    assert res.answers["q"].p == pytest.approx(float(torch.sigmoid(z_yes - z_no)), abs=1e-5)


def test_prompts_list_options_and_levels(load_calls):
    m = model().resolve()
    tok = tiny_lm.build()[0]
    p = m._prompt(tok, "STATE", BANK["kind"])
    assert "A. a: first kind\nB. b\nC. c: third" in p and p.index("STATE") < p.index("Question:")
    assert p.endswith("\nAnswer:")  # no chat template on the tiny tokenizer
    assert "0. short\n1. medium\n2. long" in m._prompt(tok, "S", BANK["size"])


def test_weights_load_once_and_are_shared_by_clones(load_calls):
    X = tiny_lm.texts(3)
    base = model()
    for seed_q in ("The text is about space.", "The text is long."):
        QuestionFeaturizer({"q": noul(seed_q)}, model=base).fit(X).transform(X)
    NoulClassifier("x", model=base).fit(X).predict(X)
    assert load_calls == [tiny_lm.NAME]


def test_fit_does_not_load_weights(load_calls):
    QuestionFeaturizer(BANK, model=model()).fit(tiny_lm.texts(3))
    assert load_calls == []


def test_namespace_tracks_revision_and_prompt():
    a = model().cache_namespace()
    assert a.startswith(f"hf:{tiny_lm.NAME}@main:")
    assert model(revision="abc").cache_namespace() != a
    assert model(templates={"noul": "{state}\n{instructions}"}).cache_namespace() != a
    assert model(batch_size=32).cache_namespace() == a  # batching doesn't change answers


def test_capabilities_block_choice_encoder(load_calls):
    X = tiny_lm.texts(4)
    enc = ChoiceEncoder([f"c{i}" for i in range(40)], model=model()).fit(X)
    assert len(enc.featurizer_.questions) == 2  # 26 + 14 options
    Z = enc.set_params(link="identity").fit(X).transform(X)
    for cols in enc._simplex_groups_:
        np.testing.assert_allclose(Z[:, cols].sum(axis=1), 1, rtol=1e-6)
    with pytest.raises(ValueError, match="limit of 26"):
        ChoiceClassifier("x", {f"c{i}": None for i in range(27)}, model=model()).fit(X)


def test_no_native_confidence():
    assert not hasattr(ChoiceClassifier("x", {"a": None, "b": None}, model=model()), "predict_confidence")


def test_long_prompts_fail_per_row(load_calls):
    X = ["w1 " * 600, "w2 w3"]
    f = QuestionFeaturizer({"q": BANK["yes"]}, model=model(), on_error="nan").fit(X)
    with pytest.warns(UserWarning, match="1 of 2"):
        Z = f.transform(X)
    assert np.isnan(Z[0]).all() and not np.isnan(Z[1]).any()
    g = QuestionFeaturizer({"q": BANK["yes"]}, model=model(max_state_tokens=50)).fit(X)
    assert not np.isnan(g.transform(X)).any()


def test_registry_and_validation():
    m = resolve_model("hf:google/gemma-4-12b-it@0a1b2c")
    assert isinstance(m, TransformersModel) and m.name == "google/gemma-4-12b-it" and m.revision == "0a1b2c"
    assert resolve_model("hf:org/x").revision == "main"
    with pytest.raises(ValueError, match="batch_size"):
        resolve_model(model(batch_size=0))
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_model(model(templates={"maybe": "x"}))


def test_fitted_estimator_pickles_without_weights(load_calls):
    X = tiny_lm.texts(3)
    f = QuestionFeaturizer(BANK, model=model()).fit(X)
    Z = f.transform(X)
    blob = pickle.dumps(f)
    assert len(blob) < 50_000
    np.testing.assert_allclose(pickle.loads(blob).transform(X), Z)


def test_option_permutations_make_answers_order_invariant(load_calls):
    X = tiny_lm.texts(4)
    abc = choice("Which kind?", {"a": None, "b": None, "c": None})
    cab = choice("Which kind?", {"c": None, "a": None, "b": None})  # a rotation of the same options

    def probs(model, spec):
        f = QuestionFeaturizer({"q": spec}, model=model).fit(X)
        Z = f.transform(X)
        names = list(f.get_feature_names_out())
        return Z[:, [names.index(f"q__{o}") for o in "abc"]]

    plain = model(n_option_permutations=1)
    assert np.abs(probs(plain, abc) - probs(plain, cab)).max() > 1e-4  # the random LM is order-sensitive
    averaged = model(n_option_permutations=3)
    np.testing.assert_allclose(probs(averaged, abc), probs(averaged, cab), atol=1e-6)
    np.testing.assert_allclose(probs(averaged, abc).sum(axis=1), 1, rtol=1e-6)


def test_option_permutations_leave_noul_and_score_alone_and_keep_old_cache_keys(load_calls):
    X = tiny_lm.texts(3)
    bank = {"yes": BANK["yes"], "size": BANK["size"]}
    one = QuestionFeaturizer(bank, model=model(n_option_permutations=1), score_repr="probs").fit(X).transform(X)
    four = QuestionFeaturizer(bank, model=model(n_option_permutations=4), score_repr="probs").fit(X).transform(X)
    np.testing.assert_allclose(one, four, atol=1e-6)
    assert model().n_option_permutations == 4
    assert model(n_option_permutations=1).cache_namespace() != model().cache_namespace()
    # caches written with n=1 (before the option existed) keep their keys: same digest as then
    import hashlib

    from sklearn_decision._cache import canon

    m = model(n_option_permutations=1)
    before = {"v": tm.TEMPLATE_VERSION, "t": m._templates(), "chat": "auto", "chat_kw": None, "prefix": "",
              "max_state": None}
    assert m.cache_namespace().endswith(hashlib.sha256(canon(before).encode()).hexdigest()[:16])
    with pytest.raises(ValueError, match="n_option_permutations"):
        resolve_model(model(n_option_permutations=0))


# ---------------- opt-in: a real model from the Hub ----------------

@pytest.mark.hub
def test_real_instruct_model_answers_obvious_questions():
    m = TransformersModel("Qwen/Qwen2.5-0.5B-Instruct", device="cpu", dtype="float32")
    X = ["The rocket launched from Cape Canaveral and reached orbit.",
         "The recipe needs two cups of flour and an egg."]
    f = QuestionFeaturizer({"space": noul("The text is about space travel."),
                            "topic": choice("What is the text about?", ["space", "cooking"])}, model=m).fit(X)
    Z = f.transform(X)
    names = list(f.get_feature_names_out())
    space, t_space, t_cook = (names.index(n) for n in ("space", "topic__space", "topic__cooking"))
    assert Z[0, space] > Z[1, space]
    assert Z[0, t_space] > Z[0, t_cook] and Z[1, t_cook] > Z[1, t_space]
