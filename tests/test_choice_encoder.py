import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from sklearn_decision import (
    ChoiceEncoder,
    FakeModel,
    QuestionFeaturizer,
    choice_bank,
    exemplar_options,
    stitch_blocks,
)

FEAT = QuestionFeaturizer(cache_path=None)
CONCEPTS = [f"c{i}" for i in range(10)]


def docs(n, seed=0):
    rng = np.random.default_rng(seed)
    return [f"document {i} about topic {rng.integers(5)}" for i in range(n)]


def test_choice_bank_blocks_and_anchor():
    bank = choice_bank(CONCEPTS, {"a": "A?", "b": "B?"}, anchor=("none", "nothing"), max_options=4)
    assert list(bank) == [f"choice_{v}_b{j}" for v in "ab" for j in range(4)]
    assert list(bank["choice_a_b0"]["criteria"]) == ["c0", "c1", "c2", "none"]
    assert list(bank["choice_a_b3"]["criteria"]) == ["c9", "none"]
    assert list(choice_bank(CONCEPTS, "Q?")) == ["choice_v0"]  # no limit: one block
    with pytest.raises(ValueError, match="collides"):
        choice_bank(["none", "x"], "Q?", anchor=("none", None))


def test_exemplar_options_stratifies_and_truncates():
    texts = ["x" * 1000] + docs(29)
    y = [0] * 10 + [1] * 10 + [2] * 10
    opts, idx = exemplar_options(texts, 6, y=y, max_chars=50, random_state=0)
    assert len(opts) == 6 and sorted(np.bincount(np.asarray(y)[idx])) == [2, 2, 2]
    assert all(len(v) <= 50 for v in opts.values())
    opts_all, idx_all = exemplar_options(texts[:3], 10)
    assert list(opts_all) == ["ex0", "ex1", "ex2"] and list(idx_all) == [0, 1, 2]


def test_concept_encoder_layout_and_clr():
    X = docs(12)
    enc = ChoiceEncoder(CONCEPTS, {"topic": "Topic?", "tone": "Tone?"}, model=FakeModel(), featurizer=FEAT)
    Z = enc.fit_transform(X)
    assert Z.shape == (12, 20)
    assert set(enc.feature_groups_) == {"topic", "tone"}
    for cols in enc.feature_groups_.values():
        np.testing.assert_allclose(Z[:, cols].sum(axis=1), 0, atol=1e-9)
    assert enc.get_feature_names_out()[0] == "choice_topic__c0"
    assert enc.model_.usage["calls"] == 12


def test_blocks_follow_model_limit_and_stitching_recovers_global_distribution():
    X = docs(8)
    one = ChoiceEncoder(CONCEPTS, "Topic?", anchor=("none", None), stitch=True, link="identity",
                        model=FakeModel(), featurizer=FEAT).fit(X)
    blocked = ChoiceEncoder(CONCEPTS, "Topic?", anchor=("none", None), stitch=True, link="identity",
                            model=FakeModel(max_choice_options=4), featurizer=FEAT).fit(X)
    assert len(one.featurizer_.questions) == 1 and len(blocked.featurizer_.questions) == 4
    # FakeModel obeys IIA, so stitched blocks equal the single big question minus the anchor
    np.testing.assert_allclose(blocked.transform(X), one.transform(X), rtol=1e-6)
    assert list(blocked.get_feature_names_out()) == [f"choice_v0__{c}" for c in CONCEPTS]


def test_public_stitch_blocks_matches_encoder():
    X = docs(5)
    enc = ChoiceEncoder(CONCEPTS, "Topic?", anchor=("none", None), link="identity",
                        model=FakeModel(max_choice_options=4), featurizer=FEAT).fit(X)
    raw = enc.transform(X)
    logp, labels = stitch_blocks(raw, enc.get_feature_names_out(), "choice_v0", "none")
    assert labels == CONCEPTS
    enc.set_params(stitch=True).fit(X)
    np.testing.assert_allclose(np.exp(logp), enc.transform(X), rtol=1e-4)


def test_stitch_requires_anchor():
    with pytest.raises(ValueError, match="anchor"):
        ChoiceEncoder(CONCEPTS, stitch=True, model=FakeModel(), featurizer=FEAT).fit(docs(3))


def test_max_options_cannot_exceed_model_limit():
    with pytest.raises(ValueError, match="exceeds"):
        ChoiceEncoder(CONCEPTS, max_options=9, model=FakeModel(max_choice_options=5), featurizer=FEAT).fit(docs(3))


def test_exemplar_self_match_renormalized_for_train_and_duplicates():  # default "reask" == this under IIA
    X = docs(20)
    enc = ChoiceEncoder("exemplars", n_exemplars=6, link="identity", random_state=0,
                        model=FakeModel(), featurizer=FEAT)
    Z = enc.fit_transform(X)
    raw = ChoiceEncoder("exemplars", n_exemplars=6, link="identity", random_state=0, self_match="keep",
                        model=FakeModel(), featurizer=FEAT).fit_transform(X)
    idx = enc.exemplar_indices_
    assert len(idx) == 6 and len(enc.codebook_) == 6
    for n, i in enumerate(idx):
        assert Z[i, n] == 0  # own option removed
        np.testing.assert_allclose(Z[i].sum(), 1)
        expected = raw[i].copy()
        expected[n] = 0
        np.testing.assert_allclose(Z[i], expected / expected.sum())  # IIA renormalization
    others = np.setdiff1d(np.arange(len(X)), idx)
    np.testing.assert_array_equal(Z[others], raw[others])
    # an identical row seen later gets the same treatment, so fit_transform == fit().transform()
    np.testing.assert_array_equal(enc.transform([X[idx[0]]]), Z[[idx[0]]])
    np.testing.assert_array_equal(enc.transform(X), Z)


def test_exemplar_default_size_uses_model_limit():
    enc = ChoiceEncoder("exemplars", model=FakeModel(max_choice_options=5), anchor=("none", None),
                        random_state=0, featurizer=FEAT).fit(docs(30))
    assert len(enc.codebook_) == 4  # limit minus the anchor
    with pytest.raises(ValueError, match="n_exemplars"):
        ChoiceEncoder("exemplars", model=FakeModel(), featurizer=FEAT).fit(docs(30))


def test_exemplar_encoder_in_cross_validation():
    X = docs(40)
    y = np.array([0, 1] * 20)
    pipe = make_pipeline(ChoiceEncoder("exemplars", n_exemplars=8, random_state=0, model=FakeModel(),
                                       featurizer=FEAT), LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,)


class PositionBiased(FakeModel):
    """Violates IIA: later options get a boost, so dropping an option shifts the rest."""

    def _answer_one(self, spec, state):
        ans = super()._answer_one(spec, state)
        if spec["type"] != "choice":
            return ans
        import math

        z = [math.log(p) + 0.4 * k for k, p in enumerate(ans.probs)]
        e = [math.exp(v - max(z)) for v in z]
        return type(ans)(tuple(v / sum(e) for v in e), ans.confidence)


def _encoders(model, **kw):
    common = dict(n_exemplars=6, link="identity", random_state=0, model=model, featurizer=FEAT, **kw)
    return {m: ChoiceEncoder("exemplars", self_match=m, **common) for m in ("reask", "renormalize", "keep")}


def test_reask_is_the_default_and_equals_renormalize_under_iia():
    X = docs(20)
    assert ChoiceEncoder().self_match == "reask"
    enc = _encoders(FakeModel())
    Z = {m: e.fit_transform(X) for m, e in enc.items()}
    np.testing.assert_allclose(Z["reask"], Z["renormalize"], rtol=1e-9, atol=1e-12)
    others = np.setdiff1d(np.arange(len(X)), enc["reask"].exemplar_indices_)
    np.testing.assert_array_equal(Z["reask"][others], Z["keep"][others])


def test_reask_asks_the_reduced_question_when_iia_fails():
    X = docs(20)
    enc = _encoders(PositionBiased())
    Z = {m: e.fit_transform(X) for m, e in enc.items()}
    reask = enc["reask"]
    (qname, spec), = reask.featurizer_.questions.items()
    labels = list(spec["criteria"])
    for n, i in enumerate(reask.exemplar_indices_):
        own = labels[n]
        reduced = {"type": "choice", "instructions": spec["instructions"],
                   "criteria": {lb: d for lb, d in spec["criteria"].items() if lb != own}}
        direct = QuestionFeaturizer({"r": reduced}, model=PositionBiased(), cache_path=None).fit(None)
        expected = np.insert(direct.transform([X[i]])[0], labels.index(own), 0.0)
        np.testing.assert_allclose(Z["reask"][i], expected, rtol=1e-9)
    # without IIA, renormalizing is not the same thing
    idx = reask.exemplar_indices_
    assert np.abs(Z["reask"][idx] - Z["renormalize"][idx]).max() > 1e-3


def test_reask_costs_one_extra_question_per_exemplar_row_and_view():
    X = docs(20)
    enc = ChoiceEncoder("exemplars", {"a": "A?", "b": "B?"}, n_exemplars=5, random_state=0, model=FakeModel(),
                        featurizer=FEAT).fit(X)
    enc.transform(X)
    usage = enc.model_.usage
    assert usage["answers_fetched"] == len(X) * 2 + 5 * 2
    calls = usage["calls"]
    enc.transform(X)  # re-asked answers are cached too
    assert enc.model_.usage["calls"] == calls


def test_reask_only_touches_the_block_holding_the_own_option():
    X = docs(12)
    kw = dict(n_exemplars=7, anchor=("none", None), link="identity", random_state=0, featurizer=FEAT)
    reask = ChoiceEncoder("exemplars", model=PositionBiased(max_choice_options=4), **kw).fit(X)
    keep = ChoiceEncoder("exemplars", model=PositionBiased(max_choice_options=4), self_match="keep", **kw).fit(X)
    Zr, Zk = reask.transform(X), keep.transform(X)
    groups = reask.featurizer_.feature_groups_
    for n, i in enumerate(reask.exemplar_indices_):
        own = list(reask.codebook_)[n]
        for q, cols in groups.items():
            touched = own in reask.featurizer_.questions[q]["criteria"]
            same = np.allclose(Zr[i, cols], Zk[i, cols])
            assert same != touched, (q, own)
            if touched:
                np.testing.assert_allclose(Zr[i, cols].sum(), 1.0)
