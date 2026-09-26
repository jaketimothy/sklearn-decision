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


def test_exemplar_self_match_renormalized_for_train_and_duplicates():
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
