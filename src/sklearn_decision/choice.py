"""Choice-only encoders: the option set is the codebook.

A choice question over K options maps each row to a point on the K-simplex.
With a concept codebook (a taxonomy of labels) that point is a soft
classification; with an exemplar codebook (real rows as options, "which
reference is this most like?") it is a similarity profile against landmarks,
Nyström-style. Several *views* ask the same codebook under different framings.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._cache import canon
from ._state import prepare_states
from .featurizer import SEP, QuestionFeaturizer, UsageMixin, _input_tags, apply_link
from .models import DecisionModel, resolve_model

__all__ = ["ChoiceEncoder", "choice_bank", "exemplar_options", "stitch_blocks"]

_DEFAULT_VIEWS = {
    "concepts": "Which option best describes this?",
    "exemplars": "Which reference example is most similar to this one?",
}


def _block_plan(labels: list[str], per_block: int | None) -> list[list[str]]:
    if per_block is None or len(labels) <= per_block:
        return [labels]
    return [labels[i : i + per_block] for i in range(0, len(labels), per_block)]


def _qname(name: str, view: str, j: int, n_blocks: int) -> str:
    return f"{name}_{view}" if n_blocks == 1 else f"{name}_{view}_b{j}"


def choice_bank(options: Sequence[str] | Mapping[str, str | None], views: str | Mapping[str, str], *,
                name: str = "choice", anchor: tuple[str, str | None] | None = None,
                max_options: int | None = None) -> dict:
    """Build a choice-only question bank from one codebook and one or more views.

    Parameters
    ----------
    options : list of str or dict
        The codebook: labels, or ``{label: description}``.
    views : str or dict
        One question statement, or ``{view name: statement}``. Every view asks
        the same codebook under a different framing, so M views give M
        simplex embeddings of the same row.
    name : str
        Prefix of the question names.
    anchor : (label, description) or None
        Option appended to every block, so the blocks can be stitched back
        into one distribution (:func:`stitch_blocks`).
    max_options : int or None
        Largest question the model accepts, anchor included. None means no
        limit, so the codebook stays in one question.

    Returns
    -------
    dict
        ``{question name: spec}``. Names are ``{name}_{view}``, or
        ``{name}_{view}_b{j}`` when the codebook is split into blocks.
    """
    opts = dict(options) if isinstance(options, Mapping) else {str(o): None for o in options}
    if isinstance(views, str):
        views = {"v0": views}
    if anchor is not None and anchor[0] in opts:
        raise ValueError(f"anchor label {anchor[0]!r} collides with an option")
    per_block = None if max_options is None else max_options - (1 if anchor else 0)
    if per_block is not None and per_block < 1:
        raise ValueError("max_options leaves no room for options next to the anchor")
    blocks = _block_plan(list(opts), per_block)
    bank = {}
    for vname, stmt in views.items():
        for j, blk in enumerate(blocks):
            crit = {o: opts[o] for o in blk}
            if anchor is not None:
                crit[anchor[0]] = anchor[1]
            bank[_qname(name, vname, j, len(blocks))] = {"type": "choice", "instructions": stmt, "criteria": crit}
    return bank


def exemplar_options(texts: Sequence, k: int, *, y=None, max_chars: int = 400, label_prefix: str = "ex",
                     random_state=0):
    """Sample k rows as landmark options ("which reference is this most like?").

    Parameters
    ----------
    texts : sequence
        Candidate rows. Non-string rows are serialized as canonical JSON.
    k : int
        Number of exemplars; all rows if ``k >= len(texts)``.
    y : array-like or None
        Labels to stratify by, when every class can be represented.
    max_chars : int
        Exemplar text is truncated to this length.
    label_prefix : str
        Option labels are ``{label_prefix}{n}``.
    random_state : int, RandomState or None

    Returns
    -------
    options : dict
        ``{label: truncated text}``.
    indices : ndarray
        The sampled rows' positions in ``texts``.

    Notes
    -----
    If you build a bank from these by hand, drop ``indices`` from the
    downstream training set: an exemplar matches itself.
    :class:`ChoiceEncoder` handles that for you (``self_match``).
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(texts))
    if k >= len(texts):
        chosen = idx
    else:
        rng = check_random_state(random_state)
        strat = y if y is not None and k >= len(np.unique(y)) else None
        try:
            chosen, _ = train_test_split(idx, train_size=k, stratify=strat, random_state=rng)
        except ValueError:  # a class too small to stratify
            chosen, _ = train_test_split(idx, train_size=k, random_state=rng)
        chosen = np.sort(chosen)
    width = len(str(max(len(chosen) - 1, 0)))
    opts = {}
    for n, i in enumerate(chosen):
        t = texts[i] if isinstance(texts[i], str) else canon(texts[i])
        opts[f"{label_prefix}{n:0{width}d}"] = t[:max_chars]
    return opts, chosen


def _stitch_log(P: np.ndarray, blocks: Sequence[tuple[Sequence[int], int]], eps: float) -> np.ndarray:
    """Global log-distribution from blocks of (option columns, anchor column)."""
    cols = []
    for opt_cols, anchor_col in blocks:
        a = np.log(np.clip(P[:, anchor_col], eps, 1))[:, None]
        cols.append(np.log(np.clip(P[:, list(opt_cols)], eps, 1)) - a)
    Z = np.concatenate(cols, axis=1)
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def stitch_blocks(X, feature_names: Sequence[str], question_prefix: str, anchor_label: str, eps: float = 1e-6):
    """Rebuild one global log-distribution from blocked choice questions.

    Uses the anchor shared by every block: log p_i - log p_anchor is the same
    within-block log-ratio a single big question would have produced, *if*
    the model's option scores obey independence of irrelevant alternatives.
    Test that before trusting it; neither Jev nor Qwen2.5-0.5B fully did.

    Parameters
    ----------
    X : array-like of shape (n_rows, n_features)
        Identity-link features (probabilities).
    feature_names : sequence of str
        Column names, as from ``get_feature_names_out()``.
    question_prefix : str
        Blocked questions are those named ``{question_prefix}_b{j}``.
    anchor_label : str
        The option every block shares.
    eps : float
        Probabilities are clipped to ``[eps, 1]`` before taking logs.

    Returns
    -------
    log_probs : ndarray of shape (n_rows, K)
        Log-probabilities over all K codebook options.
    labels : list of str
        The options, in column order.
    """
    X = np.asarray(X, dtype=float)
    names = list(feature_names)
    qnames = sorted({n.split(SEP, 1)[0] for n in names if n.startswith(question_prefix + "_b") and SEP in n})
    if not qnames:
        raise ValueError(f"no blocked questions start with {question_prefix!r}_b")
    blocks, labels = [], []
    for q in qnames:
        anchor_col = names.index(f"{q}{SEP}{anchor_label}")
        opt_cols = []
        for c, n in enumerate(names):
            head, _, opt = n.partition(SEP)
            if head == q and opt not in (anchor_label, "confidence"):
                opt_cols.append(c)
                labels.append(opt)
        blocks.append((opt_cols, anchor_col))
    return _stitch_log(X, blocks, eps), labels


def _state_hash(state) -> str:
    return hashlib.sha256(canon(state).encode()).hexdigest()


class ChoiceEncoder(UsageMixin, TransformerMixin, BaseEstimator):
    """Encode rows as distributions over a codebook of options.

    Parameters
    ----------
    codebook : list, dict or "exemplars"
        The options: labels, {label: description}, or "exemplars" to sample
        ``n_exemplars`` training rows in ``fit`` and use them as options.
    views : str, dict or None
        One question statement, or {view name: statement}. None uses a
        generic statement suited to the codebook type.
    n_exemplars : int or None
        Exemplar codebook size. None fills one question: the model's option
        limit minus the anchor. Required when the model has no limit.
    anchor : (label, description) or None
        Extra option added to every block, needed for ``stitch``.
    max_options : int or None
        Block size (anchor included). Defaults to the model's option limit;
        None with no model limit keeps the codebook in one question.
    max_exemplar_chars : int
        Exemplar text is truncated to this length in the option description.
    self_match : {"reask", "renormalize", "keep"}
        A row that is itself an exemplar trivially picks its own option. This
        applies to training rows and to any identical row seen later, so
        exemplars can stay in the training set.

        - "reask" (default): ask that row the affected question again with
          its own option removed. Exact; costs one extra question per
          exemplar row and view.
        - "renormalize": zero the own option and renormalize the rest of its
          simplex. Free, but only valid if the model obeys independence of
          irrelevant alternatives, which neither Jev nor Qwen2.5-0.5B did in
          our behaviour checks (benchmarks/RESULTS.md).
        - "keep": leave answers untouched (for ablations).
    link : {"clr", "logit", "identity"}
        Output scale; "clr" is the natural geometry for compositions.
    stitch : bool
        Output one distribution over the whole codebook per view, rebuilt
        from the blocks through the anchor (requires ``anchor``; assumes IIA).
    random_state : int, RandomState or None
        Exemplar sampling.
    model : str or DecisionModel
        The decision model (required), as for :class:`QuestionFeaturizer`.
    featurizer : QuestionFeaturizer or None
        Template for cache, state and error settings (``cache_path``,
        ``state_columns``, ``state_fn``, ``on_error``, ``logit_eps``).

    Attributes
    ----------
    codebook_ : dict[str, str | None]
    exemplar_indices_ : ndarray
        Training-row indices used as exemplars (empty for concept codebooks).
    featurizer_ : QuestionFeaturizer
        Answers the underlying bank (``featurizer_.questions``).
    model_ : DecisionModel
    feature_names_out_ : list[str]
    feature_groups_ : dict[str, list[int]]
        View -> its output columns.

    Warnings
    --------
    Exemplar text is sent to the model as option descriptions, so it is as
    untrusted as the state itself.
    """

    def __init__(self, codebook=None, views: str | Mapping[str, str] | None = None, *,
                 n_exemplars: int | None = None, anchor: tuple[str, str | None] | None = None,
                 max_options: int | None = None, max_exemplar_chars: int = 400,
                 self_match: str = "reask", link: str = "clr", stitch: bool = False,
                 random_state=None, model: str | DecisionModel | None = None,
                 featurizer: QuestionFeaturizer | None = None):
        self.codebook = codebook
        self.views = views
        self.n_exemplars = n_exemplars
        self.anchor = anchor
        self.max_options = max_options
        self.max_exemplar_chars = max_exemplar_chars
        self.self_match = self_match
        self.link = link
        self.stitch = stitch
        self.random_state = random_state
        self.model = model
        self.featurizer = featurizer

    def fit(self, X, y=None):
        """Build the codebook (sampling exemplars if asked) and the question
        bank. No model calls."""
        self._validate_params()
        base = self.featurizer if self.featurizer is not None else QuestionFeaturizer()
        if not isinstance(base, QuestionFeaturizer):
            raise TypeError(f"featurizer must be a QuestionFeaturizer or None, got {type(base).__name__}")
        caps = resolve_model(self.model).capabilities()
        limit = self.max_options if self.max_options is not None else caps.max_choice_options
        if self.max_options is not None and caps.max_choice_options is not None \
                and self.max_options > caps.max_choice_options:
            raise ValueError(f"max_options={self.max_options} exceeds the model's limit "
                             f"of {caps.max_choice_options}")

        states = None
        if X is not None:
            states = prepare_states(self, X, reset=True, state_columns=base.state_columns,
                                    state_fn=base.state_fn)
        else:
            self.__dict__.pop("n_features_in_", None)
            self.__dict__.pop("feature_names_in_", None)

        self._exemplar_keys_: dict[str, list[str]] = {}
        if isinstance(self.codebook, str):  # "exemplars", checked in _validate_params
            if states is None:
                raise ValueError("codebook='exemplars' samples from X; X must not be None")
            k = self.n_exemplars
            if k is None:
                if limit is None:
                    raise ValueError("Set n_exemplars: this model has no option limit to default to")
                k = limit - (1 if self.anchor is not None else 0)
            if y is not None:
                y = np.asarray(y)
                if len(y) != len(states):
                    raise ValueError(f"X has {len(states)} rows but y has {len(y)}")
                if y.ndim != 1:
                    y = None  # only used to stratify; multi-output targets are ignored
            opts, idx = exemplar_options(states, k, y=y, max_chars=self.max_exemplar_chars,
                                         random_state=self.random_state)
            if len(opts) < 2:
                raise ValueError(f"An exemplar codebook needs at least 2 rows; got {len(states)} sample(s)")
            for label, i in zip(opts, idx):
                self._exemplar_keys_.setdefault(_state_hash(states[i]), []).append(label)
            self.exemplar_indices_ = np.asarray(idx)
            default_view = _DEFAULT_VIEWS["exemplars"]
        else:
            opts = (dict(self.codebook) if isinstance(self.codebook, Mapping)
                    else {str(o): None for o in self.codebook})
            if len(opts) < 2:
                raise ValueError("codebook needs at least 2 options")
            self.exemplar_indices_ = np.array([], dtype=int)
            default_view = _DEFAULT_VIEWS["concepts"]
        self.codebook_ = opts

        views = self.views if self.views is not None else default_view
        views = {"v0": views} if isinstance(views, str) else dict(views)
        bank = choice_bank(opts, views, name="choice", anchor=self.anchor, max_options=limit)
        feat = clone(base).set_params(questions=bank, model=self.model, link="identity", drop_redundant=False,
                                      include_confidence=False, score_repr="probs")
        self.featurizer_ = feat.fit(None)
        self.model_ = feat.model_
        self._layout(views, list(opts), limit)
        return self

    def transform(self, X) -> np.ndarray:
        """Each row's distribution over the codebook, per view, on the ``link`` scale.

        Parameters
        ----------
        X : array-like of shape (n_samples,), DataFrame or 2-D array

        Returns
        -------
        ndarray of shape (n_samples, n_features_out)
            Columns as in ``get_feature_names_out()``.
        """
        check_is_fitted(self, "featurizer_")
        base = self.featurizer_
        states = prepare_states(self, X, reset=False, state_columns=base.state_columns, state_fn=base.state_fn)
        P = base._answer_matrix(states)
        if self.self_match == "reask" and self._exemplar_keys_:
            self._reask_self_matches(P, states)
        if self.stitch:
            out = np.exp(np.concatenate([_stitch_log(P, blocks, base.logit_eps_)
                                         for blocks in self._stitch_plan_], axis=1))
        else:
            out = P
        if self.self_match == "renormalize" and self._exemplar_keys_:
            self._renormalize_self_matches(out, states)
        mask = np.ones(out.shape[1], dtype=bool)
        return apply_link(out, self.link, mask, self._simplex_groups_, base.logit_eps_)

    def get_feature_names_out(self, input_features=None):
        """Output column names, ``choice_{view}__{option}``.

        Parameters
        ----------
        input_features : None
            Ignored; checked against the fitted input only.

        Returns
        -------
        ndarray of str
        """
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and hasattr(self, "n_features_in_") \
                and len(input_features) != self.n_features_in_:
            raise ValueError(f"input_features should have length equal to number of features "
                             f"({self.n_features_in_}), got {len(input_features)}")
        return np.asarray(self.feature_names_out_, dtype=object)

    def __sklearn_tags__(self):
        return _input_tags(super().__sklearn_tags__(), self.model)

    # ---------------- internals ----------------

    def _validate_params(self) -> None:
        cb = self.codebook
        if isinstance(cb, str):
            if cb != "exemplars":
                raise ValueError(f"codebook must be a list, a dict or 'exemplars', got {cb!r}")
        elif cb is None or not isinstance(cb, (Mapping, Sequence, np.ndarray)):
            raise ValueError("codebook must be a list, a dict or 'exemplars'")
        if self.self_match not in ("reask", "renormalize", "keep"):
            raise ValueError(f"self_match must be 'reask', 'renormalize' or 'keep', got {self.self_match!r}")
        if self.link not in ("identity", "logit", "clr"):
            raise ValueError(f"link must be 'identity', 'logit' or 'clr', got {self.link!r}")
        if self.anchor is not None and (not isinstance(self.anchor, (tuple, list)) or len(self.anchor) != 2):
            raise ValueError("anchor must be a (label, description) pair or None")
        if self.stitch and self.anchor is None:
            raise ValueError("stitch=True needs an anchor option shared by every block")
        if self.n_exemplars is not None and (not isinstance(self.n_exemplars, (int, np.integer))
                                             or self.n_exemplars < 2):
            raise ValueError("n_exemplars must be an int >= 2 or None")

    def _layout(self, views: dict, labels: list[str], limit: int | None) -> None:
        """Output columns, simplex groups and the self-match lookup."""
        fgroups = self.featurizer_.feature_groups_
        fnames = self.featurizer_.feature_names_out_
        per_block = None if limit is None else limit - (1 if self.anchor is not None else 0)
        blocks = _block_plan(labels, per_block)
        names, groups, simplex, self_cols, stitch_plan = [], {}, [], {}, []
        for v in views:
            if self.stitch:
                plan = []
                for j, blk in enumerate(blocks):
                    cols = fgroups[_qname("choice", v, j, len(blocks))]
                    plan.append((cols[: len(blk)], cols[len(blk)]))  # anchor is last
                stitch_plan.append(plan)
                start = len(names)
                vcols = list(range(start, start + len(labels)))
                names += [f"choice_{v}{SEP}{lb}" for lb in labels]
                simplex.append(vcols)
                for lb, c in zip(labels, vcols):
                    self_cols.setdefault(lb, []).append((c, len(simplex) - 1))
                groups[v] = vcols
            else:
                vcols = []
                for j, blk in enumerate(blocks):
                    cols = fgroups[_qname("choice", v, j, len(blocks))]
                    simplex.append(cols)
                    for lb, c in zip(blk, cols):
                        self_cols.setdefault(lb, []).append((c, len(simplex) - 1))
                    vcols += cols
                groups[v] = vcols
        if not self.stitch:
            names = list(fnames)
        self.feature_names_out_ = names
        self.feature_groups_ = groups
        self._simplex_groups_ = simplex
        self._self_cols_ = self_cols
        self._stitch_plan_ = stitch_plan

    def _reask_self_matches(self, P: np.ndarray, states: list) -> None:
        """For rows that are exemplars, replace each affected question's answer
        with the answer to the same question minus the row's own option(s)."""
        feat = self.featurizer_
        rows, row_questions = [], []
        for i, s in enumerate(states):
            own = set(self._exemplar_keys_.get(_state_hash(s), ()))
            if not own:
                continue
            questions = {}
            for q, spec in feat.questions.items():
                labels = list(spec["criteria"])
                if own.isdisjoint(labels):
                    continue
                cols = feat.feature_groups_[q]
                kept = [lb for lb in labels if lb not in own]
                P[i, cols] = 0.0
                if len(kept) >= 2:
                    questions[q] = {**spec, "criteria": {lb: spec["criteria"][lb] for lb in kept}}
                elif kept:  # only one option left: it gets all the mass
                    P[i, cols[labels.index(kept[0])]] = 1.0
            if questions:
                rows.append(i)
                row_questions.append(questions)
        if not rows:
            return
        found = feat._fetch([states[i] for i in rows], row_questions)
        for i, questions, answers in zip(rows, row_questions, found):
            for q, spec in questions.items():
                cols = feat.feature_groups_[q]
                if q not in answers:  # failed with on_error="nan"
                    P[i, cols] = np.nan
                    continue
                labels = list(feat.questions[q]["criteria"])
                for lb, p in zip(spec["criteria"], answers[q][0].probs):
                    P[i, cols[labels.index(lb)]] = p

    def _renormalize_self_matches(self, out: np.ndarray, states: list) -> None:
        for i, s in enumerate(states):
            labels = self._exemplar_keys_.get(_state_hash(s))
            if not labels:
                continue
            touched = set()
            for lb in labels:
                for col, g in self._self_cols_.get(lb, ()):
                    out[i, col] = 0.0
                    touched.add(g)
            for g in touched:
                cols = self._simplex_groups_[g]
                tot = out[i, cols].sum()
                if tot > 0:
                    out[i, cols] /= tot
