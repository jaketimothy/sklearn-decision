"""QuestionFeaturizer: a decision model as a semantic encoder for scikit-learn.

Each input row (a string, or a JSON-able record) is sent to the model as
``state`` together with a *question bank*. The model's answer probabilities
become numeric feature columns, so any sklearn estimator can sit downstream.

    from sklearn_decision import QuestionFeaturizer, noul, choice, score
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier

    feat = QuestionFeaturizer({
        "asks_refund": noul("The customer asks for money back."),
        "product": choice("Which product is discussed?", ["app", "api", "billing"]),
        "urgency": score("How urgent is the request?", ["can wait", "this week", "today"]),
    }, model="hf:google/gemma-4-12b-it")
    clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)

The featurizer is stateless (``fit`` makes no model calls) and row-wise, so it
cannot leak labels across CV folds. Every answer is cached per
(model, question, state), so re-running, cross-validating, or adding and
pruning questions only pays for answers it has never seen.
"""
from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._answers import DecisionModelError, DistAnswer, NoulAnswer
from ._cache import AnswerCache, answer_key, canon, question_key
from ._state import prepare_states, states_only
from .models import DecisionModel, resolve_model
from .models.base import model_capabilities
from .questions import validate_question

__all__ = ["QuestionFeaturizer", "SEP", "apply_link"]

SEP = "__"
_LINKS = ("identity", "logit", "clr")


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def apply_link(out: np.ndarray, link: str, prob_mask: np.ndarray, simplex_groups: Sequence[Sequence[int]],
               eps: float) -> np.ndarray:
    """Map probability columns to the output scale, in place.

    ``logit`` maps every probability column to log-odds. ``clr`` maps each
    simplex group to its centred log-ratio and every other probability column
    to log-odds.
    """
    if link == "logit":
        out[:, prob_mask] = _logit(out[:, prob_mask], eps)
    elif link == "clr":
        done = np.zeros(out.shape[1], dtype=bool)
        for g in simplex_groups:
            lp = np.log(np.clip(out[:, g], eps, 1.0))
            out[:, g] = lp - lp.mean(axis=1, keepdims=True)
            done[g] = True
        rest = prob_mask & ~done
        out[:, rest] = _logit(out[:, rest], eps)
    return out


class UsageMixin:
    """Adds ``usage()`` to an estimator whose fitted model is ``model_``."""

    def usage(self) -> dict:
        """What this fitted estimator has cost so far.

        Returns a dict with ``calls`` (model requests), ``answers_fetched``,
        ``cache_hits``, ``input_tokens`` and ``output_tokens``, plus
        ``versions``: the concrete model versions whose answers it served.
        Counts cover this estimator's model since ``fit``; answers served
        from the cache cost nothing.
        """
        check_is_fitted(self, "model_")
        return {**self.model_.usage, "versions": sorted(self.model_.versions_seen)}


def _input_tags(tags, model) -> Any:
    """Input tags shared by every estimator in the package.

    1-D input (a list of documents) is accepted, but ``one_d_array`` means
    "1-D only" to sklearn's checks, so it stays False, as on CountVectorizer.
    """
    tags.input_tags.string = True
    tags.input_tags.dict = True
    tags.input_tags.allow_nan = True
    caps = model_capabilities(model)
    tags.non_deterministic = caps is not None and not caps.deterministic
    return tags


class QuestionFeaturizer(UsageMixin, TransformerMixin, BaseEstimator):
    """Turn unstructured rows into a numeric matrix of decision-model answers.

    Parameters
    ----------
    questions : dict[str, dict]
        Question bank: name -> spec from ``noul`` / ``choice`` / ``score``
        (or the equivalent JSON). The bank *is* the encoder; tune it like a
        hyperparameter.
    model : str or DecisionModel
        The decision model; required. A string resolves through the model
        registry: "hf:<repo id>[@revision]" -> a local open-weights
        ``TransformersModel``, "jev-latest" -> ``JevModel`` (hosted: rows are
        sent to TypeSafe). Pass an instance to configure it; its parameters
        are nested (``model__timeout``).
    link : {"identity", "logit", "clr"}
        "logit" maps every probability column to log-odds, which suits linear
        models. "clr" (centred log-ratio) treats each choice/score question as
        a point on the simplex: log p minus the mean log p over that
        question's options. Use it when a choice question is the embedding.
        Noul columns still get logit. Trees don't care.
    score_repr : {"ev", "probs", "both"}
        Score questions become the expected level (0..L-1), the per-level
        probabilities, or both.
    drop_redundant : bool
        Drop the last option/level column of each simplex (sums to 1). Use for
        unregularized linear models.
    include_confidence : bool
        Add the model's confidence for choice/score questions as an extra
        column (NaN if the model reports none).
    state_columns : list[str] or None
        For DataFrame input, which columns go into the JSON state (default all).
    state_fn : callable or None
        Applied to each row before sending, e.g. to trim or template text.
        Use a module-level function so the estimator stays picklable.
    cache_path : str or None
        SQLite file for a persistent answer cache. None (default) keeps
        answers in a process-wide in-memory cache, shared by clones and
        grid-search candidates; nothing is written to disk.
    on_error : {"raise", "nan"}
        On a failed row, raise (after caching the successes) or emit NaNs.
        HistGradientBoosting handles NaN natively. Fatal errors such as bad
        credentials always raise.
    logit_eps : float or None
        Probabilities are clipped to [eps, 1 - eps] before logit/clr. None
        (default) uses half the model's ``probability_resolution`` (0.005 for
        Jev, which rounds to 0.01), else 1e-4. Resolved as ``logit_eps_``.
    verbose : bool

    Attributes
    ----------
    model_ : DecisionModel
        The resolved model. ``model_.usage`` holds call, cache-hit and token
        totals; ``model_.versions_seen`` the concrete versions behind answers.
    cache_ : AnswerCache
    feature_names_out_ : list[str]
    feature_groups_ : dict[str, list[int]]
        Question -> its column indices.
    n_features_in_, feature_names_in_ :
        Set for DataFrame and 2-D input only; 1-D text input has neither.
    """

    def __init__(
        self,
        questions: Mapping[str, Mapping] | None = None,
        *,
        model: str | DecisionModel | None = None,
        link: str = "identity",
        score_repr: str = "ev",
        drop_redundant: bool = False,
        include_confidence: bool = False,
        state_columns: Sequence[str] | None = None,
        state_fn: Callable[[Any], Any] | None = None,
        cache_path: str | None = None,
        on_error: str = "raise",
        logit_eps: float | None = None,
        verbose: bool = False,
    ):
        self.questions = questions
        self.model = model
        self.link = link
        self.score_repr = score_repr
        self.drop_redundant = drop_redundant
        self.include_confidence = include_confidence
        self.state_columns = state_columns
        self.state_fn = state_fn
        self.cache_path = cache_path
        self.on_error = on_error
        self.logit_eps = logit_eps
        self.verbose = verbose

    # ---------------- sklearn API ----------------

    def fit(self, X, y=None):
        """Validate the question bank and lay out the output columns.

        No model calls happen here. ``X`` is only used to record input
        features, and may be None.
        """
        self._validate_params()
        self.model_ = resolve_model(self.model)
        caps = self.model_.capabilities()
        self.logit_eps_ = (self.logit_eps if self.logit_eps is not None
                           else caps.probability_resolution / 2 if caps.probability_resolution else 1e-4)
        for name, q in self.questions.items():
            validate_question(name, q, max_choice_options=caps.max_choice_options,
                              max_score_levels=caps.max_score_levels)

        names, groups, prob_mask, simplex = [], {}, [], []
        for qname, q in self.questions.items():
            cols = self._layout(qname, q)
            groups[qname] = list(range(len(names), len(names) + len(cols)))
            if q["type"] != "noul":
                simplex.append([groups[qname][k] for k, (_, is_p) in enumerate(cols) if is_p])
            for col, is_prob in cols:
                names.append(col)
                prob_mask.append(is_prob)
        self.feature_names_out_ = names
        self.feature_groups_ = groups
        self._prob_mask_ = np.array(prob_mask, dtype=bool)
        self._simplex_groups_ = [g for g in simplex if len(g) > 1]
        self.cache_ = AnswerCache(self.cache_path)
        if X is not None:
            prepare_states(self, X, reset=True, state_columns=self.state_columns, state_fn=None)
        else:
            self.__dict__.pop("n_features_in_", None)
            self.__dict__.pop("feature_names_in_", None)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        states = prepare_states(self, X, reset=False, state_columns=self.state_columns,
                                state_fn=self.state_fn)
        out = self._answer_matrix(states)
        return apply_link(out, self.link, self._prob_mask_, self._simplex_groups_, self.logit_eps_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and hasattr(self, "n_features_in_"):
            if len(input_features) != self.n_features_in_:
                raise ValueError(f"input_features should have length equal to number of features "
                                 f"({self.n_features_in_}), got {len(input_features)}")
            if hasattr(self, "feature_names_in_") and not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError("input_features is not equal to feature_names_in_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def __sklearn_tags__(self):
        return _input_tags(super().__sklearn_tags__(), self.model)

    # ---------------- helpers you'll actually use ----------------

    def estimate_cost(self, X) -> dict:
        """Rough cost of featurizing X with an empty cache, from the model."""
        if not self.questions:
            raise ValueError("questions must be set")
        model = self.model_ if hasattr(self, "model_") else resolve_model(self.model)
        states = states_only(X, state_columns=self.state_columns, state_fn=self.state_fn)
        return model.estimate_cost(states, self._requests(list(self.questions), model))

    # ---------------- internals ----------------

    def _validate_params(self) -> None:
        if not isinstance(self.questions, Mapping) or not self.questions:
            raise ValueError("questions must be a non-empty dict of question specs")
        if self.link not in _LINKS:
            raise ValueError(f"link must be one of {_LINKS}, got {self.link!r}")
        if self.link == "clr" and self.drop_redundant:
            raise ValueError("link='clr' needs the full simplex; set drop_redundant=False")
        if self.score_repr not in ("ev", "probs", "both"):
            raise ValueError(f"score_repr must be 'ev', 'probs' or 'both', got {self.score_repr!r}")
        if self.on_error not in ("raise", "nan"):
            raise ValueError(f"on_error must be 'raise' or 'nan', got {self.on_error!r}")
        if self.logit_eps is not None and not 0 < self.logit_eps < 0.5:
            raise ValueError(f"logit_eps must be in (0, 0.5), got {self.logit_eps!r}")
        if self.state_fn is not None and not callable(self.state_fn):
            raise TypeError("state_fn must be callable or None")
        if self.state_fn is not None:
            try:
                pickle.dumps(self.state_fn)
            except Exception:
                warnings.warn(f"state_fn {self.state_fn!r} can't be pickled (a lambda or a local function?), so "
                              "this estimator can't be saved with pickle or joblib. Define it at module level.",
                              UserWarning, stacklevel=3)

    def _requests(self, qnames: list[str], model: DecisionModel) -> list[dict]:
        size = model.capabilities().max_questions_per_call or len(qnames)
        return [{q: self.questions[q] for q in qnames[j : j + size]} for j in range(0, len(qnames), size)]

    def _answer_matrix(self, states: list) -> np.ndarray:
        """Answers for every (state, question), from the cache or the model,
        as identity-link columns."""
        qnames = list(self.questions)
        found = self._fetch(states, [self.questions] * len(states))
        out = np.full((len(states), len(self.feature_names_out_)), np.nan)
        versions = set()
        for i, row in enumerate(found):
            for q in qnames:
                if q in row:
                    ans, v = row[q]
                    versions.add(v)
                    out[i, self.feature_groups_[q]] = self._values(q, self.questions[q], ans)
        if len(versions) > 1:
            warnings.warn(f"Features in this matrix came from several model versions: {sorted(versions)}. "
                          "Pin the model version or use a fresh cache_path.", stacklevel=3)
        return out

    def _fetch(self, states: list, row_questions: Sequence[Mapping[str, dict]]) -> list[dict]:
        """Answer ``row_questions[i]`` about ``states[i]`` for every row, from the
        cache or the model. Rows may ask different questions.

        Package-internal: ``ChoiceEncoder`` uses it to re-ask exemplar rows.

        Returns one ``{question name: (answer, version)}`` per row; failed
        answers are missing (after applying ``on_error``).
        """
        ns = self.model_.cache_namespace()

        # 1) cache lookup, one key per (row, question); identical requests are asked once
        state_keys = [canon(s) for s in states]
        keys = [{q: answer_key(ns, question_key(spec), sk) for q, spec in qs.items()}
                for qs, sk in zip(row_questions, state_keys)]
        flat = [k for row in keys for k in row.values()]
        hits = self.cache_.get_many(list(dict.fromkeys(flat)))
        self.model_.usage["cache_hits"] += sum(k in hits for k in flat)

        # 2) group the misses by state, then chunk them into requests
        missing: dict[str, dict[str, dict]] = {}  # state key -> {answer key: spec}
        first_row: dict[str, int] = {}
        name_of: dict[str, str] = {}  # answer key -> question name used in the request
        for i, (qs, sk) in enumerate(zip(row_questions, state_keys)):
            for q, spec in qs.items():
                if keys[i][q] not in hits:
                    first_row.setdefault(sk, i)
                    missing.setdefault(sk, {})[keys[i][q]] = spec
                    name_of.setdefault(keys[i][q], q)
        size = self.model_.capabilities().max_questions_per_call
        tasks = []  # (row index, [answer keys])
        for sk, specs in missing.items():
            ks = list(specs)
            step = size or len(ks)
            tasks += [(first_row[sk], ks[j : j + step]) for j in range(0, len(ks), step)]

        # 3) fetch, cache every success, then deal with failures
        if tasks:
            if self.verbose:
                print(f"[{type(self).__name__}] {len(tasks)} requests for {len(missing)} unique rows")
            # questions keep their own names; two specs sharing a name in one request get a suffix
            names = []
            for _, ks in tasks:
                used, row = set(), []
                for k in ks:
                    name = name_of[k]
                    while name in used:
                        name += "~"
                    used.add(name)
                    row.append(name)
                names.append(row)
            items = [(states[i], {n: missing[state_keys[i]][k] for n, k in zip(ns_, ks)})
                     for (i, ks), ns_ in zip(tasks, names)]
            responses = self.model_.answer(items)
            failures, new_rows = [], []
            for (_, ks), ns_, res in zip(tasks, names, responses):
                if isinstance(res, BaseException):
                    failures.append(res)
                    continue
                for n, k in zip(ns_, ks):
                    hits[k] = (res.answers[n], res.version)
                    new_rows.append((k, *hits[k]))
            self.cache_.put_many(new_rows)
            if failures:
                fatal = [e for e in failures if getattr(e, "fatal", False)]
                if fatal:
                    raise fatal[0]
                if self.on_error == "raise":
                    raise failures[0]
                warnings.warn(f"{len(failures)} of {len(tasks)} model requests failed; "
                              "those answers are NaN.", stacklevel=4)

        found = [{q: hits[k] for q, k in row.items() if k in hits} for row in keys]
        self.model_.versions_seen.update(v for row in found for _, v in row.values())
        return found

    def _layout(self, qname: str, q: Mapping) -> list[tuple[str, bool]]:
        """[(column name, is_probability)] for one question."""
        t = q["type"]
        if t == "noul":
            return [(qname, True)]
        cols: list[tuple[str, bool]] = []
        if t == "choice":
            opts = list(q["criteria"])
            if self.drop_redundant:
                opts = opts[:-1]
            cols += [(f"{qname}{SEP}{o}", True) for o in opts]
        else:  # score
            L = len(q["criteria"])
            if self.score_repr in ("probs", "both"):
                lv = range(L - 1) if self.drop_redundant else range(L)
                cols += [(f"{qname}{SEP}L{i}", True) for i in lv]
            if self.score_repr in ("ev", "both"):
                cols.append((f"{qname}{SEP}ev", False))
        if self.include_confidence:
            cols.append((f"{qname}{SEP}confidence", False))
        return cols

    def _values(self, qname: str, q: Mapping, ans) -> list[float]:
        t = q["type"]
        if t == "noul":
            if not isinstance(ans, NoulAnswer):
                raise DecisionModelError(f"{qname}: expected a noul answer, got {ans!r}")
            return [ans.p]
        if not isinstance(ans, DistAnswer):
            raise DecisionModelError(f"{qname}: expected a distribution, got {ans!r}")
        p = list(ans.probs)
        vals: list[float] = []
        if t == "choice":
            vals += p[:-1] if self.drop_redundant else p
        else:
            if self.score_repr in ("probs", "both"):
                vals += p[:-1] if self.drop_redundant else p
            if self.score_repr in ("ev", "both"):
                vals.append(sum(i * pi for i, pi in enumerate(p)))
        if self.include_confidence:
            vals.append(math.nan if ans.confidence is None else ans.confidence)
        return vals
