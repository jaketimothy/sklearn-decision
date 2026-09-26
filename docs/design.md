# sklearn-decision: design and experiment plan

**Thesis.** Use TypeSafe's Jev as an encoder. Unstructured rows go in, and the answers to a bank of typed questions come out as a numeric matrix of calibrated probabilities. That turns an unstructured-data problem into an ordinary scikit-learn problem. The question bank plays the role of the encoder's weights, and every weight is a sentence you can read.

**Status.** `sklearn_decision` implements everything below. The estimators are model-agnostic: the decision model is a `model` parameter backed by a small `DecisionModel` protocol, and It ships two real backends: `TransformersModel`, local open weights read out through their answer-token logits (`model="hf:<repo id>"`), and `JevModel`, TypeSafe's hosted API (`model="jev-1.13"`). An offline `FakeModel` is for tests. There is no default model: nothing is sent anywhere until the user chooses. `JevModel` has been tested against a mock of the documented `/v1/systemone` wire format, not yet against the live API. Target model: jev-1.13, the only published version as of Sept 2026.

This document started as the design notes for the single-file prototype `jev_featurizer.py`; names have been updated to the package (`JevFeaturizer` → `QuestionFeaturizer`, `Jev*Classifier` → `*Classifier`, API settings → `JevModel`).

---

## 1. Components

### 1.1 `QuestionFeaturizer`: the general encoder

```python
from sklearn_decision import QuestionFeaturizer, noul, choice, score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

feat = QuestionFeaturizer(questions={
    "asks_refund": noul("The customer asks for money back."),
    "product":     choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency":     score("How urgent is the request?", ["can wait", "this week", "today"]),
})
print(feat.estimate_cost(texts))
clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)
```

| Question | Columns |
|---|---|
| `noul` | `name` = P(true) |
| `choice` (K options) | `name__<option>` × K (`drop_redundant` drops one) |
| `score` (L levels) | `name__ev` (expected level) and/or `name__L0..` (`score_repr`) |
| choice/score | `name__confidence` with `include_confidence=True` |

The `link` parameter sets the output scale:

- `identity` keeps raw probabilities; tree models don't care about scale.
- `logit` gives log-odds, the right scale for linear models.
- `clr` applies the centred log-ratio transform per choice/score question, the natural geometry when a question is used as an embedding.

`feature_groups_` maps each question to its columns, so you can prune by question.

### 1.2 Choice-only encoder: the option set is the codebook

```python
from sklearn_decision import ChoiceEncoder

enc = ChoiceEncoder("exemplars", {                   # landmark rows as options, sampled in fit
    "content": "Which reference example is most similar in subject matter?",
    "intent":  "Which reference example has the most similar goal or request?",
}, anchor=("none", "None of the references is a meaningful match"), link="clr", random_state=0)
clf = make_pipeline(enc, LogisticRegression()).fit(train_texts, y_train)
```

- **Views.** Each view is a different question statement over the same codebook.
- **Codebook types.** A concept codebook is a taxonomy of labels (`ChoiceEncoder(["billing", "outage", ...])`). An exemplar codebook (`"exemplars"`) uses real rows as options, sampled in `fit` and stratified by `y`, which acts like a similarity kernel to landmarks (Nyström-style).
- **Self-matches.** An exemplar trivially matches itself. With `self_match="renormalize"` (the default), a row that is an exemplar has its own option zeroed and the rest of that simplex renormalized, in `fit_transform` and for identical rows later, so exemplars can stay in the training set and the encoder works inside `Pipeline` and CV. This assumes IIA (Phase 1, test 4); `self_match="keep"` is there for ablations.
- **Option limits come from the model.** `model_.capabilities().max_choice_options` is 255 for Jev today. Larger codebooks are split into blocks automatically, and a backend without a limit keeps one question.
- **`stitch=True`** (or the `stitch_blocks` helper) rebuilds one global distribution from the blocks through the shared anchor option. It is only valid if the model satisfies independence of irrelevant alternatives (Phase 1 checks this).
- `choice_bank` and `exemplar_options` remain public helpers for building a bank by hand.

### 1.3 Pass-through estimators: Jev directly, with sklearn's API

| Estimator | Jev type | API |
|---|---|---|
| `ChoiceClassifier(instructions, criteria)` | choice | `predict`, `predict_proba`, `predict_confidence` |
| `NoulClassifier(instructions, positive_label)` | noul | `predict`, `predict_proba` |
| `ScoreRegressor(instructions, levels, level_values)` | score | `predict`, `predict_levels_proba`, `predict_confidence` |

`fit` only records the label set, so these are honest zero-shot models. They work with `CalibratedClassifierCV`, `TunedThresholdClassifierCV`, `StackingClassifier`, `GridSearchCV` over wording, and selective prediction via `predict_confidence` (present only when the model reports a confidence). The decision model is the `model` parameter (`"jev-1.13"` or `JevModel(...)`, nested as `model__timeout`); cache and state settings come from an optional `featurizer=QuestionFeaturizer(...)` template (`featurizer__cache_path`).

Recalibration is likely to matter. One independent audit on civil_comments found that at about 75% stated confidence only about 10% of comments were human-flagged, and a two-parameter recalibration fixed it. Calibrate on your own domain by default.

---

## 2. Key design decisions

- **Stateless fit.** Featurization makes no API calls in `fit` and is row-wise, so it can't leak labels across folds. Featurize once, then cross-validate only the downstream model.
- **Cache per (model, question, state)** in SQLite. Adding a question fetches one column; pruning, `clone`, grid search and re-runs are free.
- **Pin the version.** The cache records the concrete model version behind every answer, and `transform` warns when one matrix mixes versions. An upgraded model is a new encoder: start a fresh cache and refit.
- **Model-agnostic estimators.** Every estimator takes `model=`: a registered name (`"jev-1.13"`) or a `DecisionModel` instance. A backend normalizes answers to one schema, declares its limits through `capabilities()`, and supplies the cache namespace, so a new backend (open-weights local models, openjev) is a new `DecisionModel` subclass plus `register_model(prefix, factory)`, with no estimator changes.
- **Chunking.** `JevModel(max_questions_per_call=...)` controls request size, and each request re-sends the state. The limits are 64k tokens per request, and the state plus the longest single question must fit in 32k. The maximum number of questions per call isn't documented, so we measure it (Phase 1).
- **Failures.** Transient errors retry with backoff and honour `Retry-After`. Successful answers are cached before an error is raised. `on_error="nan"` works with HistGradientBoosting.
- **Mixed data.** Use `ColumnTransformer`: `QuestionFeaturizer` on the text column, and numeric columns passed through untouched.

---

## 3. Rules for writing questions

These come from TypeSafe's jev-1.13 jaggedness page. Whether the question bank is written by hand or generated by an LLM, it has to follow them.

1. **One judgment per question.** Hiding several judgments inside one question is listed as an anti-pattern. Split compound questions.
2. **Say exactly what you mean.** The model answers the question you wrote, not the one you meant, and reads scoping words, negations and implied conditions at face value. State the exact condition and put boundary cases in the criteria. No double negatives.
3. **Never ask Jev for numbers.** Counting, numeric representations and arithmetic through a Score are all weak, and it reads dates as text rather than ordered quantities. Compute numeric and date features in code and pass them through the ColumnTransformer. At most, give Jev a named bucket.
4. **Keep the state minimal.** Accuracy falls as unrelated material is added to the state; TypeSafe calls this context rot. Use `state_columns`/`state_fn` to send only what the bank needs.
5. **Treat the state as hostile.** The model reads the state as data and has no defence against instructions hidden in it. User-generated text needs a red-team pass (Phase 1).

---

## 4. Experiment plan

### Phase 0: setup

- Read the jaggedness page and the community robustness index (awesome-jev-robustness, about 109 entries, almost all on jev-1.13.0). Don't re-run a test someone has already published; cite it and move on.
- Datasets, chosen so results line up with published numbers:
  - **20 Newsgroups**, the same split and header/footer/quote removal as glemaitre/jev-classification-topic. That benchmark provides the TF-IDF, MiniLM+LR and HGB-on-LSA baselines plus Jev zero-shot.
  - **Banking77**, where independent evals already compare Jev with supervised encoders.
  - **One of your own domain datasets.** This is the one that matters for the ventures.
- Pin `jev-1.13` and use one `cache_path` per model version.

### Phase 1: model-behaviour checks (cheap; run before anything else)

| # | Test | Why it matters | Pass condition |
|---|---|---|---|
| 1 | **Precision of small probabilities**: are they nonzero or rounded to 0.0? | Decides whether one choice view is an embedding or just a one-hot classifier | Most rows have meaningful mass beyond the top option |
| 2 | **Noise floor**: send identical requests ×15 on 200 rows | Published tests put Noul spread at about 0.01, with borderline answers crossing 0.5 | Spread is far smaller than the feature signal |
| 3 | **Order bias**: shuffle options ×4 | A competitor claims primacy bias in Jev | Probabilities are stable under shuffling |
| 4 | **IIA**: full option set vs random subsets; is p_i/p_j preserved? | Gates `stitch_blocks` and swapping exemplars | Ratios preserved within the noise floor |
| 5 | **Co-question coupling**: 50 questions alone vs inside the full bank | The per-question cache assumes answers don't depend on neighbouring questions | Differences are within the noise floor; otherwise send the whole bank in one request |
| 6 | **Max questions per call** | Undocumented; sets chunking and cost | Largest count that fits the token budget without errors or accuracy loss |
| 7 | **Injection**: 20 rows with embedded instructions | Hostile state | Features for those rows are flagged or unchanged |

### Phase 2: choice-only encoder

1. **Accuracy vs K**, with K ∈ {16, 32, 64, 128, 255}, for concept and exemplar codebooks, using `link="clr"` into logistic regression. The slope as K approaches 255 predicts what a higher option limit would buy.
2. **Views vs K at equal token cost**, with M ∈ {1, 2, 4, 8}.
3. **Blocks + stitching vs a single block at K=255**, only if tests 1.4 and 1.5 passed.

### Phase 3: question-bank loop

Generate 100–300 candidate questions with an LLM, following section 3's rules, then featurize, fit, prune with `grouped_permutation_importance`, and probe the residuals with new questions. Iterate on the validation split only. The bank is a fitted hyperparameter, so the test split stays untouched until the end.

### Phase 4: comparisons, as learning curves over the number of labels

Run every method at n_labels ∈ {50, 200, 1k, full}. This matters because independent benchmarks show trained local classifiers beating Jev on raw accuracy when there are thousands of labels, yet Jev winning on Yelp (67.2% vs 51.9%). The hypothesis is that Jev features plus a supervised head keep the zero-shot prior when labels are few and catch up to supervised models when labels are many.

| Arm | Description |
|---|---|
| A | TF-IDF + LinearSVC |
| B | Sentence embeddings + LR |
| C | Jev zero-shot (`ChoiceClassifier`), raw and calibrated |
| D | Choice-only encoder + LR (Phase 2 winner) |
| E | Question-bank featurizer + HGB/LR (Phase 3 winner) |
| F | B + E concatenated |
| G | Stacking C + B |

Report accuracy, macro-F1, log-loss and ECE, plus cost per 1k rows and latency.

### Decision criteria

- **Featurizer is worth shipping** if D or E beats B at ≤1k labels, or matches B at full labels while staying interpretable.
- **Choice-codebook line is worth continuing** if accuracy is still rising at K=255, or views give gains at equal cost.
- **Pass-through estimators ship regardless.** They are the zero-shot baseline and useful on their own.

---

## 5. Related work and positioning

- **sklearn wrappers for LLMs.** `scikit-llm` (`ZeroShotGPTClassifier`) and `stormtrooper` wrap chat models and NLI models as sklearn classifiers. The pass-through estimators here are the Jev counterpart, with real probabilities, a confidence score, caching and version pinning.
- **Jev in sklearn workflows.** glemaitre/jev-classification-topic is a benchmark harness, zero-shot only, not a reusable estimator. No Jev featurizer, question-bank encoder or codebook encoder was found as of 2026-09-25.
- **Question answers as features.** Research on interpretable embeddings built from LLM yes/no answers (QA-Emb), prompt-based decision trees (Tree-Prompt) and concept-bottleneck models is the direct lineage. Its practical blocker was the cost of one LLM call per question. Jev's pricing ($0.042 per million input tokens, free output) removes it.
- **Open Jev-like models.** `TransformersModel` implements the "frozen model + logit readout" recipe from the open-weights research (e.g. Gemma 4 12B/31B, Cygnet-style), which suits the featurizer: answers are deterministic and graded rather than decision-sharpened, and provenance is clean. decider-4b v2 speaks TypeSafe's wire format, so it needs only a small generalization of `JevModel` (a configurable model name and base URL); that is the natural next backend.

---

## 6. Next steps

1. Run Phase 1 against the live API and fix any wire-format mismatches.
2. ~~Package it: `pyproject`, tests using the mock transport, CI.~~ Done as `sklearn-decision`. Publish once Phase 1 passes, since the gap in the ecosystem is real but the window is short.
3. Run Phases 2–4 and publish the results alongside the package.
