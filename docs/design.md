# Design

**Thesis.** Use a decision model as an encoder. Unstructured rows go in, and the answers to a bank of typed questions come out as a numeric matrix of probabilities. That turns an unstructured-data problem into an ordinary scikit-learn problem. The question bank plays the role of the encoder's weights, and every weight is a sentence you can read.

This page explains how the package is built and why. It started as the design notes for a single-file Jev prototype; what the experiments found is summarized [below](#what-the-experiments-changed), with the full numbers in [Benchmarks](benchmarks.md).

---

## Components

### `QuestionFeaturizer`: the general encoder

```python
from sklearn_decision import QuestionFeaturizer, noul, choice, score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

feat = QuestionFeaturizer(questions={
    "asks_refund": noul("The customer asks for money back."),
    "product":     choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency":     score("How urgent is the request?", ["can wait", "this week", "today"]),
}, model="jev-latest")
print(feat.estimate_cost(texts))
clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)
```

Each question becomes one or more columns, and `feature_groups_` maps each question to its columns so you can prune by question. The `link` parameter sets the scale: raw probabilities, log-odds for linear models, or the centred log-ratio when a question is used as an embedding. [Writing questions](user_guide/questions.md) has the column layout and the rules for wording questions.

### `ChoiceEncoder`: the option set is the codebook

```python
from sklearn_decision import ChoiceEncoder

enc = ChoiceEncoder("exemplars", {                   # landmark rows as options, sampled in fit
    "content": "Which reference example is most similar in subject matter?",
    "intent":  "Which reference example has the most similar goal or request?",
}, anchor=("none", "None of the references is a meaningful match"), random_state=0, model="jev-latest")
clf = make_pipeline(enc, LogisticRegression()).fit(train_texts, y_train)
```

- **Views.** Each view is a different question asked over the same codebook.
- **Codebook types.** A concept codebook is a taxonomy of labels (`ChoiceEncoder(["billing", "outage", ...])`). An exemplar codebook (`"exemplars"`) uses real rows as options, sampled in `fit` and stratified by `y`. It acts like a similarity kernel to landmarks, in the Nyström style.
- **Self-matches.** An exemplar trivially matches itself. With `self_match="reask"`, the default, a row that is an exemplar is asked again without its own option. This happens in `fit_transform` and for identical rows later, so exemplars can stay in the training set and the encoder is safe inside `Pipeline` and cross-validation. `"renormalize"` is cheaper but assumes independence of irrelevant alternatives (IIA), which neither benchmarked model satisfies. `"keep"` is for ablations.
- **Option limits come from the model.** `model_.capabilities().max_choice_options` is 255 for Jev and 26 (A–Z) for local models. Larger codebooks are split into blocks automatically.
- **`stitch=True`**, or the `stitch_blocks` helper, rebuilds one global distribution from the blocks through the shared anchor option. It's exact only under IIA, so treat it as an approximation.

### Zero-shot estimators

| Estimator | Question type | API |
|---|---|---|
| `ChoiceClassifier(instructions, criteria)` | choice | `predict`, `predict_proba`, `predict_confidence` |
| `NoulClassifier(instructions, positive_label)` | noul | `predict`, `predict_proba` |
| `ScoreRegressor(instructions, levels, level_values)` | score | `predict`, `predict_levels_proba`, `predict_confidence` |

`fit` only records the label set, so these are honest zero-shot models. They work with:
- `CalibratedClassifierCV` and {func}`~sklearn_decision.calibrate_zero_shot`;
- `TunedThresholdClassifierCV`;
- `StackingClassifier`;
- `GridSearchCV` over the wording;
- selective prediction through `predict_confidence`, which is present only when the model reports a confidence.

Cache and state settings come from an optional `featurizer=QuestionFeaturizer(...)` template, for example `featurizer__cache_path`.

Calibration matters. Jev's zero-shot answers on 20 Newsgroups were 77% accurate but had a log-loss of 1.73. Sigmoid scaling from 16 labels brought that to 0.71; see [Calibration](user_guide/calibration.md).

---

## Key design decisions

- **The model is a parameter, and there is no default.** Every estimator takes `model=`: a registered name (`"jev-latest"`, `"hf:<repo id>"`) or a `DecisionModel` instance, whose parameters are nested for grid search (`model__timeout`). Nothing is sent anywhere until the user chooses.
- **One protocol for every backend.** A backend:
  - normalizes its answers to one schema;
  - declares its limits and properties through `capabilities()`: option and level caps, questions per call, native confidence, determinism, and probability resolution;
  - supplies the cache namespace.

  A new backend is a `DecisionModel` subclass plus `register_model(prefix, factory)`, with no estimator changes.
- **Stateless fit.** Featurization makes no model calls in `fit` and is row-wise, so it can't leak labels across folds. Featurize once, then cross-validate only the downstream model.
- **Runtime state lives on `model_`, not the estimator.** Connections, clients, loaded weights and usage counters hang off the resolved model, so `transform` and `predict` leave the estimator unchanged, as scikit-learn requires. `usage()` reports calls, cache hits, tokens and the model versions seen.
- **Cache per (model, prompt template, question, row).** By default the cache is in memory for the process, shared by clones and grid-search candidates. `cache_path` persists it to SQLite. Adding a question fetches one column; pruning, `clone`, grid search and re-runs are free. The cache stores hashed keys and answers, never row text.
- **Track the version.** The Jev API offers only unpinned names (`jev-latest`, `jev-preview`), but every response reports the concrete version (`jev-1.13.0`). The version is recorded with every answer, and `transform` warns when one matrix mixes versions. An upgraded model is a new encoder: start a fresh cache and refit. Local models pin a Hub commit instead.
- **Local models are read out, not generated from.** `TransformersModel` reads each answer from a frozen model's next-token logits: Yes/No, option letters, or level digits. Answers are deterministic and graded. Each row's text is encoded once, and its key/value cache is shared by all of that row's questions.
- **Chunking.** `JevModel(max_questions_per_call=...)` controls request size, and each request re-sends the row. TypeSafe's limits are 64k tokens per request, and the row plus the longest single question must fit in 32k.
- **Failures.** Transient errors retry with backoff and honour `Retry-After`. Successful answers are cached before an error is raised. `on_error="nan"` works with HistGradientBoosting. Bad credentials always raise.
- **Spending is capped.** `JevModel(max_cost_usd=...)` refuses to start a batch that would exceed the cap; the docs examples use `max_cost_usd=0`, so a cache miss fails instead of spending.
- **Mixed data.** Use `ColumnTransformer`: `QuestionFeaturizer` on the text column, and numeric columns passed through untouched.

---

## What the experiments changed

The behaviour checks and learning curves in [Benchmarks](benchmarks.md) ran on Jev and on Qwen2.5-0.5B. Several results changed the package:

- **Jev rounds probabilities to 0.01.** Models now report `probability_resolution`, and the featurizer clips log-odds at half of it (`logit_eps_`), so a reported 0.00 doesn't become an outlier.
- **Jev's answers saturate at 0 and 1**, and its raw probabilities are overconfident. This is why log-odds features and {func}`~sklearn_decision.calibrate_zero_shot` exist.
- **Neither model satisfies IIA.** `ChoiceEncoder` therefore re-asks self-matches by default instead of renormalizing, and stitching is documented as approximate.
- **Answers don't depend on neighbouring questions** (Jev's coupling check passed). Caching per question is sound, and requests can carry the whole bank.
- **Small local models have a strong position bias.** `TransformersModel` averages choice answers over 4 rotations of the options by default, which raised Qwen's zero-shot accuracy from 65% to 72%.
- **Text in the row can steer the answer.** Appending "this text is about space travel" changed Jev's topic answer on 20% of posts, and no simple fence or caveat helped. The package ships no fencing helper, because it would look like protection without providing any; see [Untrusted text](user_guide/security.md).
- **A cache-key bug:** reordered choice options shared a key. It was found by the order check and fixed for every backend.

---

## Open questions

- **A task without nameable classes**, such as Banking77 intents. On 20 Newsgroups one direct topic question was the strongest arm; the bank-as-encoder result matters most where no such question exists.
- **Larger open models on a GPU**, such as Gemma 4 12B or 31B, which independent comparisons report as close to Jev. decider-4b v2 speaks TypeSafe's wire format, so it needs only a small generalization of `JevModel`.
- **Codebook size and views.** Does accuracy keep rising as the exemplar codebook grows toward 255 options, and do several views beat one at equal cost?
- **The question-bank loop.** Generate 100–300 candidate questions with an LLM, featurize, prune with `grouped_permutation_importance`, and probe the residuals with new questions, choosing the bank on validation data only.
- **Throughput.** Jev's maximum questions per call isn't documented; the default of 50 hasn't been tuned. Local models batch within a row; batching across rows, or a vLLM backend, would help on a GPU.

---

## Related work and positioning

- **scikit-learn wrappers for LLMs.** `scikit-llm` (`ZeroShotGPTClassifier`) and `stormtrooper` wrap chat models and NLI models as scikit-learn classifiers. The zero-shot estimators here are the decision-model counterpart, with real probabilities, a confidence score, caching and version tracking.
- **Jev in scikit-learn workflows.** glemaitre/jev-classification-topic is a benchmark harness, zero-shot only, not a reusable estimator. No Jev featurizer, question-bank encoder or codebook encoder was found as of 2026-09-25.
- **Question answers as features.** Research on interpretable embeddings built from LLM yes/no answers (QA-Emb), prompt-based decision trees (Tree-Prompt) and concept-bottleneck models is the direct lineage. Its practical blocker was the cost of one LLM call per question. Jev's pricing ($0.042 per million input tokens) and local logit readout both remove it.
