# Caching and cost

Every answer is cached by **(model, question, row)**. Re-running a pipeline, cross-validating, grid-searching, cloning, or adding a question to a bank only pays for answers that aren't cached yet.

## Where answers live

- **`cache_path=None`** (default): a process-wide in-memory cache, shared by clones and grid-search candidates. Nothing is written to disk. {func}`~sklearn_decision.clear_memory_cache` empties it.
- **`cache_path="answers.sqlite"`**: a SQLite file that survives restarts. Several estimators, several models, and several processes can share one file.

Every estimator takes `cache_path`, as well as `state_columns`, `state_fn`, `on_error` and `verbose`.

A cache key covers everything that changes an answer:
- the model's `cache_namespace()`: its name, and for local models the revision, dtype and prompt templates;
- the question spec, including the order of a choice question's options;
- the row's state.

Batching settings aren't part of the key. We checked that Jev's answers don't depend on the other questions sent in the same request: the co-question coupling check in [Benchmarks](../benchmarks.md) passed.

## Versions

Each cached answer records the concrete model version that produced it. If one feature matrix mixes versions, for example after TypeSafe upgrades `jev-latest`, `transform` warns. An upgraded model is a new encoder: start a fresh cache and refit the downstream model.

## Knowing what you spend

```python
feat.estimate_cost(X)     # before: rows, requests, estimated tokens and USD (Jev)
feat.usage()              # after: calls, answers fetched, cache hits, tokens, model versions
```

For Jev, `JevModel(max_cost_usd=...)` caps spending across the whole process, including every clone that grid search and cross-validation make. See [Choosing a model](models.md).

Identical rows are asked only once per `transform`. Each request re-sends the row, so fewer, larger requests are cheaper; `JevModel(max_questions_per_call=...)` sets the request size.

## Long runs

Answers are fetched in chunks, and each chunk is cached as soon as it lands. An interrupted run (Ctrl-C, a crash, a closed laptop) keeps everything fetched so far, and rerunning it asks only for the rest. `verbose=True` prints a line per chunk: requests done, rate, time remaining, and dollars spent for Jev.

The in-memory cache belongs to one process. Parallel workers (`n_jobs > 1` in `cross_val_score` or `GridSearchCV`) would each ask the model again, and each load a local model's weights. Either set `cache_path`, or featurize once and cross-validate only the downstream model:

```python
Z = feat.fit_transform(X)                                     # one pass over the model
cross_val_score(LogisticRegression(), Z, y, n_jobs=-1)        # parallel, no model calls
```

This is safe because the featurizer learns nothing from the rows or labels in `fit`.

## Failures

`JevModel` retries transient errors (timeouts, 429s, 5xx) with backoff. Successful answers are cached before any error is raised, so a rerun only asks for what failed.

- **`on_error="raise"`** (default) re-raises the first failure.
- **`on_error="nan"`** warns and leaves the failed answers as NaN. `HistGradientBoosting` models handle NaN natively.

Fatal errors, such as bad credentials or a spending cap that would be exceeded, always raise.
