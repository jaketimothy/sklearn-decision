# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/) (while at 0.x, minor versions may
change behaviour).

## [Unreleased]

## [0.1.0] - unreleased

First release.

### Added

- Estimators that turn a decision model's typed answers into scikit-learn
  primitives: `QuestionFeaturizer` (a bank of noul/choice/score questions as
  features), `ChoiceEncoder` (a codebook of options as simplex embeddings),
  and the zero-shot `NoulClassifier`, `ChoiceClassifier` and `ScoreRegressor`.
- The `DecisionModel` protocol, with a name registry, and three backends:
  - `JevModel` for TypeSafe's hosted Jev (`model="jev-latest"`), with
    retries, a process-wide spending cap (`max_cost_usd`) that holds across
    grid-search and cross-validation clones, and 1Password `op://` secret
    references for the API key.
  - `TransformersModel` for local open-weights models
    (`model="hf:<repo id>[@revision]"`), read out through next-token logits,
    with a shared-prefix key/value cache. Logits are computed only at the
    answer position, and `dtype="auto"` means float32 on the CPU.
  - `FakeModel`, deterministic and offline, for tests.
- An answer cache keyed by (model, question, row): in memory by default,
  SQLite with `cache_path`, which several processes can share. Answers are
  fetched and cached in chunks, so an interrupted run keeps what it fetched,
  and `verbose=True` prints progress, time remaining and spend.
- `cache_path`, `state_columns`, `state_fn`, `on_error` and `verbose` are
  parameters of every estimator.
- `QuestionFeaturizer` outputs log-odds by default (`link="logit"`): the
  right scale for linear models, and no different for trees.
- `NoulClassifier` requires `positive_label` unless the labels are 0/1, -1/1
  or booleans, so string labels can't silently invert it.
- `calibrate_zero_shot()`: sigmoid calibration of a frozen zero-shot
  classifier from as few as two labels per class, the best recipe with few
  labels in the calibration benchmark.
- `TransformersModel(n_option_permutations=4)`: choice questions are asked
  under several rotations of their options and averaged, cancelling
  position bias in small local models.
- `usage()` on every estimator: model calls, cache hits, tokens, cost and
  the model versions behind its answers.
- `ChoiceEncoder` re-asks exemplar rows without their own option
  (`self_match="reask"`) and keeps codebook blocks as separate simplexes.
  Renormalizing, or stitching blocks into one distribution, would assume
  independence of irrelevant alternatives, which neither benchmarked model
  satisfies, so neither is offered.
- Documentation (`docs/`): user guide, API reference and a gallery of examples that run offline from the committed Jev answer cache.
- A benchmark harness (`benchmarks/`) with behaviour checks and learning
  curves, and results for Jev and Qwen2.5-0.5B on 20 Newsgroups.

[Unreleased]: https://github.com/jaketimothy/sklearn-decision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jaketimothy/sklearn-decision/releases/tag/v0.1.0
