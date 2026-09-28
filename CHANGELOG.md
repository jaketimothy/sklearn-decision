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
    retries, a spending cap (`max_cost_usd`) and 1Password `op://` secret
    references for the API key.
  - `TransformersModel` for local open-weights models
    (`model="hf:<repo id>[@revision]"`), read out through next-token logits,
    with a shared-prefix key/value cache.
  - `FakeModel`, deterministic and offline, for tests.
- An answer cache keyed by (model, question, row): in memory by default,
  SQLite with `cache_path`.
- `calibrate_zero_shot()`: sigmoid calibration of a frozen zero-shot
  classifier from as few as two labels per class, the best recipe with few
  labels in the calibration benchmark.
- `TransformersModel(n_option_permutations=4)`: choice questions are asked
  under several rotations of their options and averaged, cancelling
  position bias in small local models.
- `usage()` on every estimator: model calls, cache hits, tokens and the
  model versions behind its answers.
- `ChoiceEncoder(self_match="reask")`, the default: exemplar rows are
  re-asked without their own option instead of renormalizing, which
  assumed independence of irrelevant alternatives.
- A benchmark harness (`benchmarks/`) with behaviour checks and learning
  curves, and results for Jev and Qwen2.5-0.5B on 20 Newsgroups.

[Unreleased]: https://github.com/jaketimothy/sklearn-decision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jaketimothy/sklearn-decision/releases/tag/v0.1.0
