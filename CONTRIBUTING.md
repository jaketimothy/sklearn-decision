# Contributing to sklearn-decision

Thanks for helping. Bug reports, question banks that worked (or didn't), new model backends, benchmark runs on new data, and documentation fixes are all welcome.

## Set up

```bash
git clone https://github.com/jaketimothy/sklearn-decision.git
cd sklearn-decision
python -m venv .venv
```

Activate the virtual environment, then install the package in editable mode with the extras you need:

```bash
pip install -e ".[dev,local,docs]"
```

- `dev`: pytest, pandas and ruff.
- `local`: torch and transformers, for `TransformersModel`.
- `docs`: Sphinx and the example gallery.

## Test, lint, build the docs

```bash
pytest -q
```

```bash
ruff check src tests benchmarks examples docs
```

```bash
python -m sphinx -b html docs docs/_build/html -W
```

- **`pytest -q` is fully offline.** It includes scikit-learn's `parametrize_with_checks` for every estimator, and runs the local backend against a tiny in-memory model.
- **`pytest -m hub`** downloads a small real model from the Hugging Face Hub.
- **`pytest -m live`** calls the Jev API. It needs `TYPESAFE_API_KEY`, which can be a 1Password `op://` reference.
- **The docs build runs every example** from the committed answer caches, so it needs no key, and warnings are errors.

CI runs the same checks on Python 3.10, 3.12 and 3.13, on the minimum supported dependency versions, and on the minimum torch and transformers.

## Pull requests

- **One topic per PR.** Open the PR against `main`.
  - If you stack PRs, retarget each one to `main` before merging it, or it lands on its base branch instead.
- **Lead the description with the bottom line:** what changes for users and why, then the details and how you tested it. The PR template has the sections.
- **Tests:**
  - new behaviour needs a test;
  - a bug fix needs a test that fails without the fix;
  - use `FakeModel` so the test runs offline.
- **Docs:** update the relevant guide page and docstrings (numpydoc style), and add a line to `CHANGELOG.md` under "Unreleased".
- **Estimators must stay scikit-learn compliant:**
  - `__init__` only stores parameters;
  - validation happens in `fit`;
  - fitted attributes end in `_`;
  - `transform`/`predict` don't change the estimator.

  `tests/test_sklearn_checks.py` enforces this. Adding an entry to its expected failures needs a stated reason.

## Adding a model backend

Subclass `DecisionModel` and implement three methods: `answer`, `capabilities` and `cache_namespace`. Register a name prefix with `register_model`. See [Choosing a model](https://jaketimothy.github.io/sklearn-decision/user_guide/models.html#your-own-backend).

- **`cache_namespace()`** must cover everything that changes the answers: model name, pinned revision, prompt template. Otherwise stale answers are served from the cache.
- **`capabilities()`** must be honest: option limits, native confidence, determinism, and probability rounding.
- **Tests must run offline.** `tests/tiny_lm.py` shows how the local backend is tested without downloads.

## Benchmarks and API spend

`benchmarks/` holds the evaluation harness, and `benchmarks/RESULTS.md` holds the results.

- **Cap spending.** Runs against a hosted model take `--max-cost-usd` (default $2).
- **Commit the answer cache** alongside new results, so anyone can re-fit them without a key.
- **Report failures as well as wins.** The behaviour checks exist to catch what doesn't work.
- **Keep the test split for the end.** Choose question banks, prompts and settings on training or validation data only.

## Never commit secrets

API keys go in `TYPESAFE_API_KEY`, ideally as a 1Password reference, never in code, notebooks or results. Answer caches contain only hashed keys and answers, not your rows' text, but check anything else before you commit it.

## Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). For security issues, see [SECURITY.md](SECURITY.md) rather than opening a public issue.
