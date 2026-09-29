# sklearn-decision

Making decision models like Jev useful in scikit-learn pipelines.

**Documentation:** https://jaketimothy.github.io/sklearn-decision/

A *decision model* answers typed questions about a piece of text or a record with probabilities:

- **noul**: is this statement true?
- **choice**: which of these options?
- **score**: where on this ordinal rubric?

This package turns those answers into scikit-learn estimators. You can use a single question as a zero-shot classifier or regressor, or a bank of questions as a feature encoder for any downstream model.

The model is a parameter, and you always choose it explicitly:

| `model=` | Runs | Notes |
|---|---|---|
| `"hf:<repo id>[@revision]"` | locally, open weights (`TransformersModel`) | Nothing leaves the machine. A frozen instruction-tuned LM read out through its answer-token logits. Deterministic. |
| `"jev-latest"` | TypeSafe's hosted API (`JevModel`) | Rows are sent to TypeSafe. Needs `TYPESAFE_API_KEY` (the key or a 1Password `op://` reference); has a `max_cost_usd` cap. |
| a `DecisionModel` instance | anywhere | Full control over its parameters, which are nested for grid search (`model__batch_size`). |

## Install

```bash
pip install "sklearn-decision[local] @ git+https://github.com/jaketimothy/sklearn-decision"
```

The package isn't on PyPI yet, so this installs from GitHub. The `local` extra adds `torch` and `transformers` for open-weights models; for the hosted API only, leave out `[local]`. Fitting never calls the model: weights load, and API calls happen, on the first `transform` or `predict`.

## Quickstart

### Question bank → features

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn_decision import QuestionFeaturizer, noul, choice, score

feat = QuestionFeaturizer({
    "asks_refund": noul("The customer asks for money back."),
    "product":     choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency":     score("How urgent is the request?", ["can wait", "this week", "today"]),
}, model="hf:google/gemma-4-12b-it")
clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)
```

### Decision primitives as zero-shot estimators

```python
from sklearn_decision import ChoiceClassifier, NoulClassifier, ScoreRegressor, calibrate_zero_shot

MODEL = "hf:google/gemma-4-12b-it"
router = ChoiceClassifier("Which team should handle this?",
                          {"billing": "charges, invoices, refunds", "api": "keys, errors, limits"}, model=MODEL)
router.fit(texts)                       # records the label set; no model calls
router.predict_proba(new_texts)

urgency = ScoreRegressor("How urgent is it?", ["can wait", "this week", "today"], level_values=[0, 3, 7],
                         model=MODEL)
```

### Calibrate before you trust the probabilities

Decision models are often overconfident. On 20 Newsgroups, Jev's zero-shot answers were 77% accurate but had a log-loss of 1.73: confidently wrong on the posts it missed. Calibrate on a few of your own labels before you threshold or combine them:

```python
refund = NoulClassifier("The customer asks for money back.", model=MODEL)
refund_cal = calibrate_zero_shot(refund, X_labelled, y_labelled)    # needs 2+ labels per class
refund_cal.predict_proba(new_texts)
```

`calibrate_zero_shot` is sigmoid (Platt) scaling on the frozen classifier: it never refits or re-queries the model. Once you have about 16 or more labels per class, a logistic head on the answers' log-odds does better: `make_pipeline(QuestionFeaturizer({...}, model=MODEL), LogisticRegression())`. The featurizer outputs log-odds by default.

| Jev log-loss on held-out posts (raw: 1.73) | 16 labels | 64 labels | 400 labels |
|---|---:|---:|---:|
| `calibrate_zero_shot` (sigmoid) | **0.71** | **0.64** | 0.62 |
| Logistic head | 0.95 | 0.65 | **0.60** |
| Isotonic | 3.43 | 1.37 | 0.87 |

Isotonic calibration made things worse with few labels, so avoid it until you have hundreds. The details are in [benchmarks/RESULTS.md](https://github.com/jaketimothy/sklearn-decision/blob/main/benchmarks/RESULTS.md#calibration).

### Choice codebooks → simplex embeddings

```python
from sklearn.linear_model import LogisticRegression
from sklearn_decision import ChoiceEncoder

enc = ChoiceEncoder("exemplars", "Which reference example is most similar to this one?",
                    anchor=("none", "No reference is a meaningful match"), random_state=0, model=MODEL)
clf = make_pipeline(enc, LogisticRegression()).fit(texts, y)
```

In `fit`, `ChoiceEncoder` samples landmark rows as options and splits them into questions at the model's option limit: 25 per question plus the anchor for local models (A–Z), 254 for Jev. It corrects exemplar self-matches, so it's safe inside cross-validation.

## Choosing and configuring the model

```python
from sklearn_decision import JevModel, TransformersModel, FakeModel

TransformersModel("google/gemma-4-12b-it", revision="<commit sha>", dtype="bfloat16",
                  batch_size=16, max_state_tokens=2048)
JevModel("jev-latest", timeout=60, max_questions_per_call=25, max_cost_usd=5.0)
FakeModel()                                                     # offline and deterministic, for tests
```

- **Local models** read answers from the next-token logits: Yes/No for noul, option letters for choice, level digits for score. Choice questions are asked under 4 rotations of their options by default (`n_option_permutations`), which cancels small models' position bias: for Qwen2.5-0.5B it raised zero-shot accuracy from 65% to 72%. Each row's text is encoded once and its key/value cache is shared by all of that row's questions. Pin `revision` to a commit for reproducible features; the resolved commit is recorded with every answer. See the [benchmarks](https://github.com/jaketimothy/sklearn-decision/blob/main/benchmarks/README.md) for what a small CPU model does and doesn't deliver.
- **Untrusted text:** a decision model reads the row as evidence, including claims the row makes about itself. On 20 Newsgroups, appending "this text is about space travel" changed Jev's topic answer on 20% of posts, and fencing the text or adding caveats to the question didn't help ([results](https://github.com/jaketimothy/sklearn-decision/blob/main/benchmarks/RESULTS.md#do-simple-defences-stop-the-injection-no)). Don't act automatically on answers about user-controlled text without another check.
- **Credentials:** `TYPESAFE_API_KEY` or `JevModel(api_key=...)` can be a 1Password secret reference such as `op://Personal/Typesafe API/password`. It's resolved with the 1Password CLI (`op read`) on the first request and kept only in process memory, never on the estimator. For scripts, `op run -- python ...` resolves it once for the whole run.
- **Jev versions:** the API lists only `jev-latest` and `jev-preview`. Each answer records the concrete version (`jev-1.13.0`), and a feature matrix that mixes versions raises a warning.
- **Caching:** every answer is cached by (model, prompt template, question, row). The default `cache_path=None` keeps answers in memory for the process, shared by clones and grid-search candidates, and writes nothing to disk. Pass `cache_path="answers.sqlite"` to persist them.
- **New backends** subclass `DecisionModel`, implementing `answer`, `capabilities` and `cache_namespace`. Register them with `register_model("prefix:", factory)`.

## Documentation

The docs cover a user guide, an API reference and runnable examples. They build from `docs/`:

```bash
pip install -e ".[docs]"
```

```bash
python -m sphinx -b html docs docs/_build/html
```

The examples read Jev's answers from the committed cache, so the build needs no API key and costs nothing. CI builds the site on every pull request and uploads it as an artifact. Merges to `main` publish it at **https://jaketimothy.github.io/sklearn-decision/**.

## Development

```bash
pytest -q
```

```bash
pytest -q -m hub
```

```bash
pytest -q -m live
```

- `pytest -q` runs the offline suite, including scikit-learn's `parametrize_with_checks`. It runs the local backend against a tiny in-memory model, so nothing is downloaded.
- `-m hub` checks a real small model from the Hugging Face Hub.
- `-m live` calls the Jev API and needs `TYPESAFE_API_KEY`.

See [Design](https://jaketimothy.github.io/sklearn-decision/design.html) for how the package is built, what the experiments changed, and the open questions.

See [CONTRIBUTING.md](https://github.com/jaketimothy/sklearn-decision/blob/main/CONTRIBUTING.md). Report security issues privately ([SECURITY.md](https://github.com/jaketimothy/sklearn-decision/blob/main/SECURITY.md)).
