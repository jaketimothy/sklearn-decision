# sklearn-decision

Making decision models like Jev useful in scikit-learn pipelines.

A *decision model* answers typed questions about a piece of text or a record with probabilities:

- **noul**: is this statement true?
- **choice**: which of these options?
- **score**: where on this ordinal rubric?

This package turns those answers into scikit-learn estimators. You can use a single question as a zero-shot classifier or regressor, or a bank of questions as a feature encoder for any downstream model.

The model is a parameter, and you always choose it explicitly:

| `model=` | Runs | Notes |
|---|---|---|
| `"hf:<repo id>[@revision]"` | locally, open weights (`TransformersModel`) | Nothing leaves the machine. A frozen instruction-tuned LM read out through its answer-token logits. Deterministic. |
| `"jev-1.13"` | TypeSafe's hosted API (`JevModel`) | Rows are sent to TypeSafe. Needs `TYPESAFE_API_KEY`; has a `max_cost_usd` cap. |
| a `DecisionModel` instance | anywhere | Full control over its parameters, which are nested for grid search (`model__batch_size`). |

## Install

```bash
pip install -e ".[local]"
```

The `local` extra adds `torch` and `transformers` for open-weights models. For the hosted API only, `pip install -e .` is enough. Fitting never calls the model: weights load, and API calls happen, on the first `transform` or `predict`.

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn_decision import ChoiceClassifier, NoulClassifier, ScoreRegressor

MODEL = "hf:google/gemma-4-12b-it"
router = ChoiceClassifier("Which team should handle this?",
                          {"billing": "charges, invoices, refunds", "api": "keys, errors, limits"}, model=MODEL)
router.fit(texts)                       # records the label set; no model calls
router.predict_proba(new_texts)

refund = CalibratedClassifierCV(NoulClassifier("The customer asks for money back.", model=MODEL), cv=3)
refund.fit(texts, y)
urgency = ScoreRegressor("How urgent is it?", ["can wait", "this week", "today"], level_values=[0, 3, 7],
                         model=MODEL)
```

Raw probabilities from any model should be recalibrated on your own labels before you threshold them. `CalibratedClassifierCV` does that in one line.

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
JevModel("jev-1.13", timeout=60, max_questions_per_call=25, max_cost_usd=5.0)
FakeModel()                                                     # offline and deterministic, for tests
```

- **Local models** read answers from the next-token logits: Yes/No for noul, option letters for choice, level digits for score. Each row's text is encoded once and its key/value cache is shared by all of that row's questions. Pin `revision` to a commit for reproducible features; the resolved commit is recorded with every answer. See the [benchmarks](benchmarks/README.md) for what a small CPU model does and doesn't deliver.
- **Caching:** every answer is cached by (model, prompt template, question, row). The default `cache_path=None` keeps answers in memory for the process, shared by clones and grid-search candidates, and writes nothing to disk. Pass `cache_path="answers.sqlite"` to persist them.
- **New backends** subclass `DecisionModel`, implementing `answer`, `capabilities` and `cache_namespace`. Register them with `register_model("prefix:", factory)`.

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

See [docs/design.md](docs/design.md) for the design rationale, the rules for writing questions, and the experiment plan.
