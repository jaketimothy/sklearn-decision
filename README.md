# sklearn-decision

Making decision models like Jev useful in scikit-learn pipelines.

A *decision model* answers typed questions about a piece of text or a record with calibrated probabilities:

- **noul**: is this statement true?
- **choice**: which of these options?
- **score**: where on this ordinal rubric?

This package turns those answers into scikit-learn estimators. You can use a single question as a zero-shot classifier or regressor, or a bank of questions as a feature encoder for any downstream model.

The model is a parameter. v1 ships TypeSafe's Jev (`model="jev-1.13"`), and other backends plug in through the `DecisionModel` protocol.

## Install

```bash
pip install -e ".[dev]"
```

Set `TYPESAFE_API_KEY` to call Jev. Fitting never calls the model, so you only need the key for `transform` and `predict`.

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
}, model="jev-1.13")
print(feat.estimate_cost(texts))
clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)
```

### Decision primitives as zero-shot estimators

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn_decision import ChoiceClassifier, NoulClassifier, ScoreRegressor

router = ChoiceClassifier("Which team should handle this?",
                          {"billing": "charges, invoices, refunds", "api": "keys, errors, limits"})
router.fit(texts)                       # records the label set; no model calls
router.predict_proba(new_texts)

refund = CalibratedClassifierCV(NoulClassifier("The customer asks for money back."), cv=3).fit(texts, y)
urgency = ScoreRegressor("How urgent is it?", ["can wait", "this week", "today"], level_values=[0, 3, 7])
```

### Choice codebooks → simplex embeddings

```python
from sklearn.linear_model import LogisticRegression
from sklearn_decision import ChoiceEncoder

enc = ChoiceEncoder("exemplars", "Which reference example is most similar to this one?",
                    anchor=("none", "No reference is a meaningful match"), random_state=0)
clf = make_pipeline(enc, LogisticRegression()).fit(texts, y)
```

In `fit`, `ChoiceEncoder` samples landmark rows as options, blocking them to the model's option limit (255 for Jev). It corrects exemplar self-matches, so it's safe inside cross-validation.

## Choosing and configuring the model

```python
from sklearn_decision import JevModel, FakeModel

QuestionFeaturizer(bank, model="jev-1.13")                             # registered name
QuestionFeaturizer(bank, model=JevModel("jev-1.13", timeout=60,        # full control; nested
                                        max_questions_per_call=25))    # params: model__timeout
QuestionFeaturizer(bank, model=FakeModel())                            # offline, deterministic
```

Every answer is cached in SQLite by (model, question, state), at `cache_path="decision_cache.sqlite"` by default. Re-runs, cross-validation, grid search and adding questions only pay for answers that aren't cached yet.

A new backend, such as an open-weights local model, subclasses `DecisionModel`, implementing `answer`, `capabilities` and `cache_namespace`. Register it with `register_model("prefix-", factory)`.

## Development

```bash
pytest -q
```

```bash
pytest -q -m live
```

`pytest -q` runs every test that doesn't call the live API, including scikit-learn's `parametrize_with_checks`. `pytest -m live` runs the live Jev smoke test and needs `TYPESAFE_API_KEY`.

See [docs/design.md](docs/design.md) for the design rationale, the rules for writing questions, and the experiment plan.
