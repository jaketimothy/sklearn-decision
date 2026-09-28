# Concepts

## Decision models

A **decision model** reads a *state* (a piece of text, or a JSON record) and answers typed *questions* about it with probabilities, not generated text. There are three question types:

| Type | Asks | Answer |
|---|---|---|
| `noul` | Is this statement true? | P(true) |
| `choice` | Which of these options? | a distribution over the options |
| `score` | Where on this ordinal rubric? | a distribution over the levels |

Questions are plain dicts, built with {func}`~sklearn_decision.noul`, {func}`~sklearn_decision.choice` and {func}`~sklearn_decision.score`:

```python
from sklearn_decision import noul, choice, score

noul("The customer asks for money back.")
choice("Which product is discussed?", ["app", "api", "billing"])
choice("Which team should handle it?", {"billing": "charges, invoices, refunds", "api": "keys, errors"})
score("How urgent is the request?", ["can wait", "this week", "today"])
```

The package works with any backend that answers these questions; see [Choosing a model](models.md). TypeSafe's Jev is one. A local open-weights language model is another: this package reads its answers from the next-token logits.

## Two ways to use the answers

**As a classifier or regressor.** One question is one estimator:

- {class}`~sklearn_decision.NoulClassifier`: binary, from one statement;
- {class}`~sklearn_decision.ChoiceClassifier`: the classes are the options;
- {class}`~sklearn_decision.ScoreRegressor`: the expected value over a rubric's levels.

These are zero-shot: `fit` only records the label set. They still work with everything in scikit-learn that takes a classifier, such as cross-validation, calibration, threshold tuning and stacking.

**As features.** A bank of questions is an encoder. {class}`~sklearn_decision.QuestionFeaturizer` turns each row into the answers to every question, and a downstream model learns how to weigh them. Every feature is a sentence you can read, and a question that doesn't help can be removed. {class}`~sklearn_decision.ChoiceEncoder` is a special case: one choice question whose options act as a codebook, giving each row a position on the simplex. See [Choice codebooks](choice_encoder.md).

## What `fit` does, and doesn't do

`fit` never calls the model. It validates the questions against the model's limits, lays out the output columns, and records the input features. The first `transform` or `predict` makes the calls.

Because answers depend only on the row and the question, not on the labels, featurization can't leak labels across cross-validation folds. Featurize once, then cross-validate the downstream model.

## States: what the model sees

| Input | State per row |
|---|---|
| a list, Series or 1-D array of strings or dicts | each element |
| a DataFrame | a `{column: value}` record, limited to `state_columns` if set |
| a 2-D array | a list of the row's values, or the value itself for one column |

`state_fn` transforms each state before it is sent, for example to trim or template text. Define it at module level, so the fitted estimator can be pickled.

**Send only what the questions need.** Unrelated material in the state lowers accuracy, which TypeSafe calls context rot.
