# Writing questions

The question bank is the encoder: each question's wording is effectively its weights. Treat the bank like a hyperparameter. Write it carefully, measure it, and prune it.

## Rules

These come from TypeSafe's guidance for Jev, and they held up in our benchmarks with local models too.

1. **One judgment per question.** Split compound questions. "Is this a dimensional or cosmetic defect?" becomes two questions.
2. **Say exactly what you mean.** The model answers the question you wrote, and reads scope words, negations and implied conditions literally. State the exact condition, put boundary cases in the option descriptions, and avoid double negatives.
3. **Never ask for numbers.** Counting, arithmetic and dates are weak spots for this class of model. Compute them in code and pass them alongside, e.g. with a `ColumnTransformer`. At most, ask the model for a named bucket.
4. **Ask only about the row.** "Does the note say the part is on backorder?" can be answered from the text. "Is this supplier reliable?" can only be answered from the model's priors.
5. **Bound every answer space.** A choice over an enumerated list works; an open-ended question doesn't.
6. **Treat the row as hostile** if users write it. See [Untrusted text](security.md).

## From questions to columns

| Question | Columns |
|---|---|
| `noul` | `name`, P(true) |
| `choice` with K options | `name__<option>` × K |
| `score` with L levels | `name__ev` (the expected level 0..L-1) and/or `name__L0` … (`score_repr`) |
| choice or score, with `include_confidence=True` | `name__confidence`, when the model reports one |

`feature_groups_` maps each question to its columns, for pruning a question at a time.

Choose the output scale with `link`:

- **`"logit"`** (default) gives log-odds, the right scale for linear models; tree models are unaffected by it. Probabilities are clipped first; see `logit_eps`, which adapts to models that round their answers.
- **`"identity"`** gives raw probabilities, easier to read.
- **`"clr"`** gives the centred log-ratio of each choice or score question: the natural geometry when a question is used as an embedding.

Use `drop_redundant=True` to drop the last column of each simplex, since the columns sum to 1, when an unregularized linear model follows.

## Keep banks in version control

A bank is plain JSON:

```python
from sklearn_decision import load_bank, save_bank

save_bank(bank, "questions.json")
bank = load_bank("questions.json")   # validated on load
```

## Prune with data

Start broad, with 100–300 candidate questions, then keep the ones that earn their place. {func}`~sklearn_decision.grouped_permutation_importance` permutes all of a question's columns together, on held-out data:

```python
from sklearn_decision import grouped_permutation_importance

feat = clf[0]                             # the fitted QuestionFeaturizer in a pipeline
Z = feat.transform(X_valid)
ranked = grouped_permutation_importance(clf[-1], Z, y_valid, feat.feature_groups_)
```

Choose the bank on validation data only; the bank is a fitted hyperparameter. In our 20 Newsgroups benchmark, a single direct topic question beat the full 14-question bank with few labels: extra questions cost more in variance than they added. When the classes can be named in one question, ask it.
