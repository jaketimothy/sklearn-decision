# sklearn-decision

**Decision models as scikit-learn estimators.** A decision model answers typed questions about a piece of text or a record with probabilities:

- **noul**: is this statement true?
- **choice**: which of these options?
- **score**: where on this ordinal rubric?

sklearn-decision turns those answers into scikit-learn estimators. You can use a single question as a zero-shot classifier or regressor, or a bank of questions as a feature encoder for any downstream model, in pipelines, cross-validation and grid search.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn_decision import QuestionFeaturizer, noul, choice, score

feat = QuestionFeaturizer({
    "asks_refund": noul("The customer asks for money back."),
    "product":     choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency":     score("How urgent is the request?", ["can wait", "this week", "today"]),
}, model="jev-latest")
clf = make_pipeline(feat, HistGradientBoostingClassifier()).fit(texts, y)
```

The model is a parameter, and you always choose it: TypeSafe's hosted Jev (`"jev-latest"`), a local open-weights model (`"hf:<repo id>"`), or your own backend. Nothing is sent anywhere until you do.

## Install

```bash
pip install "sklearn-decision[local] @ git+https://github.com/jaketimothy/sklearn-decision"
```

The package isn't on PyPI yet, so this installs from [GitHub](https://github.com/jaketimothy/sklearn-decision). The `local` extra adds `torch` and `transformers` for open-weights models; for the hosted API only, leave out `[local]`.

## What the benchmarks say

With Jev answering on 20 Newsgroups (4 classes), 13 yes/no questions used as features beat MiniLM sentence embeddings at every label count from 8 to 400. Zero-shot, with no labels, Jev scored 76.7%. Its raw probabilities are overconfident, and {func}`~sklearn_decision.calibrate_zero_shot` fixes that from a handful of labels. See [Benchmarks](benchmarks.md) for these numbers and the failures.

```{toctree}
:hidden:
:maxdepth: 2

user_guide/index
auto_examples/index
api
benchmarks
design
changelog
```
