# Calibration

Decision models' probabilities are often **overconfident**. On 20 Newsgroups, Jev's zero-shot topic answers were 77% accurate with a log-loss of 1.73: confidently wrong on the posts it missed. Calibrate on your own labels before you threshold, rank or combine probabilities.

## With a few labels: `calibrate_zero_shot`

```python
from sklearn_decision import ChoiceClassifier, calibrate_zero_shot

clf = ChoiceClassifier("Which team should handle this?", {"billing": None, "api": None}, model="jev-latest")
cal = calibrate_zero_shot(clf, X_labelled, y_labelled)     # 2+ labels per class
cal.predict_proba(X_new)
```

{func}`~sklearn_decision.calibrate_zero_shot` fits sigmoid (Platt) scaling on the **frozen** classifier: `CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid", cv=2)`.

- **Nothing is refitted or re-queried;** the calibration labels' answers come from the cache if they're already there.
- **`cv=2`** means only two labels per class are needed. Because the classifier is frozen, the folds only re-predict, so `cv` doesn't change the result.

## With more labels: a logistic head

From about 16 labels per class, learning a small head on the answers' log-odds does as well or better:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn_decision import QuestionFeaturizer, choice

head = make_pipeline(
    QuestionFeaturizer({"team": choice("Which team should handle this?", ["billing", "api"])},
                       model="jev-latest", link="logit"),
    LogisticRegression(),
).fit(X_labelled, y_labelled)
```

## Measured

Jev's log-loss on held-out posts (raw: 1.73), from [calibration.py](https://github.com/jaketimothy/sklearn-decision/blob/main/benchmarks/calibration.py):

| Recipe | 16 labels | 64 labels | 400 labels |
|---|---:|---:|---:|
| `calibrate_zero_shot` (sigmoid) | **0.71** | **0.64** | 0.62 |
| Logistic head on log-odds | 0.95 | 0.65 | **0.60** |
| Temperature scaling (scikit-learn ≥ 1.8) | 1.06 | 1.08 | 0.71 |
| Isotonic | 3.43 | 1.37 | 0.87 |

- **Avoid isotonic calibration until you have hundreds of labels;** with few labels it made things worse than raw.
- **Temperature scaling keeps the predicted class** but fitted poorly from a handful of labels.

## Local models: average over option orders first

Small local models are also *position-biased*: their answer depends on where an option appears in the list. `TransformersModel` averages each choice question over 4 rotations of its options by default. For Qwen2.5-0.5B that alone cut zero-shot log-loss from 1.35 to 0.79, better than sigmoid calibration on top of a single ordering. See [Choosing a model](models.md).
