# Choice codebooks

{class}`~sklearn_decision.ChoiceEncoder` asks one choice question per *view*, over a codebook of options, and outputs each row's distribution over the codebook. Each row becomes a point on the simplex.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn_decision import ChoiceEncoder

enc = ChoiceEncoder("exemplars", {
    "content": "Which reference example is most similar in subject matter?",
    "intent":  "Which reference example has the most similar goal or request?",
}, anchor=("none", "No reference is a meaningful match"), random_state=0, model="jev-latest")
clf = make_pipeline(enc, LogisticRegression()).fit(texts, y)
```

## Codebooks

- **Concept codebooks:** a taxonomy of labels, `ChoiceEncoder(["billing", "outage", ...])` or `{label: description}`. The output is a soft classification.
- **Exemplar codebooks:** `"exemplars"` samples `n_exemplars` training rows in `fit`, stratified by `y`, and uses them as options: "which reference is this most like?". The output is a similarity profile against landmarks, like a Nyström kernel approximation.

## Blocks and stitching

A model can only take so many options per question: 255 for Jev, 26 for local models, which answer with letters. Larger codebooks are split into blocks automatically, using the model's `capabilities()`.

`stitch=True` rebuilds one distribution over the whole codebook through an `anchor` option that every block shares. This is only valid if the model obeys *independence of irrelevant alternatives* (IIA): removing an option mustn't change the ratios between the others. In our benchmarks neither Jev nor a small local model obeyed it fully, so treat stitched output as an approximation.

## Exemplars matching themselves

A row that is itself an exemplar trivially picks its own option. `self_match` controls what happens, both in `fit_transform` and for any identical row seen later, so exemplars can stay in the training set:

- **`"reask"`** (default): the row is asked the affected question again without its own option. This is exact, and costs one extra question per exemplar row and view.
- **`"renormalize"`**: zero the row's own option and renormalize the rest. It's free, but assumes IIA.
- **`"keep"`**: leave the answers untouched, for ablations.

Exemplar text is sent to the model as option descriptions, so it is as untrusted as the rows themselves.
