"""Recalibrating zero-shot answers: which recipe, with how many labels?

Compares, on the test split, a zero-shot ChoiceClassifier's raw probabilities
with the same classifier calibrated on n labelled training posts:

  sigmoid / isotonic / temperature   CalibratedClassifierCV(FrozenEstimator(clf), method=...)
  logistic head                      make_pipeline(QuestionFeaturizer(topic question, link="logit"),
                                                   LogisticRegression())

Runs from the answer cache written by learning_curves.py (no new model calls):

    python benchmarks/calibration.py --model jev-latest --tag full
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BANK_20NG, load_20ng  # noqa: E402
from learning_curves import HERE, TOPIC_TO_CLASS, TruncateLike, make_model, sample_labels, scores, slug  # noqa: E402

from sklearn_decision import ChoiceClassifier, QuestionFeaturizer  # noqa: E402

METHODS = ["sigmoid", "isotonic", "temperature"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--tag", default="")
    ap.add_argument("--truncate-like", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=256)
    ap.add_argument("--option-permutations", type=int, default=1,
                    help="local models: average choice answers over this many option rotations")
    ap.add_argument("--max-cost-usd", type=float, default=0.0, help="0: cache only, never call a hosted model")
    ap.add_argument("--sizes", default="16,64,400")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    X_tr, y_tr, X_te, y_te, names = load_20ng(400, 300)
    model = make_model(args)
    cache = str(HERE / "cache" / f"{slug(args.model)}.sqlite")
    state_fn = TruncateLike(args.truncate_like) if args.truncate_like else None
    template = QuestionFeaturizer(cache_path=cache, state_fn=state_fn)
    topic = BANK_20NG["topic"]
    to_label = {names.index(c): t for t, c in TOPIC_TO_CLASS.items()}
    yl_tr = np.array([to_label[c] for c in y_tr])
    yl_te = np.array([to_label[c] for c in y_te])

    clf = ChoiceClassifier(topic["instructions"], topic["criteria"], model=model, featurizer=template).fit(X_te)
    labels = list(clf.classes_)
    y_idx = np.array([labels.index(v) for v in yl_te])
    out = {"model": args.model, "tag": args.tag, "raw": scores(y_idx, clf.predict_proba(X_te), len(labels)),
           "calibrated": {}}
    print("raw zero-shot:", {k: round(v, 3) for k, v in out["raw"].items()})

    head = make_pipeline(QuestionFeaturizer({"topic": topic}, model=model, link="logit", cache_path=cache,
                                            state_fn=state_fn), LogisticRegression(max_iter=2000))
    for n in [len(X_tr) if s == "all" else int(s) for s in args.sizes.split(",")]:
        reps = 1 if n >= len(X_tr) else args.repeats
        for r in range(reps):
            idx = sample_labels(y_tr, n, np.random.default_rng(1000 + r))
            X_cal, y_cal = [X_tr[i] for i in idx], yl_tr[idx]
            # the classifier is frozen, so folds only re-predict; cv=2 needs just 2 labels per class
            recipes = {m: CalibratedClassifierCV(FrozenEstimator(clf), method=m, cv=2) for m in METHODS}
            recipes["logistic head"] = head
            for name, est in recipes.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        est.fit(X_cal, y_cal)
                        P = est.predict_proba(X_te)
                except (ValueError, TypeError) as e:  # e.g. temperature needs scikit-learn >= 1.8
                    print(f"  {name} unavailable: {e}")
                    continue
                cols = [list(est.classes_).index(lb) for lb in labels]
                res = scores(y_idx, P[:, cols], len(labels))
                out["calibrated"].setdefault(name, {}).setdefault(str(len(idx)), []).append(res)
        row = {m: np.mean([x["log_loss"] for x in c[str(len(idx))]]) for m, c in out["calibrated"].items()}
        print(f"n={len(idx):4d} log-loss " + "  ".join(f"{m}={v:.3f}" for m, v in row.items()))
    path = HERE / "results" / (slug(args.model) + (f"-{args.tag}" if args.tag else "")) / "calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
