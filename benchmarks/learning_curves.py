"""Phase 4: learning curves over the number of labels (docs/design.md).

Every arm is scored on the same held-out test set as the labelled training
set grows. Question answers are computed once (the featurizer is stateless)
and cached in SQLite, so re-runs and new arms cost nothing.

    python benchmarks/learning_curves.py --model hf:Qwen/Qwen2.5-0.5B-Instruct
    python benchmarks/learning_curves.py --model jev-1.13          # needs TYPESAFE_API_KEY
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BANK_20NG, load_20ng  # noqa: E402

from sklearn_decision import ChoiceClassifier, QuestionFeaturizer, TransformersModel, resolve_model  # noqa: E402

HERE = Path(__file__).resolve().parent
TOPIC_TO_CLASS = {"atheism": "alt.atheism", "graphics": "comp.graphics", "religion": "talk.religion.misc",
                  "space": "sci.space"}


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", spec).strip("_")


def ece(y_true, proba, bins: int = 10) -> float:
    """Top-label expected calibration error."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return float(total)


def scores(y_true, proba, n_classes):
    proba = np.clip(proba, 1e-9, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    return {"accuracy": accuracy_score(y_true, pred), "macro_f1": f1_score(y_true, pred, average="macro"),
            "log_loss": log_loss(y_true, proba, labels=list(range(n_classes))), "ece": ece(y_true, proba)}


def make_model(args):
    if args.model.startswith("hf:"):
        name, _, rev = args.model[3:].partition("@")
        return TransformersModel(name, revision=rev or "main", device=args.device, dtype=args.dtype,
                                 batch_size=args.batch_size, max_state_tokens=args.max_state_tokens)
    return resolve_model(args.model)


def embed(texts, name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu").encode(texts, batch_size=32, normalize_embeddings=True,
                                                         show_progress_bar=False)


def sample_labels(y, n, rng):
    """n labelled rows, stratified, at least one per class."""
    classes = np.unique(y)
    per = max(1, n // len(classes))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(per, (y == c).sum()), replace=False)
                          for c in classes])
    return np.sort(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="hf:Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--sizes", default="8,16,32,64,128,256,all")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--embeddings", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-transformers model for arm B, or 'none'")
    ap.add_argument("--cache", default=None, help="answer cache (default benchmarks/cache/<model>.sqlite)")
    args = ap.parse_args(argv)

    out_dir = HERE / "results" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache or str(HERE / "cache" / f"{slug(args.model)}.sqlite")
    Path(cache).parent.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_te, y_te, names = load_20ng(args.n_train, args.n_test)
    k = len(names)
    print(f"20NG: {len(X_tr)} train / {len(X_te)} test, classes {names}")
    model = make_model(args)

    # ---- question features, computed once for every row ----
    t0 = time.time()
    feat = QuestionFeaturizer(BANK_20NG, model=model, cache_path=cache).fit(X_tr)
    print("cost estimate:", feat.estimate_cost(X_tr + X_te))
    rows = X_tr + X_te
    chunks = []
    for start in range(0, len(rows), 25):  # answers hit the cache every chunk, so a crash loses little
        chunks.append(feat.transform(rows[start : start + 25]))
        print(f"  featurized {min(start + 25, len(rows))}/{len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    Q_all = np.vstack(chunks)
    Q_tr, Q_te = Q_all[: len(X_tr)], Q_all[len(X_tr):]
    feat_seconds = time.time() - t0
    print(f"question features: {Q_all.shape} in {feat_seconds:.0f}s; usage {feat.model_.usage}")
    eps = 1e-4
    L_tr, L_te = (np.log(np.clip(Q, eps, 1 - eps)) - np.log1p(-np.clip(Q, eps, 1 - eps)) for Q in (Q_tr, Q_te))

    # ---- arm C: zero-shot, shares the bank's topic answers through the cache ----
    topic = BANK_20NG["topic"]
    zs = ChoiceClassifier(topic["instructions"], topic["criteria"], model=model,
                          featurizer=QuestionFeaturizer(cache_path=cache)).fit(X_te)
    P_topic = zs.predict_proba(X_te)
    order = [names.index(TOPIC_TO_CLASS[c]) for c in zs.classes_]
    P_zs = np.zeros_like(P_topic)
    P_zs[:, order] = P_topic
    zero_shot = scores(y_te, P_zs, k)
    print("zero-shot:", {m: round(v, 3) for m, v in zero_shot.items()})

    # ---- dense and sparse baselines ----
    E_tr = E_te = None
    if args.embeddings != "none":
        try:
            E = embed(X_tr + X_te, args.embeddings)
            E_tr, E_te = E[: len(X_tr)], E[len(X_tr):]
        except ImportError:
            print("sentence-transformers not installed: skipping arm B")

    def fit_lr(A_tr, A_te, y):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(A_tr, y)
        P = np.zeros((A_te.shape[0], k))
        P[:, clf.classes_] = clf.predict_proba(A_te)
        return P

    sizes = [len(X_tr) if s == "all" else int(s) for s in args.sizes.split(",")]
    results = {"model": args.model, "model_versions": sorted(feat.model_.versions_seen), "classes": names,
               "n_train_pool": len(X_tr), "n_test": len(X_te), "n_questions": len(BANK_20NG),
               "feature_seconds": feat_seconds, "usage": feat.model_.usage, "zero_shot": zero_shot, "curves": {}}
    for n in sizes:
        reps = 1 if n >= len(X_tr) else args.repeats
        for r in range(reps):
            idx = sample_labels(y_tr, n, np.random.default_rng(1000 + r))
            y = y_tr[idx]
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1).fit([X_tr[i] for i in idx])
            T_tr, T_te = tfidf.transform([X_tr[i] for i in idx]), tfidf.transform(X_te)
            sc = StandardScaler().fit(L_tr[idx])
            Qs_tr, Qs_te = sc.transform(L_tr[idx]), sc.transform(L_te)
            arms = {
                "A: TF-IDF + LR": (T_tr, T_te),
                "E: questions + LR": (Qs_tr, Qs_te),
                "F: TF-IDF + questions": (sparse.hstack([T_tr, Qs_tr]).tocsr(), sparse.hstack([T_te, Qs_te]).tocsr()),
            }
            if E_tr is not None:
                arms["B: embeddings + LR"] = (E_tr[idx], E_te)
                arms["G: embeddings + questions"] = (np.hstack([E_tr[idx], Qs_tr]), np.hstack([E_te, Qs_te]))
            for arm, (A_tr, A_te) in arms.items():
                res = scores(y_te, fit_lr(A_tr, A_te, y), k)
                results["curves"].setdefault(arm, {}).setdefault(str(len(idx)), []).append(res)
        row = {arm: np.mean([m["accuracy"] for m in c[str(len(idx))]]) for arm, c in results["curves"].items()}
        print(f"n={len(idx):4d}  " + "  ".join(f"{a.split(':')[0]}={v:.3f}" for a, v in row.items()))

    (out_dir / "learning_curves.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("wrote", out_dir / "learning_curves.json")


if __name__ == "__main__":
    main()
