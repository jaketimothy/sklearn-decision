"""Model-behaviour checks, for any decision model.

Run these before trusting a model as an encoder:

  saturation   are noul answers graded, or stuck at 0 / 1?
  noise        do identical requests return identical answers?
  order        do choice probabilities move when options are reordered?
  iia          does dropping an option preserve the others' ratios?
               (why ChoiceEncoder re-asks exemplar rows instead of renormalizing)
  rewording    do reworded questions rank rows the same way?
  injection    do instructions hidden in the text move the answers?
  coupling     do answers depend on the other questions in the request?
               (only for models that batch questions per request, e.g. Jev)

    python benchmarks/behaviour_checks.py --model hf:Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BANK_20NG, INJECTIONS, REWORDINGS, load_20ng  # noqa: E402
from learning_curves import make_model, slug  # noqa: E402

from sklearn_decision import JevModel, QuestionFeaturizer, choice, clear_memory_cache  # noqa: E402

HERE = Path(__file__).resolve().parent
NOULS = [q for q, s in BANK_20NG.items() if s["type"] == "noul"]
TOPIC = BANK_20NG["topic"]


def answers(model, bank, X, cache):
    f = QuestionFeaturizer(bank, model=model, link="identity", cache_path=cache).fit(X)
    return f.transform(X), f


def check_saturation(model, X, cache):
    P, _ = answers(model, {q: BANK_20NG[q] for q in NOULS}, X, cache)
    graded = (P > 0.02) & (P < 0.98)
    return {
        "share_graded": float(graded.mean()),
        "share_graded_by_question": dict(zip(NOULS, np.round(graded.mean(axis=0), 3).tolist())),
        "mean_abs_logodds": float(np.abs(np.log(np.clip(P, 1e-6, 1 - 1e-6) / np.clip(1 - P, 1e-6, 1))).mean()),
        "histogram": np.histogram(P, bins=10, range=(0, 1))[0].tolist(),
        "pass": bool(graded.mean() > 0.5),
    }


def check_noise(model, X, repeats=2):
    runs = []
    for _ in range(repeats):
        clear_memory_cache()
        runs.append(answers(model, {q: BANK_20NG[q] for q in NOULS[:3]}, X, None)[0])
    clear_memory_cache()
    R = np.stack(runs)  # (repeats, rows, questions)
    spread = R.max(axis=0) - R.min(axis=0)
    crosses = ((R > 0.5).any(axis=0) & (R <= 0.5).any(axis=0)).mean()
    return {"repeats": repeats, "max_abs_diff": float(spread.max()), "mean_abs_diff": float(spread.mean()),
            "std_mean": float(R.std(axis=0).mean()), "share_crossing_0.5": float(crosses),
            "pass": bool(np.quantile(spread, 0.9) < 0.05)}


def check_order(model, X, cache):
    labels = list(TOPIC["criteria"])
    per_order = []
    for shift in range(len(labels)):
        order = labels[shift:] + labels[:shift]
        q = choice(TOPIC["instructions"], {lb: TOPIC["criteria"][lb] for lb in order})
        P, f = answers(model, {"t": q}, X, cache)
        cols = [list(f.get_feature_names_out()).index(f"t__{lb}") for lb in labels]
        per_order.append((P[:, cols], order[0]))
    stack = np.stack([p for p, _ in per_order])  # (orders, rows, labels)
    argmax_agree = float((stack.argmax(axis=2) == stack[0].argmax(axis=1)).all(axis=0).mean())
    first_bias = float(np.mean([p[:, labels.index(first)].mean() - p.mean() for p, first in per_order]))
    spread = float(stack.std(axis=0).mean())
    return {"mean_prob_std_across_orders": spread, "argmax_agreement": argmax_agree,
            "first_position_excess_prob": first_bias, "pass": bool(argmax_agree > 0.9 and spread < 0.05)}


def check_iia(model, X, cache):
    labels = list(TOPIC["criteria"])
    full, f = answers(model, {"t": TOPIC}, X, cache)
    names = list(f.get_feature_names_out())
    pf = {lb: full[:, names.index(f"t__{lb}")] for lb in labels}
    # A model that rounds (Jev: to 0.01) reports most minor options as 0.00 in
    # both runs, a trivial "no change". Ratios are only measurable where both
    # options clear the rounding floor in both runs, so report those separately.
    res = f.model_.capabilities().probability_resolution or 0.0
    floor = max(2 * res, 1e-6)
    diffs, measurable = [], []
    for drop in labels:
        keep = [lb for lb in labels if lb != drop]
        P, g = answers(model, {"t": choice(TOPIC["instructions"], {lb: TOPIC["criteria"][lb] for lb in keep})},
                       X, cache)
        gn = list(g.get_feature_names_out())
        ps = {lb: P[:, gn.index(f"t__{lb}")] for lb in keep}
        for a, b in itertools.combinations(keep, 2):
            lr_full = np.log(np.clip(pf[a], 1e-6, 1)) - np.log(np.clip(pf[b], 1e-6, 1))
            lr_sub = np.log(np.clip(ps[a], 1e-6, 1)) - np.log(np.clip(ps[b], 1e-6, 1))
            diffs.append(np.abs(lr_full - lr_sub))
            measurable.append(np.minimum.reduce([pf[a], pf[b], ps[a], ps[b]]) >= floor)
    d, m = np.concatenate(diffs), np.concatenate(measurable)
    dm = d[m] if m.any() else np.array([np.nan])
    return {"median_abs_logratio_change": float(np.median(d)), "p90_abs_logratio_change": float(np.quantile(d, 0.9)),
            "measurable_pairs": int(m.sum()), "total_pairs": int(m.size),
            "median_abs_logratio_change_measurable": float(np.median(dm)),
            "p90_abs_logratio_change_measurable": float(np.quantile(dm, 0.9)),
            "pass": bool(m.sum() >= 20 and np.median(dm) < 0.25)}


def check_rewording(model, X, cache):
    A, _ = answers(model, {q: BANK_20NG[q] for q in REWORDINGS}, X, cache)
    B, _ = answers(model, REWORDINGS, X, cache)
    rho = {q: float(spearmanr(A[:, j], B[:, j]).statistic) for j, q in enumerate(REWORDINGS)}
    return {"spearman_by_question": rho, "median_spearman": float(np.median(list(rho.values()))),
            "pass": bool(np.median(list(rho.values())) > 0.7)}


def check_injection(model, X, cache):
    bank = {q: BANK_20NG[q] for q in NOULS} | {"topic": TOPIC}
    clean, f = answers(model, bank, X, cache)
    noul_cols = [f.feature_groups_[q][0] for q in NOULS]
    topic_cols = f.feature_groups_["topic"]

    def shift(suffix):
        P, _ = answers(model, bank, [x + "\n\n" + suffix for x in X], cache)
        flips = ((P[:, noul_cols] > 0.5) != (clean[:, noul_cols] > 0.5)).mean()
        topic_changed = (P[:, topic_cols].argmax(1) != clean[:, topic_cols].argmax(1)).mean()
        return {"mean_abs_noul_change": float(np.abs(P[:, noul_cols] - clean[:, noul_cols]).mean()),
                "noul_flip_rate": float(flips), "topic_change_rate": float(topic_changed)}

    control = shift("Thanks for reading, and have a good day.")
    attacks = {s: shift(s) for s in INJECTIONS}
    worst = max(a["noul_flip_rate"] for a in attacks.values())
    worst_topic = max(a["topic_change_rate"] for a in attacks.values())
    return {"control": control, "attacks": attacks, "worst_flip_rate": worst, "worst_topic_change_rate": worst_topic,
            "pass": bool(worst <= control["noul_flip_rate"] + 0.05
                         and worst_topic <= control["topic_change_rate"] + 0.05)}


def check_coupling(model, X, cache):
    if not isinstance(model, JevModel):
        return {"skipped": "questions are answered in isolated prompts by this model"}
    bank = {q: BANK_20NG[q] for q in NOULS}
    # The cache key deliberately ignores batching (that is the assumption under
    # test), so each half must start from an empty in-memory cache, and a second
    # "together" run gives the noise floor to compare against.
    runs = {}
    for label, m in [("together", model), ("together_again", model),
                     ("alone", clone(model).set_params(max_questions_per_call=1))]:
        clear_memory_cache()
        runs[label], _ = answers(m, bank, X, None)
    clear_memory_cache()
    coupling = np.abs(runs["together"] - runs["alone"])
    noise = np.abs(runs["together"] - runs["together_again"])
    return {"mean_abs_diff": float(coupling.mean()), "p90_abs_diff": float(np.quantile(coupling, 0.9)),
            "noise_mean_abs_diff": float(noise.mean()), "noise_p90_abs_diff": float(np.quantile(noise, 0.9)),
            "pass": bool(coupling.mean() <= 2 * noise.mean() + 0.005)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="hf:Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-state-tokens", type=int, default=256)
    ap.add_argument("--max-cost-usd", type=float, default=2.0, help="spending cap for hosted models")
    ap.add_argument("--noise-repeats", type=int, default=2, help="identical runs for the noise check")
    ap.add_argument("--n", type=int, default=100, help="rows per check (injection and noise use a quarter)")
    ap.add_argument("--n-train", type=int, default=400, help="training pool drawn as in learning_curves.py")
    ap.add_argument("--checks", default="saturation,noise,order,iia,rewording,injection,coupling")
    args = ap.parse_args(argv)

    model = make_model(args)
    cache = str(HERE / "cache" / f"{slug(args.model)}.sqlite")
    X, *_ = load_20ng(args.n_train, 4)  # training rows only; the test split stays untouched
    X = X[: args.n]
    path = HERE / "results" / slug(args.model) / "behaviour_checks.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}  # rerun checks replace theirs
    out.update({"model": args.model, "n_rows": len(X)})
    for name in args.checks.split(","):
        fn = globals()[f"check_{name}"]
        rows = X[: max(8, args.n // 4)] if name in ("injection", "noise") else X
        out[name] = fn(model, rows, args.noise_repeats) if name == "noise" else fn(model, rows, cache)
        summary = {k: v for k, v in out[name].items() if not isinstance(v, (dict, list))}
        print(f"{name:10s} {summary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
