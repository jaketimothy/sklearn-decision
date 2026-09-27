"""Render benchmark results: a learning-curve chart and a markdown summary.

    python benchmarks/report.py results/<model-slug>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Reference categorical palette, slots 1-5 in order (validated: adjacent CVD dE >= 9.1,
# normal-vision >= 19.6). Slots follow the arm's identity, never its rank, and the same arm
# keeps its color in every chart. Three slots sit below 3:1 on the light surface, so every
# line also gets a distinct marker and a direct end label, and the table is the chart's
# table view. Arms F and G are in the tables only, to keep the charts readable.
STYLE = {
    "E: questions + LR": ("#2a78d6", "o"),
    "B: embeddings + LR": ("#eb6834", "s"),
    "A: TF-IDF + LR": ("#1baf7a", "^"),
    "H: topic question + LR": ("#eda100", "D"),
    "I: yes/no questions + LR": ("#e87ba4", "v"),
}
TABLE_ORDER = ["H: topic question + LR", "E: questions + LR", "I: yes/no questions + LR", "B: embeddings + LR",
               "A: TF-IDF + LR", "F: TF-IDF + questions", "G: embeddings + questions"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def summarize(curves: dict) -> dict:
    """arm -> sorted [(n, mean, std)] of accuracy."""
    out = {}
    for arm, by_n in curves.items():
        out[arm] = sorted((int(n), float(np.mean([r["accuracy"] for r in runs])),
                           float(np.std([r["accuracy"] for r in runs]))) for n, runs in by_n.items())
    return out


def _draw(ax, res: dict, title: str, legend: bool) -> None:
    """One learning-curve panel: the plotted arms, a zero-shot reference line, end labels."""
    import matplotlib

    summary = summarize(res["curves"])
    ax.set_facecolor(SURFACE)
    ends = []
    for arm, (color, marker) in STYLE.items():
        if arm not in summary:
            continue
        n, mean, std = (np.array(v) for v in zip(*summary[arm]))
        ax.fill_between(n, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.plot(n, mean, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm)
        ends.append([mean[-1], n[-1], arm.split(":")[0]])
    ends.sort()  # end labels, spread symmetrically so close finishes stay readable
    gap = 0.03  # about one line of 9pt text at this figure size
    for _ in range(100):
        for i in range(1, len(ends)):
            d = ends[i][0] - ends[i - 1][0]
            if d < gap:
                ends[i - 1][0] -= (gap - d) / 2
                ends[i][0] += (gap - d) / 2
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
    zs = res["zero_shot"]["accuracy"]
    ax.axhline(zs, color=MUTED, linewidth=1.5, label="C: zero-shot, no labels (gray line)")
    ax.set_xscale("log", base=2)
    ns = sorted({n for v in summary.values() for n, _, _ in v})
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Labelled training examples", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(f"{title}\nzero-shot, no labels: {zs:.1%}", loc="left", fontsize=11, color=INK)
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")


def _figure(ncols: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols - 1 * (ncols - 1), 4.8), dpi=150, facecolor=SURFACE,
                             sharey=True, squeeze=False)
    return plt, fig, axes[0]


def _model_label(res: dict) -> str:
    name = res["model"].removeprefix("hf:")
    versions = res.get("model_versions") or []
    if versions and res["model"].startswith("jev"):
        name = f"{name} ({versions[0]})"
    view = ", first 160 tokens of each post" if res.get("truncate_like") or res["model"].startswith("hf:") \
        else ", full posts"
    return name + view


def plot(res: dict, path: Path) -> None:
    plt, fig, (ax,) = _figure(1)
    _draw(ax, res, f"20 Newsgroups, 4 classes: accuracy vs. labels\nquestion answers from {_model_label(res)}",
          legend=True)
    ax.set_ylabel("Test accuracy", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def compare(results: list[dict], path: Path) -> None:
    """Side-by-side panels on one y-scale: same data, same arms, different answering model."""
    plt, fig, axes = _figure(len(results))
    for i, (ax, res) in enumerate(zip(axes, results)):
        _draw(ax, res, f"Question answers from\n{_model_label(res)}", legend=i == len(results) - 1)
    axes[0].set_ylabel("Test accuracy", color=INK_2)
    fig.suptitle("20 Newsgroups, 4 classes: accuracy vs. labels", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def table(res: dict) -> str:
    summary = summarize(res["curves"])
    ns = sorted({n for v in summary.values() for n, _, _ in v})
    lines = ["| Arm | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---:|" * len(ns)]
    for arm in TABLE_ORDER:
        if arm in summary:
            cells = {n: f"{m:.3f} ± {s:.3f}" if s else f"{m:.3f}" for n, m, s in summary[arm]}
            lines.append(f"| {arm} | " + " | ".join(cells.get(n, "") for n in ns) + " |")
    zs = res["zero_shot"]
    lines.append("| C: zero-shot (no labels) | " + " | ".join(f"{zs['accuracy']:.3f}" for _ in ns) + " |")
    return "\n".join(lines)


def main(argv=None):
    """report.py <results dir> [<results dir> ...]: per-run chart and table for each;
    with several, also a side-by-side comparison.png in the results root."""
    root = Path(__file__).resolve().parent
    dirs = [Path(d) if Path(d).is_absolute() else root / d for d in (argv or sys.argv[1:])]
    results = []
    for out in dirs:
        res = json.loads((out / "learning_curves.json").read_text(encoding="utf-8"))
        results.append(res)
        plot(res, out / "learning_curves.png")
        (out / "learning_curves.md").write_text(table(res) + "\n", encoding="utf-8")
        print(f"## {out.name}\n{table(res)}\n")
    if len(results) > 1:
        compare(results, root / "results" / "comparison.png")
        print("wrote", root / "results" / "comparison.png")


if __name__ == "__main__":
    main()
