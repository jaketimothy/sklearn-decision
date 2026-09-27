"""Render benchmark results: a learning-curve chart and a markdown summary.

    python benchmarks/report.py results/<model-slug>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Reference categorical palette (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6).
# Slots follow the arm's identity, never its rank. Three slots sit below 3:1 on the light
# surface, so every line also gets a distinct marker and a direct end label, and the
# markdown table is the chart's table view.
STYLE = {
    "E: questions + LR": ("#2a78d6", "o"),
    "B: embeddings + LR": ("#eb6834", "s"),
    "A: TF-IDF + LR": ("#1baf7a", "^"),
    "F: TF-IDF + questions": ("#eda100", "D"),
    "G: embeddings + questions": ("#e87ba4", "v"),
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def summarize(curves: dict) -> dict:
    """arm -> sorted [(n, mean, std)] of accuracy."""
    out = {}
    for arm, by_n in curves.items():
        out[arm] = sorted((int(n), float(np.mean([r["accuracy"] for r in runs])),
                           float(np.std([r["accuracy"] for r in runs]))) for n, runs in by_n.items())
    return out


def plot(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize(res["curves"])
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for arm in STYLE:
        if arm not in summary:
            continue
        color, marker = STYLE[arm]
        n, mean, std = (np.array(v) for v in zip(*summary[arm]))
        ax.fill_between(n, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.plot(n, mean, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm)
        ends.append([mean[-1], n[-1], arm.split(":")[0]])
    # end labels, nudged apart so close finishes stay readable
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.013)
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
    zs = res["zero_shot"]["accuracy"]
    ax.axhline(zs, color=MUTED, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ns = sorted({n for v in summary.values() for n, _, _ in v})
    ax.annotate(f"C: zero-shot, no labels ({zs:.1%})", (np.sqrt(ns[2] * ns[3]), zs), xytext=(0, -5),
                textcoords="offset points", ha="center", va="top", fontsize=9, color=INK_2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Labelled training examples", color=INK_2)
    ax.set_ylabel("Test accuracy", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    model = res["model"].removeprefix("hf:")
    ax.set_title(f"20 Newsgroups, 4 classes: accuracy vs. labels\nquestion answers from {model}",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def table(res: dict) -> str:
    summary = summarize(res["curves"])
    ns = sorted({n for v in summary.values() for n, _, _ in v})
    lines = ["| Arm | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---:|" * len(ns)]
    for arm in STYLE:
        if arm in summary:
            cells = {n: f"{m:.3f} ± {s:.3f}" if s else f"{m:.3f}" for n, m, s in summary[arm]}
            lines.append(f"| {arm} | " + " | ".join(cells.get(n, "") for n in ns) + " |")
    zs = res["zero_shot"]
    lines.append("| C: zero-shot (no labels) | " + " | ".join(f"{zs['accuracy']:.3f}" for _ in ns) + " |")
    return "\n".join(lines)


def main(argv=None):
    out = Path((argv or sys.argv[1:])[0])
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    res = json.loads((out / "learning_curves.json").read_text(encoding="utf-8"))
    plot(res, out / "learning_curves.png")
    (out / "learning_curves.md").write_text(table(res) + "\n", encoding="utf-8")
    print(table(res))
    print("wrote", out / "learning_curves.png")


if __name__ == "__main__":
    main()
