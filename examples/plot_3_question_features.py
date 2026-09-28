"""
A question bank as a feature encoder
====================================

Thirteen yes/no questions, none of which names the classes, answered by Jev
and fed to a logistic regression. With few labels they beat TF-IDF by a wide
margin, because each feature carries the model's prior knowledge. Grouped
permutation importance then shows which questions the model actually uses.

Jev's answers come from the cache committed with the repository, so this runs
offline and free.
"""

# %%
# The bank
# --------
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:  # sphinx-gallery runs each example from the examples/ directory
    REPO = Path.cwd().resolve().parent
sys.path.insert(0, str(REPO / "benchmarks"))
from data import BANK_20NG, load_20ng  # noqa: E402

bank = {q: spec for q, spec in BANK_20NG.items() if spec["type"] == "noul"}  # no topic question
for name, spec in bank.items():
    print(f"{name:>20}: {spec['instructions']}")

# %%
# Featurize once
# --------------
# ``fit`` makes no calls and the featurizer is stateless, so the whole dataset
# can be featurized once and the downstream model cross-validated cheaply.
from sklearn_decision import JevModel, QuestionFeaturizer

X_train, y_train, X_test, y_test, names = load_20ng(400, 300)
CACHE = str(REPO / "benchmarks" / "cache" / "jev-latest.sqlite")
feat = QuestionFeaturizer(bank, model=JevModel("jev-latest", max_cost_usd=0.0), link="logit",
                          cache_path=CACHE).fit(X_train)
Z_train, Z_test = feat.transform(X_train), feat.transform(X_test)
print(Z_train.shape, feat.usage())

# %%
# Accuracy vs. number of labels
# -----------------------------
# Question features against TF-IDF, both with a logistic regression, averaged
# over 5 random draws of the labelled posts.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sizes, results = [8, 16, 32, 64, 128, 400], {"questions": [], "TF-IDF": []}
for n in sizes:
    draws = {k: [] for k in results}
    for seed in range(1 if n == 400 else 5):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(y_train == c), min(n // 4, (y_train == c).sum()),
                                         replace=False) for c in np.unique(y_train)])
        q = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Z_train[idx], y_train[idx])
        draws["questions"].append(q.score(Z_test, y_test))
        t = make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(max_iter=2000))
        t.fit([X_train[i] for i in idx], y_train[idx])
        draws["TF-IDF"].append(t.score(X_test, y_test))
    for k in results:
        results[k].append(np.mean(draws[k]))
    print(f"n={n:3d}  questions {results['questions'][-1]:.1%}  TF-IDF {results['TF-IDF'][-1]:.1%}")

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)


fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120, facecolor=SURFACE)
for label, color, marker in [("questions", "#2a78d6", "o"), ("TF-IDF", "#1baf7a", "^")]:
    ax.plot(sizes, results[label], color=color, marker=marker, linewidth=2, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.annotate(label, (sizes[-1], results[label][-1]), xytext=(8, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [str(n) for n in sizes])
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Labelled training posts", color=INK_2)
ax.set_ylabel("Test accuracy", color=INK_2)
style(ax, "13 yes/no questions (Jev) vs. TF-IDF")
ax.legend(frameon=False, labelcolor=INK_2, loc="lower right")
fig.tight_layout()

# %%
# Which questions matter
# ----------------------
# :func:`~sklearn_decision.grouped_permutation_importance` shuffles one
# question's columns at a time on held-out data and measures the drop in
# accuracy. Questions near zero are candidates for pruning.
from sklearn_decision import grouped_permutation_importance

head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Z_train, y_train)
ranked = grouped_permutation_importance(head, Z_test, y_test, feat.feature_groups_, n_repeats=10)

fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=120, facecolor=SURFACE)
labels, drops, stds = zip(*ranked[::-1])
ax.barh(labels, drops, xerr=stds, color="#2a78d6", height=0.6, error_kw={"ecolor": MUTED, "elinewidth": 1})
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{round(x * 100):d}%"))
ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.05))
ax.set_xlabel("Drop in test accuracy when the question is shuffled", color=INK_2)
style(ax, "Question importance (400 labels)")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, linewidth=0.8)
fig.tight_layout()
