"""
Zero-shot classification with Jev, and why to calibrate
=======================================================

A :class:`~sklearn_decision.ChoiceClassifier` whose classes are the options of
one choice question classifies without any labels. Its accuracy can be good,
but its probabilities are overconfident. :func:`~sklearn_decision.calibrate_zero_shot`
fixes that from a handful of labels without asking the model anything new.

The data is four 20 Newsgroups categories. Jev's answers come from the cache
committed with the repository (see the note on the examples index), so this
runs offline and free.
"""

# %%
# Data and question
# -----------------
# The benchmark's loader and question, so the answers are the cached ones.
import sys
from pathlib import Path

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:  # sphinx-gallery runs each example from the examples/ directory
    REPO = Path.cwd().resolve().parent
sys.path.insert(0, str(REPO / "benchmarks"))
from data import BANK_20NG, load_20ng  # noqa: E402

X_train, y_train, X_test, y_test, names = load_20ng(400, 300)
topic = BANK_20NG["topic"]
print(topic["instructions"])
print(topic["criteria"])

# %%
# Zero-shot
# ---------
# ``fit`` only records the label set. ``max_cost_usd=0`` makes a cache miss
# raise instead of calling the API.
import numpy as np

from sklearn_decision import ChoiceClassifier, JevModel, QuestionFeaturizer

JEV = JevModel("jev-latest", max_cost_usd=0.0)
CACHE = str(REPO / "benchmarks" / "cache" / "jev-latest.sqlite")
to_topic = {"alt.atheism": "atheism", "comp.graphics": "graphics", "talk.religion.misc": "religion",
            "sci.space": "space"}
yt_train = np.array([to_topic[names[c]] for c in y_train])
yt_test = np.array([to_topic[names[c]] for c in y_test])

clf = ChoiceClassifier(topic["instructions"], topic["criteria"], model=JEV,
                       featurizer=QuestionFeaturizer(cache_path=CACHE)).fit(X_test)
P_raw = clf.predict_proba(X_test)
print(f"zero-shot accuracy: {(clf.classes_[P_raw.argmax(1)] == yt_test).mean():.1%}")
print("model versions:", clf.usage()["versions"])

# %%
# Calibrate on 16 labels
# ----------------------
# Four labelled posts per class. The classifier is frozen: its answers for
# these posts come from the cache too.
from sklearn.metrics import log_loss

from sklearn_decision import calibrate_zero_shot

rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(yt_train == c), 4, replace=False) for c in clf.classes_])
cal = calibrate_zero_shot(clf, [X_train[i] for i in idx], yt_train[idx])
P_cal = cal.predict_proba(X_test)

# Jev reports options it rules out as exactly 0.00, so one confident miss can
# dominate log-loss. Clip at 1e-9 before scoring, as the benchmarks do.
def clipped(P):
    P = np.clip(P, 1e-9, 1)
    return P / P.sum(axis=1, keepdims=True)


for label, P in [("raw", P_raw), ("calibrated", P_cal)]:
    print(f"{label:>10}: accuracy {(clf.classes_[P.argmax(1)] == yt_test).mean():.1%}, "
          f"log-loss {log_loss(yt_test, clipped(P), labels=clf.classes_):.2f}")

# %%
# Reliability
# -----------
# Group test posts by how confident the prediction was, and compare with how
# often it was right. A calibrated model sits on the diagonal. Raw answers sit
# far below it at the top: "100% sure" is right much less often than that.
import matplotlib.pyplot as plt

BINS = np.linspace(0.25, 1, 6)  # 4 classes: the top class always has >= 0.25


def reliability(P, bins=BINS):
    conf, correct = P.max(1), clf.classes_[P.argmax(1)] == yt_test
    which = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
    rows = [(conf[which == b].mean(), correct[which == b].mean()) for b in range(len(bins) - 1) if (which == b).any()]
    return np.array(rows).T


SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, label="perfectly calibrated")
for label, P, color, marker in [("raw zero-shot", P_raw, "#eb6834", "s"),
                                ("calibrated on 16 labels", P_cal, "#2a78d6", "o")]:
    x, acc = reliability(P)
    ax.plot(x, acc, color=color, marker=marker, linewidth=2, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label)
ax.set_xlim(0.2, 1.02)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Predicted probability of the chosen class", color=INK_2)
ax.set_ylabel("Share of those predictions that were right", color=INK_2)
ax.set_title("Jev zero-shot on 20 Newsgroups: reliability", loc="left", color=INK, fontsize=11)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(colors=MUTED, length=0)
ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
fig.tight_layout()
