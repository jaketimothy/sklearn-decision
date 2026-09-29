"""
Quickstart: questions as features
=================================

A bank of typed questions becomes a feature matrix, and any scikit-learn model
can sit on top. This example uses :class:`~sklearn_decision.FakeModel`, which
answers deterministically from a hash of the row and question, so it runs
anywhere with no model and no network; its answers carry no meaning. Swap in
``model="jev-latest"`` or ``model="hf:<repo id>"`` for real ones.
"""

# %%
# A question bank
# ---------------
# Three question types: a yes/no statement (noul), a pick-one (choice), and an
# ordinal rubric (score).
from sklearn_decision import FakeModel, QuestionFeaturizer, choice, noul, score

bank = {
    "asks_refund": noul("The customer asks for money back."),
    "product": choice("Which product is discussed?", ["app", "api", "billing"]),
    "urgency": score("How urgent is the request?", ["can wait", "this week", "today"]),
}
texts = [
    "Please refund my last invoice, the app crashed all week.",
    "How do I rotate my API key?",
    "Billing charged me twice this month.",
    "The API returns 500 errors since this morning, urgent!",
]

feat = QuestionFeaturizer(bank, model=FakeModel(), link="identity").fit(texts)  # fit makes no model calls
print(feat.get_feature_names_out())

# %%
# Features as a DataFrame
# -----------------------
# The featurizer supports scikit-learn's ``set_output``, so the columns keep
# their names: one per noul question, one per choice option, and the expected
# level of a score question. ``link="identity"`` above keeps raw probabilities;
# the default, ``"logit"``, gives log-odds, the better scale for linear models.
feat.set_output(transform="pandas")
feat.transform(texts).round(3)

# %%
# In a pipeline
# -------------
# Features feed any estimator; by default they are log-odds. Answers are cached
# per (model, question, row), so refitting, cross-validating and grid-searching
# don't ask again.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

y = [1, 0, 1, 0]
clf = make_pipeline(QuestionFeaturizer(bank, model=FakeModel()), LogisticRegression()).fit(texts, y)
clf.predict(texts)

# %%
# What it cost
# ------------
# ``usage()`` reports model calls, cache hits and the model versions behind the
# answers. The second featurizer found every answer in the shared in-memory cache.
print("first featurizer:", feat.usage())
print("pipeline featurizer:", clf[0].usage())
