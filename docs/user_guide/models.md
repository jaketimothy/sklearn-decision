# Choosing a model

Every estimator needs `model=`. Pass a registered name, or an instance to configure it. Instance parameters are nested, so grid search can tune them, e.g. `model__timeout`.

| `model=` | Runs | Data leaves the machine | Notes |
|---|---|---|---|
| `"jev-latest"`, `"jev-preview"` | TypeSafe's hosted API ({class}`~sklearn_decision.JevModel`) | yes, to TypeSafe | Needs `TYPESAFE_API_KEY`. Fast and cheap; answers rounded to 0.01. |
| `"hf:<repo id>[@revision]"` | locally, open weights ({class}`~sklearn_decision.TransformersModel`) | no | Needs the `local` extra; a GPU for anything but small models. |
| `"fake-1"` or `FakeModel()` | in-process, deterministic ({class}`~sklearn_decision.FakeModel`) | no | For tests and examples; its answers mean nothing. |

## Jev

```python
from sklearn_decision import JevModel

JevModel("jev-latest", timeout=60, max_questions_per_call=25, max_cost_usd=5.0)
```

- **Versions.** The API lists only `jev-latest` and `jev-preview`; there is no pinned name. Every answer records the concrete version that produced it (e.g. `jev-1.13.0`). A feature matrix that mixes versions raises a warning; when TypeSafe upgrades, start a fresh `cache_path`.
- **Credentials.** `api_key=` or `TYPESAFE_API_KEY` holds the key, or a [1Password secret reference](https://developer.1password.com/docs/cli/secret-references/) such as `op://Personal/Typesafe API/password`. A reference is resolved with `op read` on the first request and kept only in process memory, never on the estimator, so it can't leak through a repr or a pickle. Fitting never needs the key. For scripts, `op run -- python script.py` resolves it once for the whole run.
- **Spending cap.** `max_cost_usd` caps Jev spending in the whole process, across every clone a grid search or cross-validation makes. Before each chunk of requests, its estimated cost plus the spend so far is checked against the cap; a chunk that would pass it isn't sent, and answers already fetched stay cached. `JevModel.process_spend_usd()` reads the count and `JevModel.reset_process_spend()` zeroes it. Parallel worker processes (`n_jobs > 1`) each count separately.
- **Rounding.** Probabilities come back to 2 decimals. The featurizer clips log-odds at half a step, so a reported 0.00 doesn't become an outlier.
- **Other endpoints.** `base_url` points at any server that speaks the same `/v1/systemone` wire format.

## Local open-weights models

```python
from sklearn_decision import TransformersModel

TransformersModel("google/gemma-4-12b-it", revision="<commit sha>", dtype="bfloat16",
                  batch_size=16, max_state_tokens=2048)
```

A frozen instruction-tuned causal LM answers from its next-token logits: "Yes"/"No" for noul, option letters A–Z for choice, level digits 0–9 for score. It's deterministic, and nothing is generated.

- **Speed.** Each row's text is encoded once, and its key/value cache is shared by all of that row's questions.
- **Position bias.** Choice questions are asked under 4 rotations of their options by default (`n_option_permutations`), and the answers averaged. For Qwen2.5-0.5B this raised zero-shot accuracy from 65% to 72%.
- **Pin `revision`** to a commit for reproducible features; the resolved commit is recorded with every answer. Weights load lazily on the first call and are shared by clones.
- **Prompts.** `templates`, `use_chat_template`, `chat_template_kwargs` (e.g. `{"enable_thinking": False}`) and `answer_prefix` control the prompt, and all of them are part of the cache key.
- **Limits.** Choice questions are limited to 26 options (letters) and score rubrics to 10 levels (digits). `ChoiceEncoder` splits larger codebooks into blocks.
- **No native confidence.** These models report none, so `predict_confidence` isn't offered.

Small models are a floor, not a recommendation: in our benchmarks, Qwen2.5-0.5B's question features didn't beat sentence embeddings. For real work, use a model in the 4B–30B range on a GPU.

## Your own backend

Subclass {class}`~sklearn_decision.DecisionModel` and register a name prefix:

```python
from sklearn_decision import Capabilities, DecisionModel, NoulAnswer, DistAnswer, Response, register_model

class MyModel(DecisionModel):
    def __init__(self, endpoint="http://localhost:8000"):
        self.endpoint = endpoint

    def cache_namespace(self):
        return f"mine:{self.endpoint}"        # everything that changes the answers

    def capabilities(self):
        return Capabilities(max_choice_options=50, native_confidence=False, deterministic=True)

    def answer(self, items):
        # items: [(state, {question name: spec})]; return one Response (or exception) per item
        ...

register_model("mine:", lambda name: MyModel())
```

`answer` returns answers in one schema: {class}`~sklearn_decision.NoulAnswer` for noul questions, and {class}`~sklearn_decision.DistAnswer` for choice and score questions, with probabilities in the order the question lists its options. Everything else (caching, chunking, progress, links, estimators) works unchanged. Optionally, set `chunk_size` (requests per call to `answer`, default 32) and override `cost_usd()` for a priced backend, so progress lines and `usage()` report spend.
