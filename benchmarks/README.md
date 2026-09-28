# Benchmarks

**Latest results:** [RESULTS.md](RESULTS.md) (Qwen2.5-0.5B-Instruct on CPU).

Harness for the experiment plan in [docs/design.md](../docs/design.md) §4. Every script takes `--model`:
a registered name (`hf:<repo id>[@revision]` for a local open-weights model, `jev-latest` for TypeSafe's
hosted API) or anything `resolve_model` accepts.

| Script | Plan phase | What it does |
|---|---|---|
| `behaviour_checks.py` | Phase 1 | saturation, noise floor, option-order bias, IIA, rewording stability, prompt injection, co-question coupling |
| `learning_curves.py` | Phase 4 | test accuracy / macro-F1 / log-loss / ECE vs. number of labels for each arm |
| `calibration.py` | | zero-shot recalibration recipes vs. number of labels, from cached answers |
| `injection_defenses.py` | | whether fencing the state or caveating the questions reduces prompt injection (it doesn't) |
| `report.py` | | chart (`learning_curves.png`) and table (`learning_curves.md`) from a results directory |
| `data.py` | | 20 Newsgroups loader, the question bank, rewordings and injection strings |

```bash
pip install -e ".[local]" sentence-transformers matplotlib
```

```bash
python benchmarks/behaviour_checks.py --model hf:Qwen/Qwen2.5-0.5B-Instruct
```

```bash
python benchmarks/learning_curves.py --model hf:Qwen/Qwen2.5-0.5B-Instruct
```

```bash
python benchmarks/report.py results/Qwen_Qwen2.5-0.5B-Instruct
```

## Arms

| Arm | Features | Head |
|---|---|---|
| A | TF-IDF, fitted on the labelled rows | logistic regression |
| B | sentence embeddings (`all-MiniLM-L6-v2`) | logistic regression |
| C | none: `ChoiceClassifier` zero-shot over the topic question | none |
| E | the 14-question bank in `data.py`, log-odds, standardized | logistic regression |
| F | A + E | logistic regression |
| G | B + E | logistic regression |
| H | only the topic choice question (zero-shot, recalibrated) | logistic regression |
| I | only the 13 yes/no questions (E without the topic question) | logistic regression |

Arm D (`ChoiceEncoder` exemplar codebook) is supported by the package but not in these runs: a
25-exemplar prompt is ~2.5k tokens per row, too slow on the CPU used here.

## Answer caches

Model answers are cached in `cache/<model>.sqlite`, keyed by model, prompt template, question and
row. The cache files are committed, so the numbers and charts can be regenerated (and the docs
built) without a GPU, an API key or a model download: `learning_curves.py` finds every answer in
the cache and only re-fits the cheap heads.
