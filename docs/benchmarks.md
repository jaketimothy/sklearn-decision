# Benchmarks

The full results, with every table, caveat and reproduction command, are in [benchmarks/RESULTS.md](https://github.com/jaketimothy/sklearn-decision/blob/main/benchmarks/RESULTS.md). This page is the summary.

**Setup.** 20 Newsgroups, 4 classes (atheism, graphics, space, religion.misc); 400 training posts, 300 test posts. The question bank has 13 yes/no questions and one four-way topic question. Two answering models ran through the same harness:
- **Jev**, which reported itself as `jev-1.13.0`;
- **Qwen2.5-0.5B-Instruct**, running locally on a laptop CPU as a floor.

![Accuracy vs. labels, Qwen and Jev side by side](https://raw.githubusercontent.com/jaketimothy/sklearn-decision/main/benchmarks/results/comparison.png)

| Accuracy | 8 labels | 400 labels |
|---|---:|---:|
| Jev, zero-shot (no labels) | 76.7% | 76.7% |
| Jev, topic question + recalibration | 68.5% | **78.0%** |
| Jev, 13 yes/no questions only | **69.9%** | 75.3% |
| MiniLM sentence embeddings | 58.1% | 73.7% |
| TF-IDF | 35.3% | 68.0% |
| Qwen2.5-0.5B, 13 yes/no questions only | 59.2% | 68.7% |

**Findings:**
- **With Jev, question features beat sentence embeddings** at every label count, even without the question that names the classes. With the small local model they don't.
- **When the classes can be named, one direct question is the strongest arm.**
- **Jev's raw probabilities are overconfident;** see [Calibration](user_guide/calibration.md).

**Behaviour checks:**
- **Jev passes:** noise (answers repeat to within about 0.01), option order (no primacy bias), rewording stability, and co-question coupling. Coupling passing is what makes per-question caching sound.
- **Jev fails:**
  - saturation: answers pile up at 0 and 1;
  - independence of irrelevant alternatives;
  - injection: see [Untrusted text](user_guide/security.md).

**Limits.** It's one dataset, where the classes can be named; the obvious next test is one where they can't. The local model is tiny. Open models in the 4B–30B range, which independent comparisons report as close to Jev, haven't been run yet.
