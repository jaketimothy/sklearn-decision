# Security

## Reporting a vulnerability

Please **don't open a public issue.** Report privately through GitHub's private vulnerability reporting:

**[Report a vulnerability](https://github.com/jaketimothy/sklearn-decision/security/advisories/new)**

Include what you found, how to reproduce it, and what an attacker could do with it. You'll get an acknowledgement, and the fix will be coordinated with you before anything is published.

Only the latest release, and `main`, receive security fixes while the project is at 0.x.

## In scope

- **Credential handling.** API keys, including 1Password `op://` references, leaking into reprs, `get_params()`, pickles, logs, cache files or error messages.
- **Data leaving the machine unexpectedly.** Any path that sends rows to a hosted model without the user choosing one. The package has no default model.
- **Cache integrity.** Two different questions or models sharing a cache key, so one is served the other's answers.
- **Code execution.** Loading an answer cache, a question bank or a model registry entry running untrusted code. `TransformersModel` passes `trust_remote_code=False` unless you opt in.

## Known limitation: prompt injection

Decision models read the row as evidence, including what the row claims about itself. In our benchmark, a planted sentence changed Jev's topic answer on 20% of posts, and simple defences such as fencing the text or adding caveats to the questions didn't help. This is a property of the models, not a bug in the package. See [Untrusted text](https://jaketimothy.github.io/sklearn-decision/user_guide/security.html) for guidance. Reports of *new* injection techniques, or of defences that measurably work, are welcome as ordinary issues.

## What the package stores

Answer caches (`cache_path=...`) store only a SHA-256 hash of (model, question, row) and the model's answer. They don't store your rows' text. Question banks are plain JSON you write. Neither contains credentials.
