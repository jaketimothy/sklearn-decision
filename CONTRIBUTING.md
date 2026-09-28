# Contributing

Issues and pull requests are welcome.

```bash
pip install -e ".[dev,local,docs]"
```

```bash
pytest -q
```

```bash
ruff check src tests benchmarks examples docs
```

```bash
python -m sphinx -b html docs docs/_build/html -W
```

The tests and the docs build run offline: tests use `FakeModel` or a tiny in-memory model, and the examples read the committed answer caches. `pytest -m live` calls the Jev API and needs `TYPESAFE_API_KEY`.

For a pull request:
- open it against `main`;
- add a test, which for a bug fix should fail without the fix;
- update the docs and add a line to `CHANGELOG.md` if users will notice the change.

**Never commit API keys.** Put them in `TYPESAFE_API_KEY`, ideally as a 1Password `op://` reference.
