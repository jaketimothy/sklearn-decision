## Bottom line

<!-- One or two sentences: what changes for users, and why. -->

## Changes

<!-- What you changed, grouped by area. Note anything that changes a default or existing behaviour. -->

## Testing

<!-- How you verified it: tests added, commands run, benchmark numbers (with the model and data). -->

## Checklist

- [ ] Tests cover the change; a bug fix has a test that fails without it
- [ ] `pytest -q` and `ruff check src tests benchmarks examples docs` pass
- [ ] Docs and docstrings updated; the docs build with `-W`
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] No API keys, tokens or private data in the diff
