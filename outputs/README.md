# Reviewer Outputs

This folder keeps the small, public result artifacts that are useful for a
quick repository review. Runtime decision JSONs are still written to
`outputs/decisions/` and remain gitignored.

## Included Artifacts

- `eval_output_summary.png` - live evaluation summary rendered from
  `eval/results.json`.
- `model_output_summary.png` - model metrics and top RF features rendered from
  `models/*.json`.

## Tested Results

Latest verification on May 4, 2026:

- `python -m pytest tests/ -v` completed with 44 passed tests.
- GitHub CI `Offline tests (no API key)` completed successfully on PR #1.
- Tracked key-pattern scan returned no committed credentials.
- Tracked tool-trace scan found no public development-tool traces.
