# Reviewer Outputs

This folder keeps the small, public result artifacts that are useful for a
quick repository review. Runtime decision JSONs are still written to
`outputs/decisions/` and remain gitignored.

## Included Artifacts

- `eval_output_summary.png` - live evaluation summary rendered from
  `outputs/evaluation/results.json`.
- `model_output_summary.png` - model metrics and top RF features rendered from
  `outputs/model_metrics/*.json`.
- `evaluation/results.json` and `evaluation/results.md` - live Gemini
  evaluation results. The run used a local `GOOGLE_API_KEY` from the ignored
  environment file; no key is committed.
- `model_metrics/*.json` - training and tuning metrics copied into the public
  reviewer output path.
- `public_data_summary.md` - row counts from the downloaded public sample
  mapping. The mapped CSVs stay gitignored because they are large generated
  artifacts.
- `terminal/pytest_output.txt` - raw terminal output from the offline test run.
- `verified_output_summary.json` - compact JSON summary of tested outputs,
  live Gemini evaluation results, model metrics, and public-data mapping counts.

## Tested Results

Latest verification on May 4, 2026:

- `python -m pytest tests/ -v` completed with 44 passed tests.
- `python -m pytest tests/ -q` raw terminal output is saved in
  `outputs/terminal/pytest_output.txt`.
- `python -m src.train_claims_rf` wrote RF model output metrics to
  `outputs/model_metrics/`.
- `python -m src.train_labs_nn` wrote Labs NN output metrics to
  `outputs/model_metrics/`.
- Public-data mapper produced `claims.csv` 66,494 rows, `labs.csv` 5,928 rows,
  and `notes.csv` 222 rows under ignored `data_public/`.
- Live Gemini evaluation produced 100% decision agreement across sample cases
  A, B, and C in `outputs/evaluation/results.json`.
- GitHub CI `Offline tests (no API key)` completed successfully on PR #1.
- Tracked key-pattern scan returned no committed credentials.
- Tracked tool-trace scan found no public development-tool traces.
