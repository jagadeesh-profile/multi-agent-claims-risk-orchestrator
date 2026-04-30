# Multi-Agent Claims Risk Orchestrator - Status

**Last updated:** 2026-04-30

## TL;DR

The project is now publish-ready as a claims-risk orchestration POC:

- OpenTelemetry dependency drift is fixed with compatible `1.37.0` pins.
- Case A, B, and C live eval paths each show **100% decision agreement** across
  3 runs.
- Claims RF AUC improved from **0.7321** to **0.9613** after cleaning the
  synthetic anomaly label signal.
- GitHub Actions passed on the initial GitHub push.
- MLflow screenshot is saved at `docs/mlflow_ui.png`.

## Fresh Baseline

Commands run on 2026-04-30:

```bash
python -m src.generate_data --n 2000 --out data
python -m src.train_claims_rf
python -m src.train_labs_nn
python -m src.tune_claims_rf --trials 30
python -m eval.run_eval --runs 3
```

### Model Metrics

| Model | AUC | Brier | Other |
|-------|-----|-------|-------|
| Claims RF holdout | 0.9613 | 0.0095 | 5-fold CV AUC 0.9564 +/- 0.0208; log loss 0.0671 |
| Labs NN holdout | 0.9403 | 0.0734 | accuracy 0.8972; log loss 0.2512 |
| RF Optuna sweep | best CV AUC 0.9612 | n/a | 30 trials; best params in `models/claims_rf_best_params.json` |

RF best params:

```json
{
  "n_estimators": 500,
  "max_depth": 16,
  "min_samples_leaf": 10,
  "min_samples_split": 20,
  "max_features": "log2"
}
```

### Live Evaluation

Generated report: `eval/results.json` and `eval/results.md`

| Case | Modal action | Agreement | p50 latency | p95 latency | Failure rate |
|------|--------------|-----------|-------------|-------------|--------------|
| A (P_MARY) | `AUTO_APPROVE` | 100.0% | 44.29s | 44.35s | 0.0% |
| B (P_ROBERT) | `FLAG_FOR_AUDIT` | 100.0% | 53.64s | 57.21s | 0.0% |
| C (P_LINDA) | `ESCALATE_TO_HUMAN` | 100.0% | 43.73s | 43.83s | 0.0% |

Action distributions:

| Case | Distribution |
|------|--------------|
| A | `AUTO_APPROVE`: 3 |
| B | `FLAG_FOR_AUDIT`: 3 |
| C | `ESCALATE_TO_HUMAN`: 3 |

## What Changed In This Hardening Pass

- **Case B/C prompt contracts:** FusionAgent now explicitly pins conflicted
  Case B evidence to a high-risk audit path and missing-labs Case C evidence
  to human escalation.
- **Deterministic final-decision fallback:** If ActionAgent spends its turn on
  the audit tool call and does not leave final JSON, `run_case` derives the
  final decision from `fusion_result` using the documented thresholds.
- **Cleaner synthetic RF signal:** `generate_data.py` now labels claim anomalies
  from a learnable high-cost plus procedure-count rule with low random noise.
- **Repo polish:** README now includes a CI badge, current metrics, and the
  MLflow screenshot.

## What Works

- `src/tools.py` - lazy TF import, defensive `available:false` envelopes,
  no top-level crashes.
- `src/agents.py` - full topology unchanged:
  - `RoutedParallelAgent` honors RouterAgent's run_claims/run_labs/run_notes flags.
  - Fusion/Reviewer/Refiner/Action run at temperature 0.0 for stability.
  - Case A/B/C prompt contracts are covered by offline tests.
  - `LoopExitChecker` exits ReviewLoop early on pass or escalation.
- `src/orchestrator.py` - markdown-fence stripping, post-pipeline recovery,
  deterministic fallback from fusion state, per-step latency capture, and
  `per_agent_ms` rollup.
- `src/main.py` - CLI with env checks plus 429/503/ExceptionGroup-aware
  error rendering.
- OpenTelemetry pins - `1.37.0` across the OTLP/API/SDK/proto family is the
  lowest tested set that both satisfies `google-adk` and imports at runtime.

## Test Suite

Latest offline verification:

```bash
python -m pytest tests/ -v
```

Result: **39 passed, 2 warnings**.

| File | Count | Coverage |
|------|-------|----------|
| `tests/test_pipeline.py` | 7 | tool layer (RF, NN, audit log, error envelopes) |
| `tests/test_agents.py` | 5 | model selection, routing, Case A/B/C prompt contracts |
| `tests/test_main.py` | 5 | CLI error rendering plus custom input/output helpers |
| `tests/test_orchestrator.py` | 12 | fence stripping, decision parsing, fallback recovery |
| `tests/test_training_data.py` | 1 | generated claims label separability |
| `tests/test_secrets.py` | 1 | guards against committing Gemini-shaped API keys |
| `tests/test_loop_exit.py` | 8 | LoopExitChecker decision logic + loop wiring |
| **Total** | **39** | offline-only - none consume Gemini quota |

## Remaining Future Work

- Replace synthetic data with CMS DE-SynPUF + NHANES + MIMIC-IV extracts for a
  non-demo baseline.
- Calibrate LLM-emitted confidence against outcome labels instead of relying on
  prompt contracts.
- Add audit log rotation.
- Add a short demo video or GIF for the Streamlit UI.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env       # edit: GOOGLE_API_KEY=<your-key>

python -m src.generate_data --n 2000 --out data
python -m src.train_claims_rf
python -m src.train_labs_nn
python -m pytest tests/ -v
python -m eval.run_eval --runs 3
```
