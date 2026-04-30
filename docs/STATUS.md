# Claims-Risk Orchestrator - Status

**Last updated:** 2026-04-30

## TL;DR

Local POC is code-complete and freshly baselined. The deterministic/offline
test suite passes, OpenTelemetry pins are compatible with `google-adk`, and
the live Case A path now agrees with the README: `AUTO_APPROVE` / `LOW` across
3 of 3 eval runs.

The live eval still shows useful production-risk signal: Case B and Case C
need more output-contract hardening before their LinkedIn numbers should be
presented as stable product behavior.

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
| Claims RF holdout | 0.7321 | 0.0606 | 5-fold CV AUC 0.6960 +/- 0.0767; log loss 0.2520 |
| Labs NN holdout | 0.9382 | 0.0782 | accuracy 0.9056; log loss 0.2647 |
| RF Optuna sweep | best CV AUC 0.7170 | n/a | 30 trials; best params in `models/claims_rf_best_params.json` |

RF best params:

```json
{
  "n_estimators": 600,
  "max_depth": 16,
  "min_samples_leaf": 8,
  "min_samples_split": 20,
  "max_features": "sqrt"
}
```

### Live Evaluation

Generated report: `eval/results.json` and `eval/results.md`

| Case | Modal action | Agreement | p50 latency | p95 latency | Failure rate |
|------|--------------|-----------|-------------|-------------|--------------|
| A (P_MARY) | `AUTO_APPROVE` | 100.0% | 43.85s | 51.63s | 0.0% |
| B (P_ROBERT) | `UNKNOWN` | 66.7% | 82.82s | 263.73s | 0.0% |
| C (P_LINDA) | `PARSE_ERROR/UNKNOWN` | 33.3% | 63.24s | 69.34s | 33.3% |

Action distributions:

| Case | Distribution |
|------|--------------|
| A | `AUTO_APPROVE`: 3 |
| B | `UNKNOWN`: 2, `FLAG_FOR_AUDIT`: 1 |
| C | `PARSE_ERROR/UNKNOWN`: 1, `ESCALATE_TO_HUMAN`: 1, `ROUTINE_FOLLOWUP`: 1 |

MLflow screenshot: `docs/mlflow_ui.png`

## What Works

- `src/tools.py` - lazy TF import, defensive `available:false` envelopes,
  no top-level crashes.
- `src/agents.py` - guarded ADK imports; full topology unchanged:
  - `RoutedParallelAgent` honors RouterAgent's run_claims/run_labs/run_notes flags.
  - `FusionAgent`, `ReviewerAgent`, and `ActionAgent` now run at temperature 0.0.
  - Fusion prompt explicitly preserves mild Case A-style signals as LOW risk
    with confidence above the 0.7 escalation threshold.
  - `LoopExitChecker` exits ReviewLoop early on pass or escalation.
- `src/orchestrator.py` - markdown-fence stripping, post-pipeline error
  recovery, per-step latency capture, `per_agent_ms` rollup.
- `src/main.py` - CLI with env checks plus 429/503/ExceptionGroup-aware
  error rendering.
- OpenTelemetry pins - `1.37.0` across the OTLP/API/SDK/proto family is the
  lowest tested set that both satisfies `google-adk` and imports at runtime.

## Test Suite

Latest offline verification:

```bash
python -m pytest tests/ -v
```

Result: **32 passed, 2 warnings**.

| File | Count | Coverage |
|------|-------|----------|
| `tests/test_pipeline.py` | 7 | tool layer (RF, NN, audit log, error envelopes) |
| `tests/test_agents.py` | 4 | model selection, routing, Case A prompt/temperature contract |
| `tests/test_main.py` | 3 | CLI error message rendering |
| `tests/test_orchestrator.py` | 9 | fence stripping, decision parsing, recovery |
| `tests/test_secrets.py` | 1 | guards against committing Gemini-shaped API keys |
| `tests/test_loop_exit.py` | 8 | LoopExitChecker decision logic + loop wiring |
| **Total** | **32** | offline-only - none consume Gemini quota |

## Known Gaps

- RF holdout AUC is **0.7321**, below the earlier 0.75 target. The tuned sweep
  found best CV AUC **0.7170**, so the current synthetic label signal may need
  feature/label-generation work rather than more random tuning.
- Case B live eval agreement is **66.7%** and p95 latency is **263.73s**.
- Case C live eval agreement is **33.3%** with a **33.3%** parse-error rate.
- Live output contracts should be tightened further for Cases B/C before using
  those as stability claims.
- Audit log rotation is still pending.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env       # edit: GOOGLE_API_KEY=<your-key>

# 3. Generate + train
python -m src.generate_data --n 2000 --out data
python -m src.train_claims_rf
python -m src.train_labs_nn

# 4. Tune + inspect
python -m src.tune_claims_rf --trials 30
mlflow ui --backend-store-uri ./mlruns

# 5. Test
python -m pytest tests/ -v

# 6. Live eval
python -m eval.run_eval --runs 3
```
