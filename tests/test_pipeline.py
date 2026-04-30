"""
POC tests — verify the model-loading and tool layer.

These do NOT call Gemini (that requires API key + budget). They prove the
deterministic core works:
  - Random Forest model loads and predicts
  - TensorFlow NN loads and predicts
  - score_labs_risk gracefully handles missing panels
  - write_audit_log creates the expected jsonl row

Run: pytest tests/ -v
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.tools import (
    predict_claims_anomaly,
    score_labs_risk,
    write_audit_log,
)


def test_claims_tool_returns_unavailable_when_model_prediction_fails(monkeypatch) -> None:
    class BrokenModel:
        def predict_proba(self, row):
            raise PermissionError("named pipe creation denied")

    monkeypatch.setattr("src.tools._load_claims_rf", lambda: BrokenModel())

    out = predict_claims_anomaly(
        cost_usd=10000.0,
        procedure_count=2,
        los_days=3,
        age=65,
        drg_code="470",
    )

    assert out["available"] is False
    assert "claims prediction failed" in out["reason"].lower()


@pytest.mark.skipif(
    not Path("models/claims_rf.joblib").exists(),
    reason="Run `python -m src.train_claims_rf` first",
)
def test_claims_tool_returns_valid_score() -> None:
    out = predict_claims_anomaly(
        cost_usd=10000.0,
        procedure_count=2,
        los_days=3,
        age=65,
        drg_code="470",
    )
    assert out["available"] is True
    assert 0.0 <= out["score"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0
    assert "RF score" in out["rationale"]
    # top_features is XAI metadata. Optional (empty when shap is missing),
    # but when present each entry must have the documented shape.
    assert "top_features" in out
    assert isinstance(out["top_features"], list)
    for entry in out["top_features"]:
        assert {"feature", "shap_value", "direction"} <= set(entry)
        assert entry["direction"] in {"increases_anomaly", "decreases_anomaly"}


@pytest.mark.skipif(
    not Path("models/labs_nn.keras").exists(),
    reason="Run `python -m src.train_labs_nn` first",
)
def test_labs_tool_handles_full_panel() -> None:
    out = score_labs_risk(a1c=7.0, ldl=130, egfr=70, troponin=0.05)
    assert out["available"] is True
    assert 0.0 <= out["score"] <= 1.0


def test_labs_tool_handles_all_missing() -> None:
    out = score_labs_risk()
    assert out["available"] is False
    assert "no lab panels" in out["reason"].lower()


def test_labs_tool_handles_all_missing_without_loading_model(monkeypatch) -> None:
    def fail_if_loaded():
        raise AssertionError("model should not load when all labs are missing")

    monkeypatch.setattr("src.tools._load_labs_nn", fail_if_loaded)

    out = score_labs_risk()

    assert out["available"] is False
    assert "no lab panels" in out["reason"].lower()


@pytest.mark.skipif(
    not Path("models/labs_nn.keras").exists(),
    reason="Run `python -m src.train_labs_nn` first",
)
def test_labs_tool_imputes_partial() -> None:
    out = score_labs_risk(a1c=7.0)
    assert out["available"] is True
    assert "imputed" in out["rationale"]


def test_audit_log_writes_jsonl(monkeypatch) -> None:
    work_dir = (Path("tmp") / f"audit-test-{uuid.uuid4().hex}").resolve()
    work_dir.mkdir(parents=True)
    monkeypatch.chdir(work_dir)
    decision = {"risk_level": "HIGH", "anomaly_score": 0.81}
    out = write_audit_log("P_TEST", decision)
    assert out["logged"] is True
    log_lines = (work_dir / "logs" / "audit.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1
    record = json.loads(log_lines[0])
    assert record["patient_id"] == "P_TEST"
    assert record["decision"] == decision
