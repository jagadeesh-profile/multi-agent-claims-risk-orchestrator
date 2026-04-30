"""Tests for synthetic training-data signal quality."""
from __future__ import annotations

from src.generate_data import make_claims
from src.generate_data import make_labs
from src.lab_schema import LAB_FEATURES, LAB_TARGET, normalize_labs_dataframe


def test_claim_anomaly_label_is_learnable_from_claim_features() -> None:
    claims = make_claims(2000)
    high_cost = claims["cost_usd"] > claims["cost_usd"].quantile(0.85)
    rule_prediction = high_cost & (claims["procedure_count"] >= 5)
    positives = claims["is_anomaly"] == 1

    recall = (rule_prediction & positives).sum() / positives.sum()
    precision = (rule_prediction & positives).sum() / rule_prediction.sum()

    assert 0.06 <= positives.mean() <= 0.10
    assert recall >= 0.75
    assert precision >= 0.75


def test_generated_labs_use_readable_column_names() -> None:
    labs = make_labs(3, ["P1", "P2", "P3"])

    assert LAB_FEATURES + [LAB_TARGET] == [
        "hemoglobin_a1c_percent",
        "ldl_cholesterol_mg_dl",
        "estimated_gfr_ml_min_1_73m2",
        "troponin_ng_ml",
        "lab_risk_flag",
    ]
    assert set(LAB_FEATURES + [LAB_TARGET]) <= set(labs.columns)


def test_normalize_labs_dataframe_accepts_legacy_short_columns() -> None:
    legacy = make_labs(3, ["P1", "P2", "P3"]).rename(
        columns={
            "hemoglobin_a1c_percent": "a1c",
            "ldl_cholesterol_mg_dl": "ldl",
            "estimated_gfr_ml_min_1_73m2": "egfr",
            "troponin_ng_ml": "troponin",
            "lab_risk_flag": "at_risk",
        }
    )

    normalized = normalize_labs_dataframe(legacy)

    assert set(LAB_FEATURES + [LAB_TARGET]) <= set(normalized.columns)
