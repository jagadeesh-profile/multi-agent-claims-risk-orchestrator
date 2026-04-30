"""Tests for synthetic training-data signal quality."""
from __future__ import annotations

from src.generate_data import make_claims


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
