"""Shared pytest fixtures and skip markers for the claims-risk-orchestrator tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


models_rf_present = pytest.mark.skipif(
    not Path("models/claims_rf.joblib").exists(),
    reason="Run `python -m src.train_claims_rf` first",
)

models_nn_present = pytest.mark.skipif(
    not Path("models/labs_nn.keras").exists(),
    reason="Run `python -m src.train_labs_nn` first",
)

api_key_present = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — set it in .env for live API tests",
)


@pytest.fixture(scope="session")
def google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key
