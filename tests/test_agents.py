"""Tests for deterministic agent orchestration helpers."""
from __future__ import annotations

from src.agents import (
    CASE_A_MILD_SIGNAL_GUIDANCE,
    FINAL_RISK_THRESHOLDS,
    MODEL_FAST,
    MODEL_SMART,
    action_agent,
    fusion_agent,
    reviewer_agent,
    specialist_team,
)


def test_smart_model_uses_free_tier_friendly_model() -> None:
    assert MODEL_SMART == MODEL_FAST == "gemini-2.5-flash"


def test_specialist_team_selects_only_routed_agents() -> None:
    selected = specialist_team._selected_sub_agents(
        {"run_claims": True, "run_labs": False, "run_notes": True}
    )

    assert [agent.name for agent in selected] == ["ClaimsAgent", "NotesAgent"]


def test_specialist_team_defaults_invalid_routing_to_all_agents() -> None:
    selected = specialist_team._selected_sub_agents("not-json")

    assert [agent.name for agent in selected] == [
        "ClaimsAgent",
        "LabsAgent",
        "NotesAgent",
    ]


def test_case_a_mild_signals_are_pinned_to_low_risk_contract() -> None:
    case_a_signals = {
        "claims_score": 0.3906,
        "labs_score": 0.2186,
        "notes_severity_score": 0.1,
        "billing_consistency": "consistent",
    }

    max_signal = max(
        case_a_signals["claims_score"],
        case_a_signals["labs_score"],
        case_a_signals["notes_severity_score"],
    )

    assert max_signal <= CASE_A_MILD_SIGNAL_GUIDANCE["max_mild_signal_score"]
    assert case_a_signals["billing_consistency"] == "consistent"
    assert (
        CASE_A_MILD_SIGNAL_GUIDANCE["target_fused_anomaly_score_max"]
        < FINAL_RISK_THRESHOLDS["medium"]
    )
    assert (
        CASE_A_MILD_SIGNAL_GUIDANCE["target_fused_confidence_min"]
        >= FINAL_RISK_THRESHOLDS["escalate"]
    )
    assert "mild Case A" in fusion_agent.instruction
    assert fusion_agent.generate_content_config.temperature == 0.0
    assert reviewer_agent.generate_content_config.temperature == 0.0
    assert action_agent.generate_content_config.temperature == 0.0
