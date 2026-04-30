"""Tests for CLI-facing behavior."""
from __future__ import annotations

import json

from src.main import format_runtime_error, load_case_from_path, save_decision


def test_format_runtime_error_explains_gemini_high_demand() -> None:
    message = format_runtime_error(
        RuntimeError("503 UNAVAILABLE: model is currently experiencing high demand")
    )

    assert "Gemini service is temporarily unavailable" in message
    assert "try again later" in message


def test_format_runtime_error_explains_quota_exhaustion() -> None:
    message = format_runtime_error(
        RuntimeError("429 RESOURCE_EXHAUSTED: free-tier quota exceeded")
    )

    assert "Gemini quota is exhausted" in message
    assert "wait for quota reset" in message


def test_format_runtime_error_unwraps_exception_group() -> None:
    message = format_runtime_error(
        ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError("503 UNAVAILABLE: temporary model overload")],
        )
    )

    assert "Gemini service is temporarily unavailable" in message


def test_load_case_from_path_reads_custom_case_json(tmp_path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "patient_id": "P_CUSTOM",
                "claim": {
                    "cost_usd": 12000,
                    "procedure_count": 3,
                    "los_days": 2,
                    "age": 61,
                    "drg_code": "470",
                },
                "labs": None,
                "notes": "Follow-up note.",
            }
        )
    )

    case = load_case_from_path(case_path)

    assert case["patient_id"] == "P_CUSTOM"
    assert case["claim"]["cost_usd"] == 12000


def test_save_decision_writes_patient_decision_json(tmp_path) -> None:
    output_path = save_decision(
        patient_id="P_CUSTOM",
        decision={"recommended_action": "AUTO_APPROVE"},
        out_dir=tmp_path,
    )

    assert output_path.name.startswith("P_CUSTOM_")
    assert output_path.suffix == ".json"
    assert json.loads(output_path.read_text())["recommended_action"] == "AUTO_APPROVE"
