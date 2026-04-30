"""Tests for CLI-facing behavior."""
from __future__ import annotations

from src.main import format_runtime_error


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
