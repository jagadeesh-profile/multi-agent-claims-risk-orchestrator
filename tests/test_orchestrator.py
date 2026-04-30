"""Tests for orchestrator helpers (decision parsing, fence stripping, recovery)."""
from __future__ import annotations

import asyncio

import pytest

from src import orchestrator
from src.orchestrator import _parse_decision, _strip_markdown_fence, run_case


def test_strip_markdown_fence_removes_json_fence() -> None:
    fenced = '```json\n{"k": 1}\n```'
    assert _strip_markdown_fence(fenced) == '{"k": 1}'


def test_strip_markdown_fence_removes_bare_fence() -> None:
    fenced = '```\n{"k": 1}\n```'
    assert _strip_markdown_fence(fenced) == '{"k": 1}'


def test_strip_markdown_fence_passes_through_unfenced() -> None:
    payload = '{"k": 1}'
    assert _strip_markdown_fence(payload) == payload


def test_parse_decision_handles_fenced_json() -> None:
    raw = '```json\n{"risk_level": "HIGH", "anomaly_score": 0.81}\n```'
    decision = _parse_decision(raw)
    assert decision["risk_level"] == "HIGH"
    assert decision["anomaly_score"] == 0.81


def test_parse_decision_handles_plain_json() -> None:
    decision = _parse_decision('{"risk_level": "LOW"}')
    assert decision["risk_level"] == "LOW"


def test_parse_decision_passes_dict_through() -> None:
    raw = {"risk_level": "MEDIUM"}
    assert _parse_decision(raw) == raw


def test_parse_decision_returns_error_envelope_for_invalid_json() -> None:
    decision = _parse_decision("not json at all")
    assert decision["error"] == "ActionAgent did not produce valid JSON"
    assert decision["raw"] == "not json at all"


class _FakeSession:
    def __init__(self, state: dict) -> None:
        self.state = state


class _FakeSessionService:
    def __init__(self, state: dict) -> None:
        self._state = state

    async def create_session(self, **_kwargs) -> _FakeSession:
        return _FakeSession(self._state)

    async def get_session(self, **_kwargs) -> _FakeSession:
        return _FakeSession(self._state)


class _FakeRunner:
    def __init__(self, state: dict, raise_after: bool) -> None:
        self._state = state
        self._raise_after = raise_after

    def run_async(self, **_kwargs):
        async def gen():
            self._state["final_decision"] = '{"risk_level": "LOW", "recommended_action": "AUTO_APPROVE"}'
            if self._raise_after:
                raise RuntimeError("503 UNAVAILABLE: post-pipeline transient error")
            if False:
                yield None  # make this an async generator
        return gen()


def _patch_runner(monkeypatch, *, raise_after: bool) -> dict:
    state: dict = {}
    monkeypatch.setattr(orchestrator, "InMemorySessionService", lambda: _FakeSessionService(state))
    monkeypatch.setattr(orchestrator, "Runner", lambda **_kw: _FakeRunner(state, raise_after))
    monkeypatch.setattr(
        orchestrator.genai_types,
        "Content",
        lambda **_kw: object(),
    )
    monkeypatch.setattr(
        orchestrator.genai_types,
        "Part",
        lambda **_kw: object(),
    )
    return state


def test_run_case_recovers_decision_when_runner_raises_after_action(monkeypatch) -> None:
    _patch_runner(monkeypatch, raise_after=True)
    result = asyncio.run(run_case({"patient_id": "P_TEST"}))
    assert result["decision"]["risk_level"] == "LOW"
    assert result["decision"]["recommended_action"] == "AUTO_APPROVE"


def test_run_case_reraises_when_runner_fails_before_action(monkeypatch) -> None:
    state: dict = {}
    monkeypatch.setattr(orchestrator, "InMemorySessionService", lambda: _FakeSessionService(state))

    class _EarlyFailRunner:
        def run_async(self, **_kwargs):
            async def gen():
                raise RuntimeError("boom")
                if False:
                    yield None
            return gen()

    monkeypatch.setattr(orchestrator, "Runner", lambda **_kw: _EarlyFailRunner())
    monkeypatch.setattr(orchestrator.genai_types, "Content", lambda **_kw: object())
    monkeypatch.setattr(orchestrator.genai_types, "Part", lambda **_kw: object())

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run_case({"patient_id": "P_TEST"}))
