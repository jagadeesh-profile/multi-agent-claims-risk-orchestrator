"""Tests for the ReviewLoop early-exit logic.

The LoopExitChecker is a tiny BaseAgent that emits an ``actions.escalate=True``
event whenever the Reviewer says we're done — either it passed validation
(``passed=true``) or it gave up (``escalate=true``). LoopAgent honors the
escalate signal and stops iterating, so the Refiner doesn't burn an LLM
call doing a no-op rewrite.

These tests exercise the parsing + decision logic directly. They don't spin
up the ADK runtime; the escalate-emission behavior on top of this logic is
ADK's responsibility.
"""
from __future__ import annotations

from src.agents import LoopExitChecker, review_loop


def test_should_exit_true_when_review_passed() -> None:
    assert LoopExitChecker.should_exit({"passed": True}) is True


def test_should_exit_true_when_review_escalates() -> None:
    assert LoopExitChecker.should_exit({"passed": False, "escalate": True}) is True


def test_should_exit_false_when_review_failed_without_escalate() -> None:
    assert (
        LoopExitChecker.should_exit({"passed": False, "failed_rules": ["R1"]})
        is False
    )


def test_should_exit_handles_fenced_json_string() -> None:
    raw = '{"passed": true, "failed_rules": []}'
    assert LoopExitChecker.should_exit(raw) is True


def test_should_exit_handles_invalid_json_string() -> None:
    assert LoopExitChecker.should_exit("not-json-at-all") is False


def test_should_exit_handles_none() -> None:
    assert LoopExitChecker.should_exit(None) is False


def test_review_loop_includes_exit_checker_between_reviewer_and_refiner() -> None:
    names = [agent.name for agent in review_loop.sub_agents]
    assert names == ["ReviewerAgent", "LoopExitChecker", "RefinerAgent"], (
        f"Expected Reviewer -> ExitChecker -> Refiner ordering, got {names}"
    )


def test_review_loop_max_iterations_unchanged() -> None:
    assert review_loop.max_iterations == 3
