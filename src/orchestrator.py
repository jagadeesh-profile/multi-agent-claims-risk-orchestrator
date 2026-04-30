"""
Run the ADK pipeline for one patient case and return the structured decision.

This module is the thin glue between user-facing entry points (CLI, Streamlit,
tests) and the agents defined in src/agents.py. It owns session creation,
input formatting, and result extraction.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fence(payload: str) -> str:
    """Strip a single ```json ... ``` (or ``` ... ```) fence if present."""
    match = _FENCE_RE.match(payload)
    return match.group(1) if match else payload


def _parse_decision(raw: Any) -> dict[str, Any]:
    """Parse the ActionAgent output, tolerating LLM-style markdown fences."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {"raw": raw}
    try:
        return json.loads(_strip_markdown_fence(raw))
    except json.JSONDecodeError:
        return {"error": "ActionAgent did not produce valid JSON", "raw": raw}


def _parse_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(_strip_markdown_fence(raw))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _decision_from_fusion(patient_id: str, fusion_result: Any) -> dict[str, Any]:
    """Deterministic fallback for when ActionAgent spends its turn on a tool call."""
    fusion = _parse_state_dict(fusion_result)
    score = float(fusion.get("fused_anomaly_score", 0.5))
    confidence = float(fusion.get("fused_confidence", 0.0))

    if score >= 0.7:
        risk_level = "HIGH"
    elif score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if confidence < 0.7:
        action = "ESCALATE_TO_HUMAN"
    elif risk_level == "HIGH":
        action = "FLAG_FOR_AUDIT"
    elif risk_level == "MEDIUM":
        action = "ROUTINE_FOLLOWUP"
    else:
        action = "AUTO_APPROVE"

    return {
        "patient_id": patient_id,
        "risk_level": risk_level,
        "anomaly_score": round(score, 4),
        "confidence": round(confidence, 4),
        "recommended_action": action,
        "reasoning": fusion.get("reasoning", "Deterministic fallback from fusion_result."),
        "audit_trail": {
            "signals_used": fusion.get("signals_used", []),
            "conflict_detected": bool(fusion.get("conflict_detected", False)),
            "fallback": "fusion_result",
        },
    }

try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types
except ImportError as _e:
    raise ImportError(
        "google-adk and google-genai are required. Run: pip install google-adk>=0.4.0 google-genai>=0.3.0"
    ) from _e

from .agents import root_pipeline

APP_NAME = "claims-risk-orchestrator"


def _format_input(case: dict[str, Any]) -> str:
    """Format the patient case as the input message the Router will read."""
    parts = ["Patient case for claims-risk evaluation.\n"]
    parts.append(f"patient_id: {case.get('patient_id', 'UNKNOWN')}")

    if claim := case.get("claim"):
        parts.append("\nclaim:")
        parts.append(json.dumps(claim, indent=2))
    else:
        parts.append("\nclaim: <missing>")

    if labs := case.get("labs"):
        parts.append("\nlabs:")
        parts.append(json.dumps(labs, indent=2))
    else:
        parts.append("\nlabs: <missing>")

    if notes := case.get("notes"):
        parts.append("\nnotes:")
        parts.append(notes if isinstance(notes, str) else json.dumps(notes))
    else:
        parts.append("\nnotes: <missing>")

    return "\n".join(parts)


async def run_case(case: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    """Run the full pipeline against one patient case.

    Returns a dict with keys:
      - decision: the final ActionAgent JSON
      - trace:    list of (agent_name, output) tuples for observability
      - timing:   wall-clock seconds for the whole pipeline
    """
    session_service = InMemorySessionService()
    user_id = "demo-user"
    session_id = f"sess-{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_pipeline,
        session_service=session_service,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=_format_input(case))],
    )

    trace: list[dict[str, Any]] = []
    started = time.time()
    last_event_t = started
    runner_error: BaseException | None = None

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.author and event.content and event.content.parts:
                text_chunks = [p.text for p in event.content.parts if p.text]
                if text_chunks:
                    now = time.time()
                    delta_ms = round((now - last_event_t) * 1000, 1)
                    last_event_t = now
                    payload = "".join(text_chunks).strip()
                    trace.append({
                        "agent": event.author,
                        "output": payload,
                        "delta_ms": delta_ms,
                        "elapsed_sec": round(now - started, 2),
                    })
                    if verbose:
                        print(f"[{event.author} +{delta_ms:.0f}ms] {payload[:200]}")
    except BaseException as exc:  # noqa: BLE001 - capture so a late 503 doesn't shadow a written decision.
        runner_error = exc

    elapsed = round(time.time() - started, 2)

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    state = session.state if session else {}
    final_decision_raw = state.get("final_decision")

    if final_decision_raw is None:
        if state.get("fusion_result") is not None:
            decision = _decision_from_fusion(
                patient_id=str(case.get("patient_id", "UNKNOWN")),
                fusion_result=state.get("fusion_result"),
            )
        elif runner_error is not None:
            raise runner_error
        else:
            decision = _parse_decision("{}")
    else:
        decision = _parse_decision(final_decision_raw)
        if "recommended_action" not in decision and state.get("fusion_result") is not None:
            decision = _decision_from_fusion(
                patient_id=str(case.get("patient_id", "UNKNOWN")),
                fusion_result=state.get("fusion_result"),
            )

    per_agent_ms: dict[str, float] = {}
    for step in trace:
        per_agent_ms[step["agent"]] = round(
            per_agent_ms.get(step["agent"], 0.0) + step.get("delta_ms", 0.0), 1
        )

    return {
        "decision": decision,
        "trace": trace,
        "timing_sec": elapsed,
        "per_agent_ms": per_agent_ms,
    }
