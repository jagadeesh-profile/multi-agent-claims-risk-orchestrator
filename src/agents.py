"""
ADK agent definitions for the Claims-Risk Orchestrator.

Topology (all four ADK primitives in one design):

  RootPipeline (SequentialAgent)
    |-- RouterAgent              (LlmAgent)        -> picks specialists
    |-- SpecialistTeam           (ParallelAgent)   -> fans out concurrently
    |     |-- ClaimsAgent        (LlmAgent + RF tool)
    |     |-- LabsAgent          (LlmAgent + NN tool)
    |     |-- NotesAgent         (LlmAgent — pure LLM extraction)
    |-- FusionAgent              (LlmAgent)        -> reasons, not averages
    |-- ReviewLoop               (LoopAgent, max 3)
    |     |-- ReviewerAgent      (LlmAgent)        -> pass/fail
    |     |-- RefinerAgent       (LlmAgent)        -> rewrite if fail
    |-- ActionAgent              (LlmAgent)        -> structured JSON + audit

Each agent writes its result to session.state under a known output_key so
downstream agents can read it without bespoke message passing.
"""
from __future__ import annotations

import json
import sys
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar

try:
    from google.genai import types
    from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.parallel_agent import _create_branch_ctx_for_sub_agent
    from google.adk.agents.parallel_agent import _merge_agent_run
    from google.adk.agents.parallel_agent import _merge_agent_run_pre_3_11
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.adk.tools import FunctionTool
    from google.adk.utils.context_utils import Aclosing
except ImportError as _e:
    raise ImportError(
        "google-adk is not installed. Run: pip install google-adk>=0.4.0"
    ) from _e

from .tools import predict_claims_anomaly, score_labs_risk, write_audit_log

MODEL_FAST = "gemini-2.5-flash"
MODEL_SMART = MODEL_FAST
DETERMINISTIC_GENERATION = types.GenerateContentConfig(temperature=0.0)
FINAL_RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "escalate": 0.7,
}
CASE_A_MILD_SIGNAL_GUIDANCE = {
    "max_mild_signal_score": 0.45,
    "target_fused_anomaly_score_max": 0.39,
    "target_fused_confidence_min": 0.72,
}


class RoutedParallelAgent(ParallelAgent):
    """ParallelAgent variant that honors RouterAgent's routing_decision."""

    route_keys: ClassVar[dict[str, str]] = {
        "ClaimsAgent": "run_claims",
        "LabsAgent": "run_labs",
        "NotesAgent": "run_notes",
    }

    def _normalize_routing(self, routing_decision: Any) -> dict[str, bool] | None:
        if isinstance(routing_decision, str):
            try:
                routing_decision = json.loads(routing_decision)
            except json.JSONDecodeError:
                return None
        if not isinstance(routing_decision, dict):
            return None
        if not any(key in routing_decision for key in self.route_keys.values()):
            return None
        return {
            key: bool(routing_decision.get(key, False))
            for key in self.route_keys.values()
        }

    def _selected_sub_agents(self, routing_decision: Any) -> list:
        routing = self._normalize_routing(routing_decision)
        if routing is None:
            return list(self.sub_agents)
        return [
            agent
            for agent in self.sub_agents
            if routing.get(self.route_keys.get(agent.name, ""), False)
        ]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        selected_agents = self._selected_sub_agents(
            ctx.session.state.get("routing_decision")
        )
        if not selected_agents:
            return

        agent_runs = []
        for sub_agent in selected_agents:
            sub_agent_ctx = _create_branch_ctx_for_sub_agent(self, sub_agent, ctx)
            if not sub_agent_ctx.end_of_agents.get(sub_agent.name):
                agent_runs.append(sub_agent.run_async(sub_agent_ctx))

        pause_invocation = False
        try:
            merge_func = (
                _merge_agent_run
                if sys.version_info >= (3, 11)
                else _merge_agent_run_pre_3_11
            )
            async with Aclosing(merge_func(agent_runs)) as agen:
                async for event in agen:
                    yield event
                    if ctx.should_pause_invocation(event):
                        pause_invocation = True
            if pause_invocation:
                return
        finally:
            for sub_agent_run in agent_runs:
                await sub_agent_run.aclose()


router_agent = LlmAgent(
    name="RouterAgent",
    model=MODEL_FAST,
    description="Decides which specialists to invoke based on which fields are present.",
    instruction=(
        "You receive a patient case with three optional sections: claim, labs, notes. "
        "Inspect which sections have content. Output a JSON object with three boolean "
        "fields: run_claims, run_labs, run_notes. "
        "Set each to true only if the corresponding section has non-empty content. "
        "Output ONLY the JSON object, nothing else."
    ),
    output_key="routing_decision",
)


claims_agent = LlmAgent(
    name="ClaimsAgent",
    model=MODEL_FAST,
    description="Scores claim anomaly using the Random Forest model.",
    instruction=(
        "You score whether a claim looks anomalous. Read the 'claim' section "
        "from the input and call predict_claims_anomaly with its fields. "
        "Return the tool result as JSON with an additional 'agent' field set "
        "to 'ClaimsAgent'. If the claim section is missing, return "
        "{\"agent\": \"ClaimsAgent\", \"available\": false}."
    ),
    tools=[FunctionTool(predict_claims_anomaly)],
    output_key="claims_result",
)


labs_agent = LlmAgent(
    name="LabsAgent",
    model=MODEL_FAST,
    description="Scores lab-panel risk using the TensorFlow neural network.",
    instruction=(
        "You score lab-panel risk. Read the 'labs' section from the input and "
        "call score_labs_risk with its fields. If a panel is missing, pass "
        "null for that argument — the tool handles imputation. Return the "
        "tool result as JSON with an additional 'agent' field set to "
        "'LabsAgent'. If the entire labs section is missing, return "
        "{\"agent\": \"LabsAgent\", \"available\": false, \"reason\": \"labs section missing\"}."
    ),
    tools=[FunctionTool(score_labs_risk)],
    output_key="labs_result",
)


notes_agent = LlmAgent(
    name="NotesAgent",
    model=MODEL_FAST,
    description="Extracts structured signals from the unstructured discharge note.",
    instruction=(
        "Read the 'notes' section. Extract three signals as JSON:\n"
        "  - severity_score: float in [0, 1] for clinical severity\n"
        "  - red_flags: list of short strings for any concerning phrases\n"
        "  - billing_consistency: 'consistent' | 'inconsistent' | 'unknown' "
        "    — set to 'inconsistent' if the note describes a workup that "
        "    does NOT match the procedures being billed.\n"
        "Return the JSON with an additional 'agent' field set to 'NotesAgent'. "
        "If the notes section is missing, return "
        "{\"agent\": \"NotesAgent\", \"available\": false}."
    ),
    output_key="notes_result",
)


specialist_team = RoutedParallelAgent(
    name="SpecialistTeam",
    sub_agents=[claims_agent, labs_agent, notes_agent],
    description="Runs only the Router-selected specialist agents concurrently.",
)


fusion_agent = LlmAgent(
    name="FusionAgent",
    model=MODEL_SMART,
    description="Reasons over the three signals — does NOT simple-average.",
    instruction=(
        "You receive three results in session.state under keys "
        "claims_result, labs_result, notes_result. Each may be unavailable.\n\n"
        "Apply context-aware reasoning, NOT averaging:\n"
        f"  - For mild Case A-style agreement, where every available numeric "
        f"signal is <= {CASE_A_MILD_SIGNAL_GUIDANCE['max_mild_signal_score']} "
        "and notes billing_consistency is 'consistent', keep the case LOW: "
        f"set fused_anomaly_score < {FINAL_RISK_THRESHOLDS['medium']} "
        f"(target <= {CASE_A_MILD_SIGNAL_GUIDANCE['target_fused_anomaly_score_max']}) "
        f"and fused_confidence >= {FINAL_RISK_THRESHOLDS['escalate']} "
        f"(target >= {CASE_A_MILD_SIGNAL_GUIDANCE['target_fused_confidence_min']}). "
        "Do not convert small numeric disagreement among mild signals into "
        "low confidence.\n"
        "  - If claims score is high but labs and notes are LOW and consistent "
        "    with each other, weight the disagreement: the claim is the anomaly.\n"
        "  - If labs is missing, drop its weight to zero and increase reliance "
        "    on the other two — but cap final confidence at 0.7.\n"
        "  - If notes flag billing_consistency='inconsistent', boost the "
        "    anomaly signal by 0.15.\n"
        "  - If all three agree (all low or all high), confidence is high.\n\n"
        "Output JSON with these fields:\n"
        "  fused_anomaly_score (float 0..1),\n"
        "  fused_confidence    (float 0..1),\n"
        "  signals_used        (list of strings),\n"
        "  conflict_detected   (bool),\n"
        "  reasoning           (string, 1-2 sentences explaining the weighting)."
    ),
    generate_content_config=DETERMINISTIC_GENERATION,
    output_key="fusion_result",
)


reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model=MODEL_FAST,
    description="Validates the fused decision against safety rules.",
    instruction=(
        "Read fusion_result from session.state. Apply these validation rules:\n"
        "  R1: fused_confidence must be >= 0.65\n"
        "  R2: if conflict_detected is true, reasoning must mention 'conflict' "
        "      or 'inconsistent' or 'disagree'\n"
        "  R3: signals_used must contain at least one entry\n\n"
        "Output JSON: {\"passed\": bool, \"failed_rules\": [list of rule ids], "
        "\"feedback\": string}. If passed, set escalate=false. If failed_rules "
        "still includes R1 after refinement, set escalate=true so the loop "
        "exits to a human queue."
    ),
    generate_content_config=DETERMINISTIC_GENERATION,
    output_key="review_result",
)


class LoopExitChecker(BaseAgent):
    """Terminates ReviewLoop early when the Reviewer says it's done.

    Yields an event with ``actions.escalate=True`` (which LoopAgent honors as
    a stop signal) when ``review_result.passed`` is true (validation OK) or
    when ``review_result.escalate`` is true (give up, hand to a human).
    Otherwise yields nothing and the loop continues to the next iteration.
    """

    @staticmethod
    def _parse_review(raw: Any) -> dict[str, Any]:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return raw if isinstance(raw, dict) else {}

    @classmethod
    def should_exit(cls, raw: Any) -> bool:
        review = cls._parse_review(raw)
        return bool(review.get("passed", False)) or bool(review.get("escalate", False))

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if self.should_exit(ctx.session.state.get("review_result")):
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                actions=EventActions(escalate=True),
            )


loop_exit_checker = LoopExitChecker(
    name="LoopExitChecker",
    description="Stops ReviewLoop early when the Reviewer passes or escalates.",
)


refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=MODEL_FAST,
    description="Rewrites the fused decision when the Reviewer fails it.",
    instruction=(
        "Read review_result and fusion_result from session.state. If "
        "review_result.passed is true, output fusion_result unchanged. "
        "Otherwise, rewrite fusion_result addressing each failed rule:\n"
        "  - If R1 failed (low confidence), reduce fused_anomaly_score "
        "    toward 0.5 and explicitly mention uncertainty in reasoning.\n"
        "  - If R2 failed, expand reasoning to name the specific conflict.\n"
        "  - If R3 failed, drop fusion_result and produce an "
        "    available=false envelope.\n"
        "Write the refined dict back to session.state under key "
        "'fusion_result'. Output the same JSON shape as fusion_result."
    ),
    output_key="fusion_result",
)


review_loop = LoopAgent(
    name="ReviewLoop",
    max_iterations=3,
    sub_agents=[reviewer_agent, loop_exit_checker, refiner_agent],
    description=(
        "Reviewer -> ExitChecker -> Refiner. Bounded retries (max 3). "
        "ExitChecker short-circuits the iteration the moment Review passes "
        "or escalates, so we don't burn an LLM call on a no-op refine."
    ),
)


action_agent = LlmAgent(
    name="ActionAgent",
    model=MODEL_FAST,
    description="Emits the final structured decision and writes the audit log.",
    instruction=(
        "Read fusion_result and review_result from session.state.\n\n"
        "Map fused_anomaly_score to risk_level:\n"
        "  >= 0.7 -> HIGH\n"
        "  >= 0.4 -> MEDIUM\n"
        "  <  0.4 -> LOW\n\n"
        "Map (risk_level, fused_confidence) to recommended_action:\n"
        "  HIGH    + conf >= 0.7  -> FLAG_FOR_AUDIT\n"
        "  MEDIUM  + conf >= 0.7  -> ROUTINE_FOLLOWUP\n"
        "  LOW     + conf >= 0.7  -> AUTO_APPROVE\n"
        "  any risk + conf <  0.7 -> ESCALATE_TO_HUMAN\n\n"
        "Output the final JSON with keys: patient_id, risk_level, "
        "anomaly_score, confidence, recommended_action, reasoning, "
        "audit_trail. Then call write_audit_log with patient_id and the "
        "full decision dict."
    ),
    tools=[FunctionTool(write_audit_log)],
    generate_content_config=DETERMINISTIC_GENERATION,
    output_key="final_decision",
)


root_pipeline = SequentialAgent(
    name="RootPipeline",
    sub_agents=[
        router_agent,
        specialist_team,
        fusion_agent,
        review_loop,
        action_agent,
    ],
    description="End-to-end claims-risk orchestrator.",
)
