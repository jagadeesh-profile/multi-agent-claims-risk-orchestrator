"""
Multi-run evaluation harness for the Claims-Risk Orchestrator.

Runs each sample case ``--runs`` times, then reports:
  - decision_agreement_rate : how often the same recommended_action repeats
  - latency p50 / p95       : pipeline wall-clock per case
  - per-agent latency       : mean ms per agent step
  - failure_rate            : runs that returned an error envelope

Outputs:
  outputs/evaluation/results.json (machine-readable)
  outputs/evaluation/results.md   (human-readable summary you can paste anywhere)

Usage:
  python -m eval.run_eval --runs 5
  python -m eval.run_eval --runs 3 --cases A,B
  python -m eval.run_eval --runs 10 --out outputs/evaluation/baseline.json

NOTE: This script consumes Gemini API quota. Each case is ~12 LLM calls.
Default --runs=3 means ~36 calls per case = ~108 calls total. Stay aware
of free-tier limits (https://ai.google.dev/gemini-api/docs/rate-limits).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.orchestrator import run_case
from src.sample_cases import CASES

load_dotenv()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


async def evaluate_case(case_id: str, case: dict[str, Any], runs: int) -> dict[str, Any]:
    print(f"\n=== Case {case_id}: {case['patient_id']} (runs={runs}) ===", flush=True)

    actions: list[str] = []
    risk_levels: list[str] = []
    latencies: list[float] = []
    per_agent_acc: dict[str, list[float]] = {}
    failures = 0
    errors: list[str] = []

    for i in range(1, runs + 1):
        t0 = time.time()
        try:
            result = await run_case(case, verbose=False)
        except Exception as exc:  # noqa: BLE001 - eval is supposed to keep going.
            failures += 1
            errors.append(f"run {i}: {type(exc).__name__}: {exc}")
            print(f"  [{i}/{runs}] FAILED: {exc}", flush=True)
            continue

        decision = result.get("decision", {})
        action = decision.get("recommended_action", "UNKNOWN")
        risk = decision.get("risk_level", "UNKNOWN")
        if "error" in decision:
            failures += 1
            errors.append(f"run {i}: parse_error -> {decision.get('raw', '')[:120]}")
            action = f"PARSE_ERROR/{action}"

        actions.append(action)
        risk_levels.append(risk)
        latencies.append(result.get("timing_sec", round(time.time() - t0, 2)))

        for agent, ms in result.get("per_agent_ms", {}).items():
            per_agent_acc.setdefault(agent, []).append(float(ms))

        print(
            f"  [{i}/{runs}] action={action} risk={risk} "
            f"latency={result.get('timing_sec'):.1f}s",
            flush=True,
        )

    action_counts = Counter(actions)
    most_common_action, most_common_count = (
        action_counts.most_common(1)[0] if action_counts else ("NONE", 0)
    )
    agreement = round(most_common_count / runs, 3) if runs else 0.0

    return {
        "case_id": case_id,
        "patient_id": case["patient_id"],
        "runs": runs,
        "successful_runs": len(actions),
        "failure_rate": round(failures / runs, 3) if runs else 0.0,
        "decision_agreement_rate": agreement,
        "modal_action": most_common_action,
        "action_distribution": dict(action_counts),
        "risk_distribution": dict(Counter(risk_levels)),
        "latency": {
            "p50_sec": round(_percentile(latencies, 0.50), 2),
            "p95_sec": round(_percentile(latencies, 0.95), 2),
            "mean_sec": round(statistics.mean(latencies), 2) if latencies else 0.0,
        },
        "per_agent_mean_ms": {
            a: round(statistics.mean(v), 1) for a, v in per_agent_acc.items()
        },
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Claims-Risk Orchestrator — Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Runs per case: {report['runs_per_case']}",
        f"- Total LLM-pipeline invocations: {report['total_invocations']}",
        "",
        "## Per-case summary",
        "",
        "| Case | Modal action | Agreement | p50 latency | p95 latency | Failures |",
        "|------|--------------|-----------|-------------|-------------|----------|",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['case_id']} ({r['patient_id']}) "
            f"| `{r['modal_action']}` "
            f"| {int(r['decision_agreement_rate'] * 100)}% "
            f"| {r['latency']['p50_sec']}s "
            f"| {r['latency']['p95_sec']}s "
            f"| {int(r['failure_rate'] * 100)}% |"
        )
    lines.append("")
    lines.append("## Per-agent mean latency (ms)")
    lines.append("")
    agents_seen: set[str] = set()
    for r in report["cases"]:
        agents_seen.update(r["per_agent_mean_ms"])
    if agents_seen:
        header = ["Case"] + sorted(agents_seen)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for r in report["cases"]:
            row = [r["case_id"]] + [
                f"{r['per_agent_mean_ms'].get(a, 0):.0f}" for a in sorted(agents_seen)
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


async def amain(case_ids: list[str], runs: int, out_path: Path) -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "ERROR: GOOGLE_API_KEY is not set. Add it to .env before running the eval.",
            file=sys.stderr,
        )
        return 2

    cases_to_run = [(cid, CASES[cid]) for cid in case_ids if cid in CASES]
    unknown = [cid for cid in case_ids if cid not in CASES]
    if unknown:
        print(f"WARNING: skipping unknown cases: {unknown}", file=sys.stderr)
    if not cases_to_run:
        print("ERROR: no valid cases to run.", file=sys.stderr)
        return 2

    started = time.time()
    case_reports: list[dict[str, Any]] = []
    for case_id, case in cases_to_run:
        case_reports.append(await evaluate_case(case_id, case, runs))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_per_case": runs,
        "total_invocations": runs * len(cases_to_run),
        "wall_clock_sec": round(time.time() - started, 2),
        "cases": case_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(report))

    print("\n--- SUMMARY ---")
    print(json.dumps({c["case_id"]: {
        "modal_action": c["modal_action"],
        "agreement": c["decision_agreement_rate"],
        "p50_sec": c["latency"]["p50_sec"],
    } for c in case_reports}, indent=2))
    print(f"\nFull report -> {out_path}")
    print(f"Markdown    -> {md_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-run evaluation harness")
    p.add_argument("--runs", type=int, default=3, help="Number of runs per case")
    p.add_argument(
        "--cases",
        type=str,
        default="A,B,C",
        help="Comma-separated case IDs (e.g. 'A,B' or 'A,B,C')",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/evaluation/results.json"),
        help="Output JSON path (markdown sibling is also written)",
    )
    args = p.parse_args()
    case_ids = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
    rc = asyncio.run(amain(case_ids, args.runs, args.out))
    sys.exit(rc)


if __name__ == "__main__":
    main()
