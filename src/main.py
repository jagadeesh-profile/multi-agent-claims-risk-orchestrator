"""
CLI entry point for the Claims-Risk Orchestrator.

Usage:
  python -m src.main --case A          # run sample case A
  python -m src.main --case B -v       # verbose: print every agent step
  python -m src.main --case C          # run case C (missing labs)

Requires GOOGLE_API_KEY set in env (or .env). Models must be trained first:
  python -m src.generate_data
  python -m src.train_claims_rf
  python -m src.train_labs_nn
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

from .orchestrator import run_case
from .sample_cases import CASES

load_dotenv()


def format_runtime_error(exc: Exception) -> str:
    """Convert common live API failures into concise CLI guidance."""
    messages = [str(exc)]
    if isinstance(exc, BaseExceptionGroup):
        messages.extend(str(child) for child in exc.exceptions)
    message = " | ".join(messages)
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return (
            "ERROR: Gemini quota is exhausted. For the free tier, wait for quota reset "
            "or run fewer live sample cases."
        )
    if "UNAVAILABLE" in message or "503" in message:
        return (
            "ERROR: Gemini service is temporarily unavailable or under high demand; "
            "try again later."
        )
    return f"ERROR: Pipeline failed: {message}"


def check_environment() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. Copy .env.example to .env and fill in the key.", file=sys.stderr)
        sys.exit(1)
    for needed in ["models/claims_rf.joblib", "models/labs_nn.keras"]:
        if not Path(needed).exists():
            print(f"ERROR: {needed} not found. Train the models first:", file=sys.stderr)
            print("  python -m src.generate_data", file=sys.stderr)
            print("  python -m src.train_claims_rf", file=sys.stderr)
            print("  python -m src.train_labs_nn", file=sys.stderr)
            sys.exit(1)


def load_case_from_path(path: Path) -> dict[str, Any]:
    """Load one patient case JSON file for live orchestration."""
    with path.open(encoding="utf-8") as f:
        case = json.load(f)
    if not isinstance(case, dict):
        raise ValueError("Input case JSON must be an object.")
    if not case.get("patient_id"):
        raise ValueError("Input case JSON must include patient_id.")
    return case


def save_decision(patient_id: str, decision: dict[str, Any], out_dir: Path) -> Path:
    """Persist one final decision JSON file and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_patient_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in patient_id
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{safe_patient_id}_{stamp}.json"
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return path


def resolve_case(case_id: str, input_path: Path | None) -> dict[str, Any]:
    if input_path is not None:
        return load_case_from_path(input_path)

    if case_id not in CASES:
        print(f"Unknown case '{case_id}'. Choose from: {list(CASES)}", file=sys.stderr)
        sys.exit(2)
    return CASES[case_id]


async def amain(
    case_id: str,
    verbose: bool,
    input_path: Path | None,
    out_dir: Path,
) -> None:
    case = resolve_case(case_id, input_path)
    print(f"\n{'='*60}")
    label = str(input_path) if input_path else f"case {case_id}"
    print(f"Running {label}: {case['patient_id']}")
    print(f"{'='*60}\n")

    result = await run_case(case, verbose=verbose)
    saved_path = save_decision(
        patient_id=str(case.get("patient_id", "UNKNOWN")),
        decision=result["decision"],
        out_dir=out_dir,
    )

    print("\n--- FINAL DECISION ---")
    print(json.dumps(result["decision"], indent=2))
    print(f"\nElapsed: {result['timing_sec']}s | Trace events: {len(result['trace'])}")
    print(f"Saved decision -> {saved_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Claims-Risk Orchestrator CLI")
    p.add_argument("--case", default="A", help="Case ID: A, B, or C")
    p.add_argument(
        "--input",
        type=Path,
        help="Path to a custom patient case JSON file. Overrides --case.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/decisions"),
        help="Directory where final decision JSON files are saved.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print every agent's output")
    args = p.parse_args()

    check_environment()
    try:
        asyncio.run(amain(args.case, args.verbose, args.input, args.out_dir))
    except Exception as exc:  # noqa: BLE001 - CLI boundary should render actionable errors.
        print(format_runtime_error(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
