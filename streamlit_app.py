"""
Streamlit dashboard for the Claims-Risk Orchestrator.

Pick a sample case (or paste your own JSON), hit Run, watch the agents
fire and the final structured decision render with risk-level color coding.

Usage:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.orchestrator import run_case
from src.sample_cases import CASES

load_dotenv()

st.set_page_config(page_title="Claims-Risk Orchestrator", layout="wide")
st.title("Claims-Risk Orchestrator")
st.caption("Multi-agent healthcare claims validation on Vertex AI ADK")

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not set. Add it to your .env file.")
    st.stop()

if not Path("models/claims_rf.joblib").exists() or not Path("models/labs_nn.keras").exists():
    st.error("Models not trained yet. Run: python -m src.generate_data && python -m src.train_claims_rf && python -m src.train_labs_nn")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Feed input")
    input_mode = st.radio(
        "Input source",
        options=["Sample case", "Custom JSON"],
        horizontal=True,
    )

    if input_mode == "Sample case":
        case_choice = st.radio(
            "Sample cases",
            options=list(CASES.keys()),
            format_func=lambda k: {
                "A": "Case A — Mary, 67 (routine)",
                "B": "Case B — Robert, 54 (possible fraud)",
                "C": "Case C — Linda, 72 (missing labs)",
            }[k],
            horizontal=False,
        )
        case = CASES[case_choice]
        st.json(case, expanded=False)
    else:
        raw_case = st.text_area(
            "Patient case JSON",
            value=json.dumps(CASES["A"], indent=2),
            height=360,
        )
        try:
            case = json.loads(raw_case)
            if not isinstance(case, dict) or not case.get("patient_id"):
                raise ValueError("Case JSON must be an object with patient_id.")
            st.success(f"Ready to run patient `{case['patient_id']}`")
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid input JSON: {exc}")
            st.stop()

    run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

with right:
    st.subheader("2. Final decision")
    decision_slot = st.empty()

    st.subheader("3. Agent trace")
    trace_slot = st.container()

if run_clicked:
    with st.spinner("Orchestrator running..."):
        result = asyncio.run(run_case(case, verbose=False))

    decision = result["decision"]
    risk_level = decision.get("risk_level", "UNKNOWN")
    risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}.get(risk_level, "gray")

    with decision_slot.container():
        st.markdown(f"**Risk level:** :{risk_color}[**{risk_level}**]")
        st.markdown(f"**Action:** `{decision.get('recommended_action', '?')}`")
        st.caption("Audit records are appended to `logs/audit.jsonl` when the live ActionAgent writes the audit log.")
        col1, col2 = st.columns(2)
        col1.metric("Anomaly score", f"{decision.get('anomaly_score', 0):.2f}")
        col2.metric("Confidence", f"{decision.get('confidence', 0):.2f}")
        with st.expander("Full JSON", expanded=False):
            st.json(decision)

    with trace_slot:
        for i, step in enumerate(result["trace"], 1):
            with st.expander(f"{i}. {step['agent']}", expanded=False):
                try:
                    st.json(json.loads(step["output"]))
                except (json.JSONDecodeError, TypeError):
                    st.code(step["output"], language="text")
        st.caption(f"Pipeline completed in {result['timing_sec']}s — {len(result['trace'])} agent events")
