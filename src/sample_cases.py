"""
Three canonical patient cases that exercise different paths in the orchestrator.

  CASE_A — Mary, 67  : all 3 sources, mild signals, expect AUTO_APPROVE
  CASE_B — Robert, 54: all 3 sources but conflicting signals, expect FLAG_FOR_AUDIT
  CASE_C — Linda, 72 : labs missing, expect ESCALATE_TO_HUMAN

Used by the CLI demo, the Streamlit dashboard, and the tests.
"""
from __future__ import annotations


CASE_A = {
    "patient_id": "P_MARY",
    "claim": {
        "cost_usd": 8420.0,
        "procedure_count": 2,
        "los_days": 3,
        "age": 67,
        "drg_code": "470",
    },
    "labs": {"a1c": 7.8, "ldl": 142, "egfr": 64, "troponin": 0.02},
    "notes": "Patient stable post-op knee replacement. Adherent to metformin. Daughter assists with care.",
}

CASE_B = {
    "patient_id": "P_ROBERT",
    "claim": {
        "cost_usd": 24180.0,
        "procedure_count": 6,
        "los_days": 0,
        "age": 54,
        "drg_code": "292",
    },
    "labs": {"a1c": 5.4, "ldl": 110, "egfr": 88, "troponin": 0.01},
    "notes": (
        "Chest pain ruled out as non-cardiac (likely GERD). Troponin and EKG "
        "within normal limits. Patient discharged same day. Findings inconsistent "
        "with billed procedures."
    ),
}

CASE_C = {
    "patient_id": "P_LINDA",
    "claim": {
        "cost_usd": 11300.0,
        "procedure_count": 3,
        "los_days": 4,
        "age": 72,
        "drg_code": "291",
    },
    "labs": None,
    "notes": (
        "Patient with prior CHF admission 18 days ago. Concerned for medication "
        "non-adherence. Limited social support."
    ),
}

CASES = {"A": CASE_A, "B": CASE_B, "C": CASE_C}
