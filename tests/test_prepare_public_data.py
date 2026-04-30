"""Tests for mapping verified public raw data into project schemas."""
from __future__ import annotations

import pandas as pd

from src.prepare_public_data import (
    map_cms_inpatient_claims,
    map_mimic_ed_notes,
    map_nhanes_labs,
)


def test_map_cms_inpatient_claims_outputs_training_schema() -> None:
    raw = pd.DataFrame(
        {
            "DESYNPUF_ID": ["P1", "P2"],
            "CLM_PMT_AMT": [57000.0, 1200.0],
            "CLM_FROM_DT": [20080101, 20080103],
            "CLM_THRU_DT": [20080105, 20080103],
            "CLM_DRG_CD": ["281", "470"],
            "HCPCS_CD_1": ["A", None],
            "HCPCS_CD_2": ["B", None],
        }
    )

    mapped = map_cms_inpatient_claims(raw)

    assert mapped.columns.tolist() == [
        "patient_id",
        "cost_usd",
        "procedure_count",
        "los_days",
        "drg_code",
        "age",
        "is_anomaly",
    ]
    assert mapped.loc[0, "procedure_count"] == 2
    assert mapped.loc[0, "los_days"] == 4
    assert mapped.loc[0, "is_anomaly"] == 1


def test_map_nhanes_labs_outputs_lab_training_schema() -> None:
    ghb = pd.DataFrame({"SEQN": [1, 2], "LBXGH": [9.1, 5.4]})
    hdl = pd.DataFrame({"SEQN": [1, 2], "LBDHDD": [32.0, 61.0]})

    mapped = map_nhanes_labs(ghb, hdl)

    assert mapped.columns.tolist() == [
        "patient_id",
        "hemoglobin_a1c_percent",
        "ldl_cholesterol_mg_dl",
        "estimated_gfr_ml_min_1_73m2",
        "troponin_ng_ml",
        "lab_risk_flag",
    ]
    assert mapped.loc[0, "patient_id"] == "NHANES_1"
    assert mapped.loc[0, "lab_risk_flag"] == 1
    assert mapped.loc[1, "lab_risk_flag"] == 0


def test_map_mimic_ed_notes_outputs_notes_schema() -> None:
    diagnosis = pd.DataFrame(
        {
            "subject_id": [10, 10],
            "stay_id": [100, 100],
            "icd_title": ["Chest pain, unspecified", "Type 2 diabetes mellitus"],
        }
    )
    triage = pd.DataFrame(
        {
            "subject_id": [10],
            "stay_id": [100],
            "acuity": [2],
            "chiefcomplaint": ["CHEST PAIN"],
        }
    )

    mapped = map_mimic_ed_notes(diagnosis, triage)

    assert mapped.columns.tolist() == [
        "patient_id",
        "discharge_note",
        "true_severity",
    ]
    assert mapped.loc[0, "patient_id"] == "MIMIC_10_100"
    assert "CHEST PAIN" in mapped.loc[0, "discharge_note"]
    assert mapped.loc[0, "true_severity"] == "high"
