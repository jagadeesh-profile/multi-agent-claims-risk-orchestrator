"""Map verified public raw datasets into this project's training schemas.

This is a lightweight ETL bridge for public samples:
  - CMS DE-SynPUF inpatient claims ZIP -> claims.csv shape
  - NHANES GHB/HDL XPT files          -> labs.csv shape
  - MIMIC-IV-ED Demo ZIP              -> notes.csv shape

The mapped files are useful for schema validation and demo retraining. They
are not a medically validated benchmark; labels are weak demo labels derived
from available public fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


CLAIMS_COLUMNS = [
    "patient_id",
    "cost_usd",
    "procedure_count",
    "los_days",
    "drg_code",
    "age",
    "is_anomaly",
]
LAB_COLUMNS = ["patient_id", "a1c", "ldl", "egfr", "troponin", "at_risk"]
NOTE_COLUMNS = ["patient_id", "discharge_note", "true_severity"]


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def map_cms_inpatient_claims(raw: pd.DataFrame) -> pd.DataFrame:
    """Map CMS DE-SynPUF inpatient rows to the claims training schema."""
    hcpcs_cols = [c for c in raw.columns if c.startswith("HCPCS_CD_")]
    if hcpcs_cols:
        procedure_count = raw[hcpcs_cols].notna().sum(axis=1).clip(lower=1)
    else:
        procedure_count = pd.Series(1, index=raw.index)

    from_dt = _parse_yyyymmdd(raw["CLM_FROM_DT"])
    thru_dt = _parse_yyyymmdd(raw["CLM_THRU_DT"])
    los_days = (thru_dt - from_dt).dt.days.fillna(0).clip(lower=0).astype(int)
    cost = pd.to_numeric(raw["CLM_PMT_AMT"], errors="coerce").fillna(0.0)

    mapped = pd.DataFrame(
        {
            "patient_id": raw["DESYNPUF_ID"].astype(str),
            "cost_usd": cost,
            "procedure_count": procedure_count.astype(int),
            "los_days": los_days,
            "drg_code": raw.get("CLM_DRG_CD", "UNKNOWN").astype(str).fillna("UNKNOWN"),
            "age": 65,
        }
    )
    high_cost = mapped["cost_usd"] >= mapped["cost_usd"].quantile(0.85)
    high_utilization = mapped["procedure_count"] >= mapped["procedure_count"].quantile(0.75)
    mapped["is_anomaly"] = (high_cost & high_utilization).astype(int)
    return mapped[CLAIMS_COLUMNS]


def map_nhanes_labs(ghb: pd.DataFrame, hdl: pd.DataFrame) -> pd.DataFrame:
    """Map NHANES glycohemoglobin + HDL files to the labs training schema."""
    joined = ghb[["SEQN", "LBXGH"]].merge(hdl[["SEQN", "LBDHDD"]], on="SEQN", how="inner")
    a1c = pd.to_numeric(joined["LBXGH"], errors="coerce")
    hdl_value = pd.to_numeric(joined["LBDHDD"], errors="coerce")

    # NHANES HDL is not LDL, but gives a useful public-lab signal for the demo
    # bridge. Estimate LDL inversely so the existing NN schema can be exercised.
    estimated_ldl = (190 - hdl_value).clip(lower=70, upper=220)
    mapped = pd.DataFrame(
        {
            "patient_id": "NHANES_" + joined["SEQN"].astype(int).astype(str),
            "a1c": a1c,
            "ldl": estimated_ldl,
            "egfr": 70.0,
            "troponin": 0.02,
        }
    )
    mapped["at_risk"] = (
        (mapped["a1c"] >= 8.5)
        | (mapped["ldl"] >= 160)
        | (mapped["egfr"] < 45)
        | (mapped["troponin"] > 0.4)
    ).astype(int)
    return mapped.dropna(subset=["a1c", "ldl"])[LAB_COLUMNS].reset_index(drop=True)


def map_mimic_ed_notes(diagnosis: pd.DataFrame, triage: pd.DataFrame) -> pd.DataFrame:
    """Map MIMIC-IV-ED demo tables to simple note rows."""
    diag_titles = (
        diagnosis.assign(icd_title=diagnosis["icd_title"].fillna(""))
        .groupby(["subject_id", "stay_id"])["icd_title"]
        .apply(lambda vals: "; ".join(str(v) for v in vals if str(v)))
        .reset_index()
    )
    joined = triage.merge(diag_titles, on=["subject_id", "stay_id"], how="left")
    acuity = pd.to_numeric(joined.get("acuity"), errors="coerce")
    severity = pd.Series("low", index=joined.index)
    severity = severity.mask(acuity <= 3, "med")
    severity = severity.mask(acuity <= 2, "high")

    complaint = joined.get("chiefcomplaint", "").fillna("").astype(str)
    diagnoses = joined.get("icd_title", "").fillna("").astype(str)
    notes = (
        "ED triage complaint: "
        + complaint
        + ". Diagnoses recorded: "
        + diagnoses
        + "."
    )
    return pd.DataFrame(
        {
            "patient_id": "MIMIC_"
            + joined["subject_id"].astype(int).astype(str)
            + "_"
            + joined["stay_id"].astype(int).astype(str),
            "discharge_note": notes,
            "true_severity": severity,
        }
    )[NOTE_COLUMNS]


def _read_first_csv_from_zip(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV file found inside {path}")
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f)


def _read_mimic_table(path: Path, table_name: str) -> pd.DataFrame:
    suffix = f"/{table_name}.csv.gz"
    with ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if not matches:
            raise FileNotFoundError(f"No {table_name}.csv.gz found inside {path}")
        with zf.open(matches[0]) as f:
            return pd.read_csv(f, compression="gzip")


def prepare_public_data(
    cms_inpatient_zip: Path,
    nhanes_ghb_xpt: Path,
    nhanes_hdl_xpt: Path,
    mimic_ed_demo_zip: Path,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    claims = map_cms_inpatient_claims(_read_first_csv_from_zip(cms_inpatient_zip))
    labs = map_nhanes_labs(
        pd.read_sas(nhanes_ghb_xpt, format="xport"),
        pd.read_sas(nhanes_hdl_xpt, format="xport"),
    )
    notes = map_mimic_ed_notes(
        _read_mimic_table(mimic_ed_demo_zip, "diagnosis"),
        _read_mimic_table(mimic_ed_demo_zip, "triage"),
    )

    outputs = {
        "claims": out_dir / "claims.csv",
        "labs": out_dir / "labs.csv",
        "notes": out_dir / "notes.csv",
    }
    claims.to_csv(outputs["claims"], index=False)
    labs.to_csv(outputs["labs"], index=False)
    notes.to_csv(outputs["notes"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public raw data samples.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("tmp/raw_public"),
        help="Directory containing downloaded public raw data samples.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data_public"),
        help="Output directory for mapped claims.csv, labs.csv, notes.csv.",
    )
    args = parser.parse_args()
    outputs = prepare_public_data(
        cms_inpatient_zip=args.raw_dir / "cms_de_synpuf_inpatient_sample2.zip",
        nhanes_ghb_xpt=args.raw_dir / "nhanes_ghb_j.xpt",
        nhanes_hdl_xpt=args.raw_dir / "nhanes_hdl_j.xpt",
        mimic_ed_demo_zip=args.raw_dir / "mimic_iv_ed_demo_2_2.zip",
        out_dir=args.out,
    )
    for name, path in outputs.items():
        rows = len(pd.read_csv(path))
        print(f"wrote {rows} {name} rows -> {path}")


if __name__ == "__main__":
    main()
