"""
Generate synthetic-but-realistic healthcare data for the POC.

We synthesize three data varieties matching the shape of the real public
datasets the production version would use:

  - claims.csv    -> shaped like CMS DE-SynPUF (cost + utilization features)
  - labs.csv      -> shaped like NHANES lab panels (numeric panels)
  - notes.csv     -> shaped like MIMIC-IV discharge notes (free text)

For the POC we generate them locally so you can run end-to-end without
downloading the real datasets. Once the POC is green, swap these for real
extracts and the rest of the code does not change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

NOTE_TEMPLATES_LOW = [
    "Patient stable post-operative. Adherent to medications. Family supportive.",
    "Routine follow-up. No acute concerns. Vitals within normal limits.",
    "Mild symptoms resolving. Continue current regimen. Recheck in 30 days.",
]
NOTE_TEMPLATES_MED = [
    "Patient with prior admission 18 days ago. Concern for medication non-adherence.",
    "Reports persistent symptoms despite treatment. Social support limited.",
    "Multiple comorbidities. Coordination of care needed.",
]
NOTE_TEMPLATES_HIGH = [
    "Chest pain ruled out as non-cardiac (likely GERD). Patient discharged same day.",
    "Workup completed. Findings inconsistent with billed procedures. Awaiting review.",
    "Multiple high-cost procedures performed. Clinical justification unclear from chart.",
]


def make_claims(n: int) -> pd.DataFrame:
    """CMS DE-SynPUF-style claim rows."""
    patient_ids = [f"P{i:05d}" for i in range(n)]
    cost = RNG.lognormal(mean=8.5, sigma=1.0, size=n).round(2)
    procedure_count = RNG.integers(1, 8, size=n)
    los_days = RNG.integers(0, 14, size=n)
    drg = RNG.choice(["470", "291", "292", "871", "292"], size=n)
    age = RNG.integers(45, 90, size=n)

    high_cost_threshold = np.quantile(cost, 0.85)
    anomaly = ((cost > high_cost_threshold) & (procedure_count >= 5)).astype(int)
    noise = RNG.binomial(1, 0.01, size=n)
    is_anomaly = ((anomaly + noise) > 0).astype(int)

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "cost_usd": cost,
            "procedure_count": procedure_count,
            "los_days": los_days,
            "drg_code": drg,
            "age": age,
            "is_anomaly": is_anomaly,
        }
    )


def make_labs(n: int, patient_ids: list[str]) -> pd.DataFrame:
    """NHANES-style lab panel rows. ~10% of patients have a missing panel."""
    a1c = RNG.normal(loc=6.5, scale=1.5, size=n).round(1)
    ldl = RNG.normal(loc=130, scale=35, size=n).round(0)
    egfr = RNG.normal(loc=70, scale=20, size=n).round(0)
    troponin = RNG.exponential(scale=0.05, size=n).round(3)

    at_risk = (
        ((a1c > 8.5) | (ldl > 190) | (egfr < 45) | (troponin > 0.4)).astype(int)
    )

    df = pd.DataFrame(
        {
            "patient_id": patient_ids[:n],
            "a1c": a1c,
            "ldl": ldl,
            "egfr": egfr,
            "troponin": troponin,
            "at_risk": at_risk,
        }
    )
    drop_idx = RNG.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df = df.drop(index=drop_idx).reset_index(drop=True)
    return df


def make_notes(n: int, patient_ids: list[str]) -> pd.DataFrame:
    """MIMIC-style free-text discharge notes. Bucketed by severity."""
    rows = []
    for pid in patient_ids[:n]:
        bucket = RNG.choice(["low", "med", "high"], p=[0.6, 0.25, 0.15])
        if bucket == "low":
            text = RNG.choice(NOTE_TEMPLATES_LOW)
        elif bucket == "med":
            text = RNG.choice(NOTE_TEMPLATES_MED)
        else:
            text = RNG.choice(NOTE_TEMPLATES_HIGH)
        rows.append({"patient_id": pid, "discharge_note": text, "true_severity": bucket})
    return pd.DataFrame(rows)


def main(n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = make_claims(n)
    pids = claims["patient_id"].tolist()
    labs = make_labs(n, pids)
    notes = make_notes(n, pids)

    claims.to_csv(out_dir / "claims.csv", index=False)
    labs.to_csv(out_dir / "labs.csv", index=False)
    notes.to_csv(out_dir / "notes.csv", index=False)

    print(f"wrote {len(claims)} claims, {len(labs)} labs, {len(notes)} notes to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()
    main(args.n, args.out)
