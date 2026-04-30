"""Readable lab column names plus legacy alias handling."""
from __future__ import annotations

import pandas as pd

LAB_FEATURES = [
    "hemoglobin_a1c_percent",
    "ldl_cholesterol_mg_dl",
    "estimated_gfr_ml_min_1_73m2",
    "troponin_ng_ml",
]
LAB_TARGET = "lab_risk_flag"

LEGACY_LAB_COLUMN_ALIASES = {
    "a1c": "hemoglobin_a1c_percent",
    "ldl": "ldl_cholesterol_mg_dl",
    "egfr": "estimated_gfr_ml_min_1_73m2",
    "troponin": "troponin_ng_ml",
    "at_risk": "lab_risk_flag",
}


def normalize_labs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with readable lab column names.

    Older generated files used short names like ``a1c`` and ``egfr``. The
    training code accepts those aliases, but new generated/exported CSVs use
    readable names for portfolio reviewers.
    """
    return df.rename(columns=LEGACY_LAB_COLUMN_ALIASES).copy()
