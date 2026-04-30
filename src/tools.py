"""
ADK tools — Python functions the specialist agents call.

ADK tools are plain Python functions with type hints + a docstring.
The framework introspects the signature and exposes them to the LLM as
callable tools. We wrap the three trained models here.

Each tool:
  - accepts a dict of features
  - returns a dict with score + confidence + a short rationale
  - is deterministic for the same input
  - never raises — failures return {"available": False, "reason": "..."}
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")


@lru_cache(maxsize=1)
def _load_claims_rf():
    path = MODELS_DIR / "claims_rf.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)
    clf = getattr(model, "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "n_jobs"):
        clf.n_jobs = 1
    return model


@lru_cache(maxsize=1)
def _load_shap_explainer():
    """Build a TreeExplainer once. Returns (explainer, feature_names) or None.

    SHAP explanations are computed lazily and only if shap is installed.
    Tool callers always get a score; XAI is additive.
    """
    model = _load_claims_rf()
    if model is None:
        return None
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        return None
    try:
        clf = model.named_steps["clf"]
        pre = model.named_steps["pre"]
        feat_names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(clf)
        return explainer, feat_names, pre
    except Exception:  # noqa: BLE001 - XAI failure must never break the tool.
        return None


def _top_shap_features(row: pd.DataFrame, k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k SHAP-magnitude features for the positive class.

    Empty list if shap isn't installed or anything goes wrong — the caller
    treats this as optional metadata.
    """
    bundle = _load_shap_explainer()
    if bundle is None:
        return []
    explainer, feat_names, pre = bundle
    try:
        x = pre.transform(row)
        shap_values = explainer.shap_values(x)
        # Tree explainers return per-class arrays for binary classifiers; pick class 1.
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            vals = np.asarray(shap_values[1])[0]
        else:
            arr = np.asarray(shap_values)
            vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        order = np.argsort(np.abs(vals))[::-1][:k]
        return [
            {
                "feature": feat_names[i] if i < len(feat_names) else f"f{i}",
                "shap_value": round(float(vals[i]), 4),
                "direction": "increases_anomaly" if vals[i] > 0 else "decreases_anomaly",
            }
            for i in order
        ]
    except Exception:  # noqa: BLE001 - never raise from a tool path.
        return []


@lru_cache(maxsize=1)
def _load_labs_nn():
    model_path = MODELS_DIR / "labs_nn.keras"
    scaler_path = MODELS_DIR / "labs_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: PLC0415
    return tf.keras.models.load_model(model_path), joblib.load(scaler_path)


def predict_claims_anomaly(
    cost_usd: float,
    procedure_count: int,
    los_days: int,
    age: int,
    drg_code: str,
) -> dict[str, Any]:
    """Score whether a claim is anomalous using the Random Forest model.

    Args:
      cost_usd: total billed cost of the claim in USD.
      procedure_count: number of distinct procedures on the claim.
      los_days: length of stay in days (0 for outpatient).
      age: patient age in years.
      drg_code: diagnosis-related group code as a string.

    Returns:
      {"available": bool, "score": float, "confidence": float, "rationale": str}
      score is the probability of anomaly in [0, 1].
    """
    try:
        model = _load_claims_rf()
    except Exception as exc:  # noqa: BLE001 - tools must return unavailable, never raise.
        return {
            "available": False,
            "reason": f"claims model load failed: {exc}",
        }
    if model is None:
        return {
            "available": False,
            "reason": "claims_rf.joblib not found — run train_claims_rf",
        }

    row = pd.DataFrame(
        [{
            "cost_usd": cost_usd,
            "procedure_count": procedure_count,
            "los_days": los_days,
            "age": age,
            "drg_code": drg_code,
        }]
    )
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # noqa: BLE001 - tool failures must stay inside the tool envelope.
        return {
            "available": False,
            "reason": f"claims prediction failed: {exc}",
        }
    confidence = float(abs(proba - 0.5) * 2)
    top_features = _top_shap_features(row, k=3)
    rationale_parts = [
        f"RF score {proba:.2f} on cost={cost_usd:.0f}, "
        f"procedures={procedure_count}, LOS={los_days}d"
    ]
    if top_features:
        drivers = ", ".join(f"{tf['feature']}({tf['shap_value']:+.2f})" for tf in top_features)
        rationale_parts.append(f"top drivers: {drivers}")
    rationale = " | ".join(rationale_parts)
    return {
        "available": True,
        "score": round(proba, 4),
        "confidence": round(confidence, 4),
        "rationale": rationale,
        "top_features": top_features,
    }


def score_labs_risk(
    a1c: float | None = None,
    ldl: float | None = None,
    egfr: float | None = None,
    troponin: float | None = None,
) -> dict[str, Any]:
    """Score lab-panel risk using the TensorFlow neural network.

    Args:
      a1c: hemoglobin A1C %. None if missing.
      ldl: LDL cholesterol mg/dL. None if missing.
      egfr: eGFR mL/min/1.73m^2. None if missing.
      troponin: troponin ng/mL. None if missing.

    Returns:
      {"available": bool, "score": float, "confidence": float, "rationale": str}
      Returns available=False if all panels are missing — Router uses this to
      skip the agent entirely on partial data.
    """
    values = [a1c, ldl, egfr, troponin]
    if all(v is None for v in values):
        return {"available": False, "reason": "no lab panels present"}

    try:
        bundle = _load_labs_nn()
    except Exception as exc:  # noqa: BLE001 - tools must return unavailable, never raise.
        return {
            "available": False,
            "reason": f"labs model load failed: {exc}",
        }
    if bundle is None:
        return {
            "available": False,
            "reason": "labs_nn.keras not found — run train_labs_nn",
        }

    means = np.array([6.5, 130.0, 70.0, 0.05])
    filled = np.array(
        [m if v is None else v for v, m in zip(values, means)],
        dtype=np.float32,
    ).reshape(1, -1)

    model, scaler = bundle
    try:
        scaled = scaler.transform(filled)
        proba = float(model.predict(scaled, verbose=0)[0, 0])
    except Exception as exc:  # noqa: BLE001 - tool failures must stay inside the tool envelope.
        return {
            "available": False,
            "reason": f"labs prediction failed: {exc}",
        }
    confidence = float(abs(proba - 0.5) * 2)

    missing = [name for name, v in zip(["a1c", "ldl", "egfr", "troponin"], values) if v is None]
    if missing:
        confidence *= 0.7
    rationale = (
        f"NN score {proba:.2f}"
        + (f" (imputed: {', '.join(missing)})" if missing else "")
    )
    return {
        "available": True,
        "score": round(proba, 4),
        "confidence": round(confidence, 4),
        "rationale": rationale,
    }


def write_audit_log(
    patient_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Append an immutable audit log row for the given decision.

    Args:
      patient_id: the patient identifier.
      decision: the structured decision dict from the ActionAgent.

    Returns:
      {"logged": True, "path": <log file path>}
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "audit.jsonl"
    record = {"patient_id": patient_id, "decision": decision}
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"logged": True, "path": str(log_path)}
