"""
Train the Random Forest claims anomaly classifier.

Inputs : data/claims.csv
Outputs: models/claims_rf.joblib       (model + scaler bundled)
         models/claims_rf_metrics.json (AUC + calibration + report)
         models/claims_rf_feature_importance.json (top features)
         mlruns/                       (MLflow tracking — when mlflow installed)

Usage  : python -m src.train_claims_rf
         (optional) mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = ["cost_usd", "procedure_count", "los_days", "age"]
CATEGORICAL_FEATURES = ["drg_code"]
TARGET = "is_anomaly"

EXPERIMENT_NAME = "claims-risk-rf"


def build_pipeline() -> Pipeline:
    """Tuned configuration. Targets AUC >= 0.75 on the synthetic dataset.

    Changes vs. the baseline (n_estimators=200, max_depth=8, leaf=10):
      - 500 trees: better averaging on the small dataset (n=2000, ~5% noise)
      - max_depth=None: lets trees split far enough to capture the
        cost x procedure_count interaction that defines the synthetic anomaly
      - min_samples_leaf=2 + min_samples_split=5: still regularized, but not
        as aggressively as the baseline (which over-smoothed the boundary)
      - max_features='sqrt': textbook RF default, kept explicit for clarity
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _expanded_feature_names(pipe: Pipeline) -> list[str]:
    """Recover post-transform feature names so importances are interpretable."""
    pre = pipe.named_steps["pre"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:  # noqa: BLE001 - older sklearn fallback.
        return [f"f{i}" for i in range(pipe.named_steps["clf"].n_features_in_)]


def _compute_calibration(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Brier score + reliability curve points. Useful when AUC alone misleads."""
    brier = float(brier_score_loss(y_true, proba))
    ll = float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)))
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    return {
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "reliability_curve": [
            {"mean_predicted": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
            for mp, fp in zip(mean_pred, frac_pos)
        ],
    }


def _safe_mlflow():
    """Yield an mlflow module if installed, else None. Never raises on missing dep."""
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except ImportError:
        return None


def main() -> None:
    data_path = Path("data/claims.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run `python -m src.generate_data` first")

    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()

    # 5-fold cross-validated AUC for variance estimate (hiring panels look for this).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te, proba))
    report = classification_report(y_te, pred, output_dict=True)
    calibration = _compute_calibration(np.asarray(y_te), np.asarray(proba))

    feature_names = _expanded_feature_names(pipe)
    importances = pipe.named_steps["clf"].feature_importances_.tolist()
    top_features = sorted(
        zip(feature_names, importances), key=lambda kv: kv[1], reverse=True
    )[:10]

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/claims_rf.joblib")

    metrics = {
        "auc_holdout": round(auc, 4),
        "auc_cv_mean": round(float(np.mean(cv_aucs)), 4),
        "auc_cv_std": round(float(np.std(cv_aucs)), 4),
        "auc_cv_folds": [round(float(a), 4) for a in cv_aucs],
        "calibration": calibration,
        "report": report,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "positive_rate_train": round(float(y_tr.mean()), 4),
        "positive_rate_test": round(float(y_te.mean()), 4),
    }
    Path("models/claims_rf_metrics.json").write_text(json.dumps(metrics, indent=2))
    Path("models/claims_rf_feature_importance.json").write_text(
        json.dumps(
            {"top_features": [{"feature": f, "importance": round(i, 4)} for f, i in top_features]},
            indent=2,
        )
    )

    # MLflow tracking (only if installed; otherwise no-op).
    mlflow = _safe_mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name="rf_baseline_tuned"):
            mlflow.log_params({
                "model_type": "RandomForestClassifier",
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 2,
                "min_samples_split": 5,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "n_train": len(X_tr),
                "n_test": len(X_te),
            })
            mlflow.log_metrics({
                "auc_holdout": auc,
                "auc_cv_mean": float(np.mean(cv_aucs)),
                "auc_cv_std": float(np.std(cv_aucs)),
                "brier_score": calibration["brier_score"],
                "log_loss": calibration["log_loss"],
                "precision_pos": float(report["1"]["precision"]),
                "recall_pos": float(report["1"]["recall"]),
                "f1_pos": float(report["1"]["f1-score"]),
            })
            mlflow.log_artifact("models/claims_rf.joblib")
            mlflow.log_artifact("models/claims_rf_metrics.json")
            mlflow.log_artifact("models/claims_rf_feature_importance.json")

    print(f"Random Forest holdout AUC: {auc:.4f}")
    print(f"  5-fold CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  Brier score: {calibration['brier_score']:.4f}  (lower is better)")
    print(f"  Log loss:    {calibration['log_loss']:.4f}")
    print(f"  Top features: {[f for f, _ in top_features[:3]]}")
    print("Saved -> models/claims_rf.joblib")
    if mlflow is not None:
        print("Logged -> mlflow run (open with: mlflow ui --backend-store-uri ./mlruns)")


if __name__ == "__main__":
    main()
