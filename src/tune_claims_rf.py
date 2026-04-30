"""
Bayesian hyperparameter sweep for the Random Forest claims classifier.

Replaces the hand-tuned configuration in train_claims_rf.py with a principled
Optuna search. Optimizes 5-fold CV AUC, logs every trial to MLflow (when
installed), prints the winning config, and writes it to
models/claims_rf_best_params.json so train_claims_rf.py can be re-run with it.

Usage:
  python -m src.tune_claims_rf --trials 30 --cv-folds 5
  python -m src.tune_claims_rf --trials 50 --timeout 600

Cost-conscious defaults: 30 trials × 5 folds on 2k rows finishes in ~2 minutes
on a laptop CPU. Bump --trials to 100 for a real sweep.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["cost_usd", "procedure_count", "los_days", "age"]
CATEGORICAL_FEATURES = ["drg_code"]
TARGET = "is_anomaly"
EXPERIMENT_NAME = "claims-risk-rf-sweep"


def _load_optuna():
    try:
        import optuna  # noqa: PLC0415

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for hyperparameter tuning. "
            "Install with: pip install optuna>=3.6"
        ) from exc


def _safe_mlflow():
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except ImportError:
        return None


def build_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def make_objective(X: pd.DataFrame, y: pd.Series, cv_folds: int, mlflow):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        pipe = build_pipeline(params)
        aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        score = float(np.mean(aucs))

        if mlflow is not None:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "cv_auc_mean": score,
                    "cv_auc_std": float(np.std(aucs)),
                })
        return score

    return objective


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna sweep for the claims RF")
    p.add_argument("--trials", type=int, default=30, help="Number of trials")
    p.add_argument("--cv-folds", type=int, default=5, help="Stratified K-fold folds")
    p.add_argument("--timeout", type=int, default=None, help="Max seconds (optional)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models/claims_rf_best_params.json"),
        help="Where to save the winning params",
    )
    args = p.parse_args()

    data_path = Path("data/claims.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run `python -m src.generate_data` first")

    df = pd.read_csv(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    optuna = _load_optuna()
    mlflow = _safe_mlflow()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="claims-rf-sweep")

    if mlflow is not None:
        with mlflow.start_run(run_name="sweep_parent"):
            mlflow.log_params({
                "trials": args.trials,
                "cv_folds": args.cv_folds,
                "sampler": "TPESampler(seed=42)",
                "n_train": len(X),
            })
            study.optimize(
                make_objective(X, y, args.cv_folds, mlflow),
                n_trials=args.trials,
                timeout=args.timeout,
            )
            mlflow.log_metrics({"best_cv_auc": study.best_value})
            mlflow.log_dict(study.best_params, "best_params.json")
    else:
        study.optimize(
            make_objective(X, y, args.cv_folds, mlflow),
            n_trials=args.trials,
            timeout=args.timeout,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {
            "best_cv_auc": round(float(study.best_value), 4),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        },
        indent=2,
    ))

    print(f"\nBest CV AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Saved -> {args.out}")
    if mlflow is not None:
        print("Browse trials: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
