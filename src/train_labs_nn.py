"""
Train the TensorFlow Keras DNN that scores lab-panel risk.

Inputs : data/labs.csv
Outputs: models/labs_nn.keras
         models/labs_scaler.joblib   (StandardScaler — apply at inference)
         outputs/model_metrics/labs_nn_metrics.json (AUC + accuracy + calibration)
         mlruns/                     (MLflow tracking — when mlflow installed)

Usage  : python -m src.train_labs_nn
         (optional) mlflow ui --backend-store-uri ./mlruns
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .lab_schema import LAB_FEATURES as FEATURES
from .lab_schema import LAB_TARGET as TARGET
from .lab_schema import normalize_labs_dataframe

EXPERIMENT_NAME = "claims-risk-labs-nn"


def build_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _safe_mlflow():
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except ImportError:
        return None


def main() -> None:
    tf.random.set_seed(42)
    np.random.seed(42)

    data_path = Path("data/labs.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run `python -m src.generate_data` first")

    df = normalize_labs_dataframe(pd.read_csv(data_path))
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = build_model(input_dim=len(FEATURES))
    history = model.fit(
        X_tr_s,
        y_tr,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )

    proba = model.predict(X_te_s, verbose=0).ravel()
    auc = float(roc_auc_score(y_te, proba))
    acc = float(np.mean((proba >= 0.5).astype(int) == y_te.astype(int)))
    brier = float(brier_score_loss(y_te, proba))
    ll = float(log_loss(y_te, np.clip(proba, 1e-7, 1 - 1e-7)))

    Path("models").mkdir(exist_ok=True)
    Path("outputs/model_metrics").mkdir(parents=True, exist_ok=True)
    model.save("models/labs_nn.keras")
    joblib.dump(scaler, "models/labs_scaler.joblib")

    metrics = {
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "epochs": 20,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "final_train_auc": round(float(history.history["auc"][-1]), 4),
        "final_val_auc": round(float(history.history["val_auc"][-1]), 4),
    }
    metrics_path = Path("outputs/model_metrics/labs_nn_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    mlflow = _safe_mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name="labs_nn_baseline"):
            mlflow.log_params({
                "model_type": "Keras DNN [32,16,1]",
                "epochs": 20,
                "batch_size": 64,
                "optimizer": "adam",
                "lr": 1e-3,
                "dropout": 0.2,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
            })
            mlflow.log_metrics({
                "auc": auc,
                "accuracy": acc,
                "brier_score": brier,
                "log_loss": ll,
                "final_val_auc": metrics["final_val_auc"],
            })
            mlflow.log_artifact("models/labs_nn.keras")
            mlflow.log_artifact("models/labs_scaler.joblib")
            mlflow.log_artifact(str(metrics_path))

    print(f"Labs NN AUC: {auc:.4f} | accuracy: {acc:.4f} | Brier: {brier:.4f}")
    print("Saved -> models/labs_nn.keras + models/labs_scaler.joblib")
    print(f"Metrics -> {metrics_path}")
    if mlflow is not None:
        print("Logged -> mlflow run (open with: mlflow ui --backend-store-uri ./mlruns)")


if __name__ == "__main__":
    main()
