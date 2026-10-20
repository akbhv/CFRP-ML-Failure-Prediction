"""
Train seven independent DNN regressors for composite-laminate failure indices.

Safety / reproducibility:
- Reads only failure_dataset_v2_puck_corrected_exact.csv.
- Does not modify the original dataset or existing Hashin baseline artifacts.
- Uses one shared 70/15/15 split for all targets, stratified by the dataset's
  existing generic `failed` label to mirror the established preparation workflow.
- Fits feature scaling on the training split only.
- Refuses to overwrite existing output directories.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v2_hoffman_corrected.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "ml_multitarget_v2_hoffman_corrected"
OUTPUT_DIR = ROOT / "results" / "multitarget_dnn_v2_hoffman_corrected"

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]
SEED = 42
MAX_EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FAILURE_THRESHOLD = 1.0


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def save_scatter_plot(actual: np.ndarray, predicted: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.55, s=16)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Reference failure index")
    ax.set_ylabel("Predicted failure index")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residual_plot(actual: np.ndarray, predicted: np.ndarray, path: Path, title: str) -> None:
    residuals = predicted - actual
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, residuals, alpha=0.55, s=16)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Reference failure index")
    ax.set_ylabel("Prediction error (predicted - reference)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    # Avoid accidental overwriting of any prior run.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")
    if SPLIT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing split directory: {SPLIT_DIR}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing results directory: {OUTPUT_DIR}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + TARGETS + ["failed"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) != 20000:
        raise ValueError(f"Expected 20,000 rows, found {len(df)}")

    numeric = df[FEATURES + TARGETS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Inputs or targets contain NaN or infinite values.")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    generic_labels = df["failed"].astype(int).to_numpy()
    row_indices = np.arange(len(df))

    # Reproduce the existing 70/15/15 split strategy with the generic
    # failure label used by the existing Hashin preparation workflow.
    train_idx, temp_idx = train_test_split(
        row_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=generic_labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=generic_labels[temp_idx],
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    # Create outputs only after validation and splitting succeeded.
    SPLIT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, SPLIT_DIR / "feature_scaler.joblib")
    np.save(SPLIT_DIR / "train_indices.npy", train_idx)
    np.save(SPLIT_DIR / "val_indices.npy", val_idx)
    np.save(SPLIT_DIR / "test_indices.npy", test_idx)
    np.save(SPLIT_DIR / "X_train_scaled.npy", X_train)
    np.save(SPLIT_DIR / "X_val_scaled.npy", X_val)
    np.save(SPLIT_DIR / "X_test_scaled.npy", X_test)

    config = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "row_count": int(len(df)),
        "features": FEATURES,
        "targets": TARGETS,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_method": "70/15/15; random_state=42; stratified by existing generic failed label",
        "scaler": "StandardScaler fit on training features only",
        "architecture": ["Dense(64, relu)", "Dense(64, relu)", "Dense(32, relu)", "Dense(1, linear)"],
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "MSE",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {"monitor": "val_loss", "patience": 30, "restore_best_weights": True},
        "failure_threshold": FAILURE_THRESHOLD,
        "seed_base": SEED,
        "note": "The corrected Hoffman target passed computational checks; experimental validation remains pending.",
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metric_records = []
    boundary_records = []

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Training seven independent regressors...\n")

    for target_number, target in enumerate(TARGETS):
        print(f"[{target_number + 1}/{len(TARGETS)}] Training {target} ...")
        target_dir = OUTPUT_DIR / target
        target_dir.mkdir(parents=False, exist_ok=False)

        y = df[target].to_numpy(dtype=np.float32)
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        tf.keras.utils.set_random_seed(SEED + target_number)
        model = build_model(X_train.shape[1])
        parameter_count = int(model.count_params())

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        )

        start = time.perf_counter()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0,
            shuffle=True,
        )
        training_seconds = time.perf_counter() - start

        y_pred = model.predict(X_test, verbose=0).reshape(-1)
        y_val_pred = model.predict(X_val, verbose=0).reshape(-1)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

        actual_failed = y_test >= FAILURE_THRESHOLD
        predicted_failed = y_pred >= FAILURE_THRESHOLD
        boundary_accuracy = float(accuracy_score(actual_failed, predicted_failed))
        boundary_precision = float(precision_score(actual_failed, predicted_failed, zero_division=0))
        boundary_recall = float(recall_score(actual_failed, predicted_failed, zero_division=0))
        boundary_f1 = float(f1_score(actual_failed, predicted_failed, zero_division=0))

        model.save(target_dir / f"{target}_model.keras")
        np.save(target_dir / "y_test.npy", y_test)
        np.save(target_dir / "y_pred_test.npy", y_pred)
        np.save(target_dir / "y_val.npy", y_val)
        np.save(target_dir / "y_pred_val.npy", y_val_pred)
        pd.DataFrame(
            {
                "row_index": test_idx,
                "reference_fi": y_test,
                "predicted_fi": y_pred,
                "residual_predicted_minus_reference": y_pred - y_test,
                "reference_failed_fi_ge_1": actual_failed.astype(int),
                "predicted_failed_fi_ge_1": predicted_failed.astype(int),
            }
        ).to_csv(target_dir / "test_predictions.csv", index=False)

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(target_dir / "training_history.csv", index_label="epoch")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(np.arange(1, len(history.history["loss"]) + 1), history.history["loss"], label="Training MSE")
        ax.plot(np.arange(1, len(history.history["val_loss"]) + 1), history.history["val_loss"], label="Validation MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title(f"{target}: training history")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(target_dir / "training_history.png", dpi=180)
        plt.close(fig)

        save_scatter_plot(y_test, y_pred, target_dir / "actual_vs_predicted.png", f"{target}: test set")
        save_residual_plot(y_test, y_pred, target_dir / "residuals.png", f"{target}: test residuals")

        metric_records.append(
            {
                "target": target,
                "test_MAE": mae,
                "test_RMSE": rmse,
                "test_R2": r2,
                "parameter_count": parameter_count,
                "training_time_seconds": float(training_seconds),
                "best_epoch_by_val_loss": best_epoch,
                "epochs_run": len(history.history["loss"]),
                "test_rows": len(test_idx),
            }
        )
        boundary_records.append(
            {
                "target": target,
                "threshold": FAILURE_THRESHOLD,
                "accuracy": boundary_accuracy,
                "precision": boundary_precision,
                "recall": boundary_recall,
                "F1": boundary_f1,
                "actual_failed_count": int(actual_failed.sum()),
                "predicted_failed_count": int(predicted_failed.sum()),
            }
        )

        print(
            f"  MAE={mae:.6f} | RMSE={rmse:.6f} | R2={r2:.6f} "
            f"| boundary F1={boundary_f1:.4f} | epochs={len(history.history['loss'])} "
            f"| time={training_seconds:.1f}s"
        )

    metrics_df = pd.DataFrame(metric_records)
    boundary_df = pd.DataFrame(boundary_records)
    metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    boundary_df.to_csv(OUTPUT_DIR / "failure_boundary_metrics.csv", index=False)

    print("\n=== Regression metrics (test set) ===")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== Derived failure classification at FI >= 1 (test set) ===")
    print(boundary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nCompleted. Results saved to: {OUTPUT_DIR}")
    print(f"Shared splits and scaler saved to: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
