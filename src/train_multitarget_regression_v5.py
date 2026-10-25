"""Train seven independent DNN regressors on the V5 CFRP dataset.

Run from the project root:
    python -m src.train_multitarget_regression_v5

Safety and evaluation:
- Reads only data/processed/failure_dataset_v5.csv.
- Writes to new V5-specific directories; refuses to overwrite either directory.
- Uses one shared 70/15/15 split stratified by sampling_region.
- Keeps the unique exact-zero-load anchor in the training set.
- Fits StandardScaler on training features only.
- Reports overall, sampling-region, low-load, and failure-boundary performance.
- Does not modify V2/V3/V4 datasets or existing model artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
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


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v5.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "ml_multitarget_v5"
OUTPUT_DIR = ROOT / "results" / "multitarget_dnn_v5"

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
SEED = 20261025
MAX_EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FAILURE_THRESHOLD = 1.0
NEAR_BOUNDARY_LOW = 0.9
NEAR_BOUNDARY_HIGH = 1.1


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


def save_scatter_plot(
    actual: np.ndarray, predicted: np.ndarray, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.55, s=16)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    if np.isclose(low, high):
        low -= 0.05
        high += 0.05
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Reference failure index")
    ax.set_ylabel("Predicted failure index")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residual_plot(
    actual: np.ndarray, predicted: np.ndarray, path: Path, title: str
) -> None:
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


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) == 0:
        return {
            "sample_count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    # R2 is undefined for fewer than two samples or a constant target.
    r2 = (
        float(r2_score(actual, predicted))
        if len(actual) >= 2 and not np.allclose(actual, actual[0])
        else np.nan
    )
    return {
        "sample_count": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": rmse,
        "R2": r2,
    }


def classification_metrics(actual_fi: np.ndarray, predicted_fi: np.ndarray) -> dict:
    actual_failed = np.asarray(actual_fi) >= FAILURE_THRESHOLD
    predicted_failed = np.asarray(predicted_fi) >= FAILURE_THRESHOLD
    return {
        "accuracy": float(accuracy_score(actual_failed, predicted_failed)),
        "precision": float(
            precision_score(actual_failed, predicted_failed, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_failed, predicted_failed, zero_division=0)
        ),
        "F1": float(f1_score(actual_failed, predicted_failed, zero_division=0)),
        "actual_failed_count": int(actual_failed.sum()),
        "predicted_failed_count": int(predicted_failed.sum()),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")
    if SPLIT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split directory: {SPLIT_DIR}"
        )
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results directory: {OUTPUT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + TARGETS + ["sampling_region"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) != 25_001:
        raise ValueError(f"Expected 25,001 rows, found {len(df)}")

    numeric = df[FEATURES + TARGETS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Inputs or targets contain NaN or infinite values.")

    # There must be exactly one all-zero input row and all seven target FIs
    # must be zero for that physical anchor.
    zero_mask = (df[FEATURES] == 0.0).all(axis=1)
    if int(zero_mask.sum()) != 1:
        raise ValueError(
            f"Expected exactly one zero-load row; found {int(zero_mask.sum())}"
        )
    zero_index = int(np.flatnonzero(zero_mask.to_numpy())[0])
    if not np.allclose(
        df.loc[zero_mask, TARGETS].to_numpy(dtype=float),
        0.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("Zero-load row does not have zero target failure indices.")

    # Keep the unique zero-load anchor in training; stratify the remaining
    # samples by sampling region so all load bands and boundary regions are
    # represented in train, validation, and test.
    all_indices = np.arange(len(df))
    nonzero_indices = all_indices[all_indices != zero_index]
    region_labels = df.loc[nonzero_indices, "sampling_region"].astype(str).to_numpy()

    train_nonzero, temp_idx = train_test_split(
        nonzero_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=region_labels,
    )
    temp_regions = df.loc[temp_idx, "sampling_region"].astype(str).to_numpy()
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_regions,
    )
    train_idx = np.sort(np.concatenate([train_nonzero, [zero_index]]))
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    # Sanity checks: no overlap, no missing row, and expected split sizes.
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise RuntimeError("Train/validation/test splits overlap.")
    if len(np.concatenate([train_idx, val_idx, test_idx])) != len(df):
        raise RuntimeError("Some dataset rows are missing from the splits.")
    if len(train_idx) != 17_501 or len(val_idx) != 3_750 or len(test_idx) != 3_750:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_idx)}, "
            f"validation={len(val_idx)}, test={len(test_idx)}"
        )

    X = df[FEATURES].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)

    # Create output directories only after all data and split checks succeed.
    SPLIT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    joblib.dump(scaler, SPLIT_DIR / "feature_scaler.joblib")
    np.save(SPLIT_DIR / "train_indices.npy", train_idx)
    np.save(SPLIT_DIR / "val_indices.npy", val_idx)
    np.save(SPLIT_DIR / "test_indices.npy", test_idx)
    np.save(SPLIT_DIR / "X_train_scaled.npy", X_train)
    np.save(SPLIT_DIR / "X_val_scaled.npy", X_val)
    np.save(SPLIT_DIR / "X_test_scaled.npy", X_test)

    split_summary = {
        "train": int(len(train_idx)),
        "validation": int(len(val_idx)),
        "test": int(len(test_idx)),
        "zero_load_anchor_index": zero_index,
        "zero_load_anchor_in": "training",
        "train_region_counts": df.loc[train_idx, "sampling_region"].value_counts().to_dict(),
        "validation_region_counts": df.loc[val_idx, "sampling_region"].value_counts().to_dict(),
        "test_region_counts": df.loc[test_idx, "sampling_region"].value_counts().to_dict(),
    }

    config = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "row_count": int(len(df)),
        "features": FEATURES,
        "targets": TARGETS,
        "split": split_summary,
        "split_method": (
            "70/15/15 split of nonzero rows stratified by sampling_region; "
            "the unique zero-load anchor is added to training"
        ),
        "scaler": "StandardScaler fit on training features only",
        "architecture": [
            "Dense(64, relu)",
            "Dense(64, relu)",
            "Dense(32, relu)",
            "Dense(1, linear)",
        ],
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "MSE",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 30,
            "restore_best_weights": True,
        },
        "failure_threshold": FAILURE_THRESHOLD,
        "near_boundary_interval": [NEAR_BOUNDARY_LOW, NEAR_BOUNDARY_HIGH],
        "seed_base": SEED,
        "limitations": [
            "The exact-zero anchor is in training, so its prediction is an in-sample physical-anchor check, not an independent test.",
            "Dataset/API agreement is a computational consistency check, not experimental validation.",
            "Performance must be interpreted separately for each sampling region and load magnitude.",
        ],
    }
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    metric_records = []
    boundary_records = []
    subgroup_records = []

    print(f"Dataset: {DATA_PATH}")
    print(
        f"Rows: {len(df)} | train={len(train_idx)} | "
        f"validation={len(val_idx)} | test={len(test_idx)}"
    )
    print(f"Outputs: {OUTPUT_DIR}")
    print("Split by sampling region; exact-zero anchor is included in training.")
    print("\nTraining seven independent regressors...\n")

    for target_number, target in enumerate(TARGETS):
        print(f"[{target_number + 1}/{len(TARGETS)}] Training {target} ...")
        target_dir = OUTPUT_DIR / target
        target_dir.mkdir(parents=False, exist_ok=False)

        y = df[target].to_numpy(dtype=np.float32)
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        tf.keras.backend.clear_session()
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
        y_train_zero_pred = model.predict(
            scaler.transform(X[[zero_index]]).astype(np.float32), verbose=0
        ).reshape(-1)[0]

        overall = regression_metrics(y_test, y_pred)
        boundary = classification_metrics(y_test, y_pred)
        near_boundary_mask = (
            (y_test >= NEAR_BOUNDARY_LOW) & (y_test <= NEAR_BOUNDARY_HIGH)
        )
        near_boundary_cls = (
            classification_metrics(y_test[near_boundary_mask], y_pred[near_boundary_mask])
            if near_boundary_mask.any()
            else {
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "F1": np.nan,
                "actual_failed_count": 0,
                "predicted_failed_count": 0,
            }
        )

        region_values = df.loc[test_idx, "sampling_region"].astype(str).to_numpy()
        for region in sorted(np.unique(region_values)):
            mask = region_values == region
            region_metrics = regression_metrics(y_test[mask], y_pred[mask])
            subgroup_records.append(
                {
                    "target": target,
                    "subgroup_type": "sampling_region",
                    "subgroup": region,
                    **region_metrics,
                }
            )

        # Also report nested load-magnitude subgroups based on all three
        # force resultants and all three moment resultants.
        test_loads = df.loc[test_idx, FEATURES]
        force_max = test_loads[["Nx", "Ny", "Nxy"]].abs().max(axis=1).to_numpy()
        moment_max = test_loads[["Mx", "My", "Mxy"]].abs().max(axis=1).to_numpy()
        load_groups = {
            "forces<=10_Npm_and_moments<=1_N": (force_max <= 10) & (moment_max <= 1),
            "forces<=100_Npm_and_moments<=10_N": (force_max <= 100) & (moment_max <= 10),
            "forces<=1000_Npm_and_moments<=10_N": (force_max <= 1000) & (moment_max <= 10),
        }
        for group_name, mask in load_groups.items():
            group_metrics = regression_metrics(y_test[mask], y_pred[mask])
            subgroup_records.append(
                {
                    "target": target,
                    "subgroup_type": "load_magnitude",
                    "subgroup": group_name,
                    **group_metrics,
                }
            )

        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        model.save(target_dir / f"{target}_model.keras")
        np.save(target_dir / "y_test.npy", y_test)
        np.save(target_dir / "y_pred_test.npy", y_pred)
        np.save(target_dir / "y_val.npy", y_val)
        np.save(target_dir / "y_pred_val.npy", y_val_pred)

        predictions = df.loc[test_idx, FEATURES + ["sampling_region"]].copy()
        predictions.insert(0, "row_index", test_idx)
        predictions["reference_fi"] = y_test
        predictions["predicted_fi"] = y_pred
        predictions["residual_predicted_minus_reference"] = y_pred - y_test
        predictions["reference_failed_fi_ge_1"] = (y_test >= 1.0).astype(int)
        predictions["predicted_failed_fi_ge_1"] = (y_pred >= 1.0).astype(int)
        predictions.to_csv(target_dir / "test_predictions.csv", index=False)

        pd.DataFrame(history.history).to_csv(
            target_dir / "training_history.csv", index_label="epoch"
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        epochs = np.arange(1, len(history.history["loss"]) + 1)
        ax.plot(epochs, history.history["loss"], label="Training MSE")
        ax.plot(epochs, history.history["val_loss"], label="Validation MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title(f"{target}: training history")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(target_dir / "training_history.png", dpi=180)
        plt.close(fig)

        save_scatter_plot(
            y_test, y_pred, target_dir / "actual_vs_predicted.png",
            f"{target}: V5 test set"
        )
        save_residual_plot(
            y_test, y_pred, target_dir / "residuals.png",
            f"{target}: V5 test residuals"
        )

        metric_records.append(
            {
                "target": target,
                "test_MAE": overall["MAE"],
                "test_RMSE": overall["RMSE"],
                "test_R2": overall["R2"],
                "parameter_count": parameter_count,
                "training_time_seconds": float(training_seconds),
                "best_epoch_by_val_loss": best_epoch,
                "epochs_run": len(history.history["loss"]),
                "test_rows": len(test_idx),
                "test_failure_accuracy": boundary["accuracy"],
                "test_failure_precision": boundary["precision"],
                "test_failure_recall": boundary["recall"],
                "test_failure_F1": boundary["F1"],
                "near_boundary_samples": int(near_boundary_mask.sum()),
                "near_boundary_failure_accuracy": near_boundary_cls["accuracy"],
                "near_boundary_failure_F1": near_boundary_cls["F1"],
                "zero_anchor_training_prediction": float(y_train_zero_pred),
                "zero_anchor_absolute_error_in_sample": float(abs(y_train_zero_pred)),
            }
        )
        boundary_records.append(
            {
                "target": target,
                "threshold": FAILURE_THRESHOLD,
                **boundary,
                "near_boundary_sample_count": int(near_boundary_mask.sum()),
                "near_boundary_accuracy": near_boundary_cls["accuracy"],
                "near_boundary_precision": near_boundary_cls["precision"],
                "near_boundary_recall": near_boundary_cls["recall"],
                "near_boundary_F1": near_boundary_cls["F1"],
            }
        )

        print(
            f"  MAE={overall['MAE']:.6f} | RMSE={overall['RMSE']:.6f} "
            f"| R2={overall['R2']:.6f} | failure F1={boundary['F1']:.4f} "
            f"| near-boundary n={int(near_boundary_mask.sum())} "
            f"| zero-anchor in-sample prediction={y_train_zero_pred:.6f} "
            f"| epochs={len(history.history['loss'])} "
            f"| time={training_seconds:.1f}s"
        )

    metrics_df = pd.DataFrame(metric_records)
    boundary_df = pd.DataFrame(boundary_records)
    subgroup_df = pd.DataFrame(subgroup_records)
    metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    boundary_df.to_csv(OUTPUT_DIR / "failure_boundary_metrics.csv", index=False)
    subgroup_df.to_csv(OUTPUT_DIR / "subgroup_metrics.csv", index=False)

    print("\n=== Regression metrics (overall test set) ===")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== Derived failure classification at FI >= 1 (test set) ===")
    print(boundary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== Subgroup metrics ===")
    print(subgroup_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nCompleted. Results saved to: {OUTPUT_DIR}")
    print(f"Shared splits and scaler saved to: {SPLIT_DIR}")
    print("Review subgroup_metrics.csv before making claims about low-load accuracy.")


if __name__ == "__main__":
    main()