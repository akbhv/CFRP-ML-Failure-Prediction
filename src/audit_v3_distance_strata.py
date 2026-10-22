
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "ml_multitarget_v3"
RESULTS_DIR = ROOT / "results" / "multitarget_dnn_v3"
OUT_DIR = RESULTS_DIR / "cross_criterion_diagnostics"

LOADS = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

# Distance bands are measured in standardized six-feature load space.
BANDS = [
    ("0–0.6", 0.0, 0.6),
    ("0.6–1.5", 0.6, 1.5),
    ("1.5–2.5", 1.5, 2.5),
    (">2.5", 2.5, np.inf),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SPLIT_DIR / "feature_scaler.joblib")

    train_indices = np.load(SPLIT_DIR / "train_indices.npy").astype(int)
    test_indices = np.load(SPLIT_DIR / "test_indices.npy").astype(int)

    X_train = data.iloc[train_indices][LOADS].to_numpy(dtype=float)
    X_test = data.iloc[test_indices][LOADS].to_numpy(dtype=float)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_test_scaled)
    distances = distances.ravel()

    # Load the saved test predictions and align each file by original row_index.
    prediction_files = {}
    for target in TARGETS:
        path = RESULTS_DIR / target / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")

        pred = pd.read_csv(path)
        required = {"row_index", "reference_fi", "predicted_fi"}
        if not required.issubset(pred.columns):
            raise ValueError(
                f"{path.name} must contain columns {sorted(required)}; "
                f"found {list(pred.columns)}"
            )

        pred = pred[["row_index", "reference_fi", "predicted_fi"]].copy()
        pred["row_index"] = pred["row_index"].astype(int)

        if pred["row_index"].duplicated().any():
            raise ValueError(f"Duplicate row_index values in {path}")

        if set(pred["row_index"]) != set(test_indices):
            raise ValueError(
                f"Prediction row mapping does not match test indices for {target}"
            )

        prediction_files[target] = pred.set_index("row_index")

    # Build a shared table keyed by original dataset row index.
    audit = pd.DataFrame({
        "row_index": test_indices,
        "nearest_train_distance": distances,
    }).set_index("row_index")

    for target in TARGETS:
        pred = prediction_files[target].reindex(test_indices)
        audit[f"{target}_actual"] = pred["reference_fi"].to_numpy()
        audit[f"{target}_predicted"] = pred["predicted_fi"].to_numpy()

    # Save row-level distances for traceability.
    audit.reset_index().to_csv(
        OUT_DIR / "test_nearest_train_distances.csv", index=False
    )

    summary = []
    for band_name, lower, upper in BANDS:
        if lower == 0:
            mask = (audit["nearest_train_distance"] >= lower) & (
                audit["nearest_train_distance"] <= upper
            )
        else:
            mask = (audit["nearest_train_distance"] > lower) & (
                audit["nearest_train_distance"] <= upper
            )

        group = audit.loc[mask]
        for target in TARGETS:
            actual = group[f"{target}_actual"].to_numpy(dtype=float)
            predicted = group[f"{target}_predicted"].to_numpy(dtype=float)

            if len(actual) == 0:
                continue

            y_true = actual >= 1.0
            y_pred = predicted >= 1.0

            false_safe = int(np.sum(y_true & ~y_pred))
            summary.append({
                "distance_band": band_name,
                "distance_lower": lower,
                "distance_upper": upper,
                "target": target,
                "n": len(group),
                "false_safe_count": false_safe,
                "false_safe_rate_all_samples": false_safe / len(group),
                "false_safe_rate_actual_failed": (
                    false_safe / int(y_true.sum()) if y_true.sum() else np.nan
                ),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "mae": float(np.mean(np.abs(actual - predicted))),
                "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            })

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_DIR / "distance_stratified_model_audit.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nNearest-training-sample distance distribution:")
    print(pd.Series(distances).describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_string())

    print("\nDistance-stratified audit:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved row-level distances: {OUT_DIR / 'test_nearest_train_distances.csv'}")


if __name__ == "__main__":
    main()
