"""
Read-only diagnostic for suspicious test row(s) in the CFRP seven-target DNN run.
Run from project root:
    python src/inspect_multitarget_row.py
No model training or source-dataset modification occurs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "failure_dataset_v2_puck_corrected_exact.csv"
SPLITS = ROOT / "data" / "processed" / "ml_multitarget_v1"
RESULTS = ROOT / "results" / "multitarget_dnn_v1"

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = [
    "max_stress_fi", "max_strain_fi", "tsai_hill_fi",
    "tsai_wu_fi", "hoffman_fi", "hashin_fi", "puck_fi",
]
SUSPECT_ROW = 14987
TOP_K = 10


def main():
    df = pd.read_csv(DATA)
    train_idx = np.load(SPLITS / "train_indices.npy")
    val_idx = np.load(SPLITS / "val_indices.npy")
    test_idx = np.load(SPLITS / "test_indices.npy")

    if SUSPECT_ROW < 0 or SUSPECT_ROW >= len(df):
        raise IndexError(f"Row index {SUSPECT_ROW} is outside dataset.")
    if SUSPECT_ROW not in set(test_idx.tolist()):
        print(f"WARNING: row {SUSPECT_ROW} is not in the saved test split.")

    # Print full row, keeping original dataset column names and values.
    row = df.iloc[SUSPECT_ROW]
    print(f"=== Original dataset row at positional index {SUSPECT_ROW} ===")
    print(row.to_string())

    # Show model errors for this row across all seven targets.
    print("\n=== Saved test predictions for this row ===")
    for target in TARGETS:
        pred_path = RESULTS / target / "test_predictions.csv"
        pred_df = pd.read_csv(pred_path)
        match = pred_df[pred_df["row_index"] == SUSPECT_ROW]
        if match.empty:
            print(f"{target}: no saved test prediction for this row")
            continue
        rec = match.iloc[0]
        print(
            f"{target:16s} reference={rec['reference_fi']:.9g} "
            f"predicted={rec['predicted_fi']:.9g} "
            f"abs_error={abs(rec['predicted_fi'] - rec['reference_fi']):.9g} "
            f"reference_failed={int(rec['reference_failed_fi_ge_1'])} "
            f"predicted_failed={int(rec['predicted_failed_fi_ge_1'])}"
        )

    # Compare feature values with the dataset distribution.
    print("\n=== Feature position within full-dataset distribution ===")
    for feature in FEATURES:
        values = df[feature].to_numpy(dtype=float)
        value = float(row[feature])
        percentile = 100.0 * float(np.mean(values <= value))
        print(
            f"{feature:6s} value={value: .8g} "
            f"min={values.min(): .8g} median={np.median(values): .8g} "
            f"max={values.max(): .8g} percentile~{percentile:.2f}"
        )

    # Find closest training samples in standardized feature space.
    # Scale is fit only on the training inputs, matching ML practice.
    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X[train_idx])
    X_scaled = scaler.transform(X)
    train_distances = np.linalg.norm(
        X_scaled[train_idx] - X_scaled[SUSPECT_ROW], axis=1
    )
    nearest_order = np.argsort(train_distances)[:TOP_K]
    nearest_indices = train_idx[nearest_order]
    nearest_distances = train_distances[nearest_order]

    neighbor_rows = []
    for rank, (idx, distance) in enumerate(zip(nearest_indices, nearest_distances), start=1):
        rec = {"rank": rank, "row_index": int(idx), "standardized_feature_distance": float(distance)}
        for feature in FEATURES:
            rec[feature] = float(df.iloc[idx][feature])
        for target in TARGETS:
            rec[target] = float(df.iloc[idx][target])
        neighbor_rows.append(rec)

    neighbors = pd.DataFrame(neighbor_rows)
    print(f"\n=== {TOP_K} nearest training rows in standardized six-feature space ===")
    print(neighbors.to_string(index=False, float_format=lambda x: f"{x:.7g}"))

    # Add the suspect row as a reference line in the exported diagnostic table.
    suspect_record = {"rank": 0, "row_index": SUSPECT_ROW, "standardized_feature_distance": 0.0}
    for feature in FEATURES:
        suspect_record[feature] = float(df.iloc[SUSPECT_ROW][feature])
    for target in TARGETS:
        suspect_record[target] = float(df.iloc[SUSPECT_ROW][target])
    combined = pd.concat([pd.DataFrame([suspect_record]), neighbors], ignore_index=True)

    out = RESULTS / "row_14987_nearest_training_diagnostics.csv"
    combined.to_csv(out, index=False)
    print(f"\nDiagnostic table saved to: {out}")
    print("This is a diagnostic only; it does not change model weights or dataset values.")


if __name__ == "__main__":
    main()
