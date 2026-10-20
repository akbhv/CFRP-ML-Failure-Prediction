"""
Read-only post-training audit for the seven-target CFRP DNN experiment.
Does not modify datasets, models, splits, or training outputs.
Run from project root:
    python src/audit_multitarget_results.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multitarget_dnn_v1"
SPLITS = ROOT / "data" / "processed" / "ml_multitarget_v1"
TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]
BAND_LOW = 0.90
BAND_HIGH = 1.10
THRESHOLD = 1.0


def main():
    if not RESULTS.is_dir() or not SPLITS.is_dir():
        raise FileNotFoundError("Expected multitarget results and split directories were not found.")

    test_idx = np.load(SPLITS / "test_indices.npy")
    train_idx = np.load(SPLITS / "train_indices.npy")
    val_idx = np.load(SPLITS / "val_indices.npy")

    print("=== Split integrity ===")
    print(f"train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(f"pairwise disjoint: "
          f"{not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))}")
    print(f"total unique rows: {len(set(train_idx) | set(val_idx) | set(test_idx))}")

    summary = []
    boundary = []
    worst_rows = []

    for target in TARGETS:
        target_dir = RESULTS / target
        pred_file = target_dir / "test_predictions.csv"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_file}")

        pred = pd.read_csv(pred_file)
        if len(pred) != len(test_idx):
            raise ValueError(f"{target}: expected {len(test_idx)} test predictions, got {len(pred)}")
        if not np.array_equal(pred["row_index"].to_numpy(dtype=int), test_idx):
            raise ValueError(f"{target}: prediction row indices do not match the saved test split.")

        y = pred["reference_fi"].to_numpy(dtype=float)
        p = pred["predicted_fi"].to_numpy(dtype=float)
        err = p - y
        abs_err = np.abs(err)
        near = (y >= BAND_LOW) & (y <= BAND_HIGH)
        true_fail = y >= THRESHOLD
        pred_fail = p >= THRESHOLD

        summary.append({
            "target": target,
            "test_MAE_rechecked": mean_absolute_error(y, p),
            "test_RMSE_rechecked": np.sqrt(mean_squared_error(y, p)),
            "test_R2_rechecked": r2_score(y, p),
            "median_abs_error": np.median(abs_err),
            "p95_abs_error": np.quantile(abs_err, 0.95),
            "p99_abs_error": np.quantile(abs_err, 0.99),
            "max_abs_error": np.max(abs_err),
            "over_threshold_disagreements": int(np.sum(true_fail != pred_fail)),
            "over_threshold_agreement": float(np.mean(true_fail == pred_fail)),
        })

        boundary.append({
            "target": target,
            "reference_FI_band": f"[{BAND_LOW:.2f}, {BAND_HIGH:.2f}]",
            "samples_in_band": int(near.sum()),
            "boundary_MAE": float(np.mean(abs_err[near])) if near.any() else np.nan,
            "boundary_RMSE": float(np.sqrt(np.mean(err[near] ** 2))) if near.any() else np.nan,
            "boundary_max_abs_error": float(np.max(abs_err[near])) if near.any() else np.nan,
            "boundary_threshold_disagreements": int(np.sum(true_fail[near] != pred_fail[near])),
            "boundary_threshold_accuracy": float(np.mean(true_fail[near] == pred_fail[near])) if near.any() else np.nan,
        })

        worst_indices = np.argsort(abs_err)[-5:][::-1]
        for rank, j in enumerate(worst_indices, start=1):
            worst_rows.append({
                "target": target,
                "rank_within_target": rank,
                "row_index": int(pred.iloc[j]["row_index"]),
                "reference_FI": float(y[j]),
                "predicted_FI": float(p[j]),
                "absolute_error": float(abs_err[j]),
                "reference_failed_FI_ge_1": int(true_fail[j]),
                "predicted_failed_FI_ge_1": int(pred_fail[j]),
                "near_threshold_band": bool(near[j]),
            })

    summary_df = pd.DataFrame(summary)
    boundary_df = pd.DataFrame(boundary)
    worst_df = pd.DataFrame(worst_rows)

    print("\n=== Rechecked overall test errors and largest deviations ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.7g}"))
    print("\n=== Explicit near-threshold audit (reference FI 0.90 to 1.10) ===")
    print(boundary_df.to_string(index=False, float_format=lambda x: f"{x:.7g}"))
    print("\n=== Five largest absolute errors per target ===")
    print(worst_df.to_string(index=False, float_format=lambda x: f"{x:.7g}"))

    # Save audit summaries separately; existing model artifacts are not touched.
    out_summary = RESULTS / "post_training_error_audit.csv"
    out_boundary = RESULTS / "near_threshold_audit.csv"
    out_worst = RESULTS / "largest_test_errors.csv"
    summary_df.to_csv(out_summary, index=False)
    boundary_df.to_csv(out_boundary, index=False)
    worst_df.to_csv(out_worst, index=False)
    print("\nAudit files written:")
    print(out_summary)
    print(out_boundary)
    print(out_worst)


if __name__ == "__main__":
    main()
