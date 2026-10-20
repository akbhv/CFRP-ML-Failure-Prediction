"""
Targeted error-versus-load-region analysis for the seven-target CFRP DNN run.

Run from the project root:
    python src/analyze_multitarget_error_regions.py

Read-only with respect to source data and trained models. Writes only new audit CSVs
under results/multitarget_dnn_v1/error_region_audit/.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v2_puck_corrected_exact.csv"
RESULTS_DIR = ROOT / "results" / "multitarget_dnn_v1"
OUTPUT_DIR = RESULTS_DIR / "error_region_audit_v2"

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]
THRESHOLD = 1.0
BOUNDARY_LOW = 0.90
BOUNDARY_HIGH = 1.10


def summarize_group(group: pd.DataFrame, group_name: str, group_value: object) -> dict:
    y = group["reference_fi"].to_numpy(dtype=float)
    p = group["predicted_fi"].to_numpy(dtype=float)
    error = p - y
    abs_error = np.abs(error)
    actual_failed = y >= THRESHOLD
    predicted_failed = p >= THRESHOLD
    return {
        "target": group_name,
        "group_value": str(group_value),
        "samples": int(len(group)),
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error": float(np.quantile(abs_error, 0.95)),
        "max_absolute_error": float(np.max(abs_error)),
        "threshold_disagreements": int(np.sum(actual_failed != predicted_failed)),
        "threshold_accuracy": float(np.mean(actual_failed == predicted_failed)),
        "reference_failed_count": int(np.sum(actual_failed)),
        "predicted_failed_count": int(np.sum(predicted_failed)),
        "boundary_samples_FI_0p90_to_1p10": int(np.sum((y >= BOUNDARY_LOW) & (y <= BOUNDARY_HIGH))),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(RESULTS_DIR)
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit directory: {OUTPUT_DIR}\n"
            "Move/rename that audit folder if you intentionally want a fresh run."
        )

    df = pd.read_csv(DATA_PATH)
    required = ["sampling_region", "load_multiplier"] + TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    region_rows = []
    multiplier_rows = []
    region_boundary_rows = []
    detailed_rows = []

    # Read predictions, check row alignment, and join only by saved row_index.
    for target in TARGETS:
        pred_path = RESULTS_DIR / target / "test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred = pd.read_csv(pred_path)

        if "row_index" not in pred.columns:
            raise ValueError(f"Missing row_index in {pred_path}")
        if pred["row_index"].duplicated().any():
            raise ValueError(f"Duplicate row_index values in {pred_path}")

        pred["row_index"] = pred["row_index"].astype(int)
        if ((pred["row_index"] < 0) | (pred["row_index"] >= len(df))).any():
            raise ValueError(f"Out-of-range row_index in {pred_path}")

        # Reference values in saved predictions must match the source dataset.
        source_values = df.iloc[pred["row_index"].to_numpy()][target].to_numpy(dtype=float)
        prediction_reference = pred["reference_fi"].to_numpy(dtype=float)
        if not np.allclose(prediction_reference, source_values, rtol=0.0, atol=5e-7):
            raise ValueError(
                f"Reference FI values in {pred_path} do not match the source dataset."
            )

        merged = pred.copy()
        merged["target"] = target
        merged["sampling_region"] = df.iloc[pred["row_index"].to_numpy()]["sampling_region"].to_numpy()
        merged["load_multiplier"] = df.iloc[pred["row_index"].to_numpy()]["load_multiplier"].to_numpy()
        merged["abs_error"] = np.abs(
            merged["predicted_fi"].to_numpy(dtype=float)
            - merged["reference_fi"].to_numpy(dtype=float)
        )
        merged["signed_error"] = (
            merged["predicted_fi"].to_numpy(dtype=float)
            - merged["reference_fi"].to_numpy(dtype=float)
        )
        merged["near_threshold"] = (
            (merged["reference_fi"] >= BOUNDARY_LOW)
            & (merged["reference_fi"] <= BOUNDARY_HIGH)
        )
        merged["threshold_disagreement"] = (
            (merged["reference_fi"] >= THRESHOLD)
            != (merged["predicted_fi"] >= THRESHOLD)
        )
        detailed_rows.append(merged)

        # Error by the dataset's original sampling-region categories.
        for region, group in merged.groupby("sampling_region", dropna=False):
            record = summarize_group(group, target, region)
            record["sampling_region"] = str(region)
            region_rows.append(record)

            near = group[group["near_threshold"]]
            if len(near):
                near_record = summarize_group(near, target, region)
                near_record["sampling_region"] = str(region)
                region_boundary_rows.append(near_record)

        # Load multiplier buckets use fixed-width bins, with endpoints included.
        # The bin edges are quantile-informed only for reporting, not model fitting.
        multiplier = merged["load_multiplier"].to_numpy(dtype=float)
        if not np.isfinite(multiplier).all():
            raise ValueError(f"Non-finite load_multiplier values in predictions for {target}")

        try:
            bins = pd.qcut(
                merged["load_multiplier"],
                q=min(5, merged["load_multiplier"].nunique()),
                duplicates="drop",
            )
        except ValueError:
            bins = pd.Series(["all"] * len(merged), index=merged.index)

        merged_for_bins = merged.copy()
        merged_for_bins["load_multiplier_bin"] = bins.astype(str)
        for bin_label, group in merged_for_bins.groupby("load_multiplier_bin", dropna=False, sort=False):
            record = summarize_group(group, target, bin_label)
            record["load_multiplier_bin"] = str(bin_label)
            record["load_multiplier_min"] = float(group["load_multiplier"].min())
            record["load_multiplier_median"] = float(group["load_multiplier"].median())
            record["load_multiplier_max"] = float(group["load_multiplier"].max())
            multiplier_rows.append(record)

    all_predictions = pd.concat(detailed_rows, ignore_index=True)
    by_region = pd.DataFrame(region_rows)
    by_multiplier = pd.DataFrame(multiplier_rows)
    by_region_boundary = pd.DataFrame(region_boundary_rows)

    # List the largest absolute errors across all targets with row metadata.
    worst = all_predictions.sort_values("abs_error", ascending=False).groupby("target", sort=False).head(10)
    worst = worst[
        [
            "target", "row_index", "sampling_region", "load_multiplier",
            "reference_fi", "predicted_fi", "signed_error", "abs_error",
            "reference_failed_fi_ge_1", "predicted_failed_fi_ge_1",
            "near_threshold", "threshold_disagreement",
        ]
    ].sort_values(["target", "abs_error"], ascending=[True, False])

    by_region.to_csv(OUTPUT_DIR / "errors_by_sampling_region.csv", index=False)
    by_multiplier.to_csv(OUTPUT_DIR / "errors_by_load_multiplier_bin.csv", index=False)
    by_region_boundary.to_csv(OUTPUT_DIR / "near_threshold_errors_by_region.csv", index=False)
    worst.to_csv(OUTPUT_DIR / "largest_errors_with_region.csv", index=False)

    print("=== Audit scope ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Test predictions checked: {len(all_predictions)} target-row predictions")
    print(f"Targets: {len(TARGETS)}")
    print("Source FI values match saved test references: yes")
    print(f"Sampling regions: {sorted(df['sampling_region'].dropna().astype(str).unique().tolist())}")

    print("\n=== MAE by sampling region ===")
    print(
        by_region.pivot(index="target", columns="sampling_region", values="MAE")
        .to_string(float_format=lambda x: f"{x:.6g}")
    )

    print("\n=== Threshold disagreements by sampling region ===")
    print(
        by_region.pivot(index="target", columns="sampling_region", values="threshold_disagreements")
        .to_string()
    )

    print("\n=== MAE by load-multiplier bin ===")
    print(
        by_multiplier[
            ["target", "load_multiplier_bin", "samples", "MAE", "RMSE",
             "threshold_disagreements", "threshold_accuracy",
             "load_multiplier_min", "load_multiplier_median", "load_multiplier_max"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\n=== Near-threshold MAE by sampling region (reference FI 0.90–1.10) ===")
    if len(by_region_boundary):
        print(
            by_region_boundary.pivot(
                index="target", columns="sampling_region", values="MAE"
            ).to_string(float_format=lambda x: f"{x:.6g}")
        )
    else:
        print("No samples in the specified reference-FI band.")

    print("\n=== Largest errors with region metadata (top rows) ===")
    print(worst.head(25).to_string(index=False, float_format=lambda x: f"{x:.7g}"))

    print(f"\nAudit CSVs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
