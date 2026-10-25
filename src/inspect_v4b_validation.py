"""Inspect the worst V4B-vs-classical validation cases.

Run from project root:
    python -m src.inspect_v4b_validation

Read-only: prints diagnostics from results/validation_v4b_vs_classical/case_predictions.csv.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results" / "validation_v4b_vs_classical" / "case_predictions.csv"
TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]
LOAD_COLUMNS = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Validation report not found: {REPORT_PATH}")

    df = pd.read_csv(REPORT_PATH)
    required = [
        "case_id", "sampling_region", "target", "classical_fi",
        "predicted_fi", "absolute_error", "classical_failed",
        "predicted_failed", *LOAD_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Validation report is missing columns: {missing}")

    print(f"Report: {REPORT_PATH}")
    print(f"Rows: {len(df)}\n")

    print("=== Worst 15 absolute errors across all criteria ===")
    cols = [
        "case_id", "sampling_region", "target", "classical_fi",
        "predicted_fi", "absolute_error", "classical_failed",
        "predicted_failed", *LOAD_COLUMNS,
    ]
    print(
        df.sort_values("absolute_error", ascending=False)[cols]
        .head(15)
        .to_string(index=False, float_format=lambda value: f"{value:.7g}")
    )

    false_safe = df[
        (df["classical_failed"] == 1) & (df["predicted_failed"] == 0)
    ]
    print("\n=== False-safe cases (classical FAILURE, ML SAFE) ===")
    if false_safe.empty:
        print("None in this validation set.")
    else:
        print(
            false_safe[cols]
            .sort_values("absolute_error", ascending=False)
            .to_string(index=False, float_format=lambda value: f"{value:.7g}")
        )

    false_failure = df[
        (df["classical_failed"] == 0) & (df["predicted_failed"] == 1)
    ]
    print("\n=== False-failure cases (classical SAFE, ML FAILURE) ===")
    if false_failure.empty:
        print("None in this validation set.")
    else:
        print(
            false_failure[cols]
            .sort_values("absolute_error", ascending=False)
            .to_string(index=False, float_format=lambda value: f"{value:.7g}")
        )

    print("\n=== Worst error by criterion ===")
    worst_by_target = (
        df.sort_values("absolute_error", ascending=False)
        .groupby("target", sort=False)
        .head(1)
        .sort_values("target")
    )
    print(
        worst_by_target[
            ["target", "case_id", "sampling_region", "classical_fi",
             "predicted_fi", "absolute_error", *LOAD_COLUMNS]
        ].to_string(index=False, float_format=lambda value: f"{value:.7g}")
    )

    print("\n=== Error summary by sampling region ===")
    region_summary = (
        df.groupby(["target", "sampling_region"], sort=False)["absolute_error"]
        .agg(["count", "mean", "max"])
        .reset_index()
    )
    print(
        region_summary.to_string(index=False, float_format=lambda value: f"{value:.7g}")
    )


if __name__ == "__main__":
    main()
