
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
RESULTS = ROOT / "results" / "multitarget_dnn_v3"

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

BANDS = [
    ("FI < 0.9", -np.inf, 0.9),
    ("0.9 <= FI <= 1.1", 0.9, 1.1),
    ("FI > 1.1", 1.1, np.inf),
]

data = pd.read_csv(DATA_PATH)
summary = []
boundary = []

for target in TARGETS:
    path = RESULTS / target / "test_predictions.csv"
    df = pd.read_csv(path)

    actual = df["reference_fi"].to_numpy(dtype=float)
    pred = df["predicted_fi"].to_numpy(dtype=float)

    if not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError(f"Non-finite values found: {target}")

    err = pred - actual
    abs_err = np.abs(err)
    actual_failed = actual >= 1.0
    pred_failed = pred >= 1.0

    false_safe = actual_failed & ~pred_failed
    false_failure = ~actual_failed & pred_failed

    # Overall threshold classification.
    failed_count = int(actual_failed.sum())
    summary.append({
        "target": target,
        "test_rows": len(actual),
        "actual_failed": failed_count,
        "false_safe": int(false_safe.sum()),
        "false_safe_rate_pct": (
            100 * false_safe.sum() / failed_count
            if failed_count else np.nan
        ),
        "false_failure": int(false_failure.sum()),
        "overall_MAE": float(abs_err.mean()),
        "overall_max_abs_error": float(abs_err.max()),
    })

    # Error and classification results by actual FI band.
    for band, lower, upper in BANDS:
        mask = (actual >= lower) & (actual <= upper)
        if not mask.any():
            continue

        failed_in_band = actual_failed[mask]
        fs = false_safe[mask]

        boundary.append({
            "target": target,
            "FI_band": band,
            "rows": int(mask.sum()),
            "MAE": float(abs_err[mask].mean()),
            "max_abs_error": float(abs_err[mask].max()),
            "actual_failed": int(failed_in_band.sum()),
            "false_safe": int(fs.sum()),
            "false_safe_rate_pct": (
                100 * fs.sum() / failed_in_band.sum()
                if failed_in_band.sum() else np.nan
            ),
            "false_failure": int(false_failure[mask].sum()),
        })

summary_df = pd.DataFrame(summary)
bands_df = pd.DataFrame(boundary)

OUT = RESULTS / "error_band_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(OUT / "overall_summary.csv", index=False)
bands_df.to_csv(OUT / "error_bands.csv", index=False)

print("\n=== OVERALL TEST-SET SUMMARY ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

print("\n=== RESULTS BY ACTUAL FAILURE-INDEX BAND ===")
print(bands_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# Verify how prediction rows map to original dataset rows.
print("\n=== ROW MAPPING CHECK ===")
for target in TARGETS:
    df = pd.read_csv(RESULTS / target / "test_predictions.csv")
    if "row_index" not in df.columns:
        print(f"{target}: no row_index column; mapping not verifiable")
        continue

    idx = df["row_index"].to_numpy(dtype=int)
    if idx.min() < 0 or idx.max() >= len(data):
        print(f"{target}: row_index is outside dataset bounds")
        continue

    # Compare stored reference FIs against the source dataset.
    expected = data.iloc[idx][target].to_numpy(dtype=float)
    actual = df["reference_fi"].to_numpy(dtype=float)
    matches = np.isclose(expected, actual, rtol=1e-6, atol=1e-6)

    print(
        f"{target}: reference mapping matches "
        f"{matches.sum()}/{len(matches)} rows"
    )

print(f"\nReports saved to: {OUT}")
print("No models or datasets were modified.")
