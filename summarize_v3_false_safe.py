
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "results" / "multitarget_dnn_v3"
OUT = PRED_DIR / "cross_criterion_diagnostics"

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

rows = []

for target in TARGETS:
    path = PRED_DIR / target / "test_predictions.csv"
    df = pd.read_csv(path)

    required = {
        "row_index",
        "reference_fi",
        "predicted_fi",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")

    actual = df["reference_fi"]
    predicted = df["predicted_fi"]

    false_safe = (actual >= 1.0) & (predicted < 1.0)
    selected = df.loc[false_safe].copy()

    for _, r in selected.iterrows():
        rows.append({
            "target": target,
            "row_index": int(r["row_index"]),
            "actual_fi": float(r["reference_fi"]),
            "predicted_fi": float(r["predicted_fi"]),
            "prediction_error": float(
                r["predicted_fi"] - r["reference_fi"]
            ),
            "actual_excess_above_threshold": float(
                r["reference_fi"] - 1.0
            ),
            "predicted_shortfall_below_threshold": float(
                1.0 - r["predicted_fi"]
            ),
        })

result = pd.DataFrame(rows)

if result.empty:
    print("No false-safe cases found.")
    raise SystemExit(0)

result["absolute_error"] = result["prediction_error"].abs()

result = result.sort_values(
    "predicted_shortfall_below_threshold",
    ascending=False,
)

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "all_test_false_safe_cases.csv"
result.to_csv(path, index=False)

print("Full-test false-safe analysis completed.")
print(f"False-safe criterion-row pairs: {len(result)}")
print(f"Saved report: {path}")

print("\nSummary by criterion:")
summary = result.groupby("target").agg(
    false_safe_cases=("row_index", "count"),
    mean_actual_fi=("actual_fi", "mean"),
    mean_predicted_fi=("predicted_fi", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
    maximum_absolute_error=("absolute_error", "max"),
    mean_threshold_shortfall=(
        "predicted_shortfall_below_threshold", "mean"
    ),
).round(6)

print(summary.to_string())

print("\nLargest false-safe prediction shortfalls:")
print(result.head(10).to_string(index=False))
