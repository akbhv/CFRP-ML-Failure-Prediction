
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "multitarget_dnn_v3"
OUT = RESULTS / "cross_criterion_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

LOADS = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
data_path = ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
data = pd.read_csv(data_path)

# Each row_index identifies the source-dataset row.
all_rows = None

for target in TARGETS:
    pred_path = RESULTS / target / "test_predictions.csv"
    pred = pd.read_csv(pred_path)

    required = {"row_index", "reference_fi", "predicted_fi"}
    if not required.issubset(pred.columns):
        raise ValueError(f"{pred_path} must contain {required}")

    part = pred[["row_index", "reference_fi", "predicted_fi"]].copy()
    part["row_index"] = part["row_index"].astype(int)

    if part["row_index"].duplicated().any():
        raise ValueError(f"Duplicate row_index values in {target}")

    # Verify saved reference values against the original dataset.
    source = data.iloc[part["row_index"].to_numpy()][target].to_numpy()
    saved = part["reference_fi"].to_numpy(dtype=float)

    if not np.all(np.isclose(source, saved, rtol=1e-6, atol=1e-6)):
        raise ValueError(f"Source-reference mismatch in {target}")

    part = part.rename(columns={
        "reference_fi": f"{target}__actual",
        "predicted_fi": f"{target}__predicted",
    })

    all_rows = (
        part if all_rows is None
        else all_rows.merge(part, on="row_index", how="inner", validate="one_to_one")
    )

if len(all_rows) != 3000:
    raise ValueError(
        f"Expected 3000 common test rows, found {len(all_rows)}"
    )

# False-safe means actual FI >= 1 but predicted FI < 1.
# False-failure means actual FI < 1 but predicted FI >= 1.
for target in TARGETS:
    actual = all_rows[f"{target}__actual"]
    predicted = all_rows[f"{target}__predicted"]

    all_rows[f"{target}__error"] = predicted - actual
    all_rows[f"{target}__false_safe"] = (
        (actual >= 1.0) & (predicted < 1.0)
    )
    all_rows[f"{target}__false_failure"] = (
        (actual < 1.0) & (predicted >= 1.0)
    )
    all_rows[f"{target}__near_boundary"] = (
        (actual >= 0.9) & (actual <= 1.1)
    )

false_safe_cols = [f"{t}__false_safe" for t in TARGETS]
false_failure_cols = [f"{t}__false_failure" for t in TARGETS]

all_rows["false_safe_count"] = all_rows[false_safe_cols].sum(axis=1)
all_rows["false_failure_count"] = all_rows[false_failure_cols].sum(axis=1)

# Save samples with false-safe errors in multiple criteria.
multi_safe = all_rows[all_rows["false_safe_count"] >= 2].copy()
multi_safe = multi_safe.sort_values(
    ["false_safe_count", "row_index"], ascending=[False, True]
)

# Include load components for engineering interpretation.
load_data = (
    data.iloc[all_rows["row_index"].to_numpy()][LOADS]
    .reset_index(drop=True)
)

all_rows = all_rows.reset_index(drop=True)

all_rows[LOADS] = data.iloc[
    all_rows["row_index"].astype(int).to_numpy()
][LOADS].to_numpy()

multi_safe = all_rows[
    all_rows["false_safe_count"] >= 2
].copy()

multi_safe = multi_safe.sort_values(
    ["false_safe_count", "row_index"],
    ascending=[False, True]
).reset_index(drop=True)

multi_safe.to_csv(
    OUT / "multi_criterion_false_safe.csv",
    index=False
)
# Per-criterion error concentration and boundary statistics.
summary = []
for target in TARGETS:
    actual = all_rows[f"{target}__actual"].to_numpy()
    predicted = all_rows[f"{target}__predicted"].to_numpy()
    error = predicted - actual
    near = (actual >= 0.9) & (actual <= 1.1)
    fs = (actual >= 1.0) & (predicted < 1.0)
    ff = (actual < 1.0) & (predicted >= 1.0)

    summary.append({
        "target": target,
        "test_rows": len(actual),
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "max_absolute_error": np.max(np.abs(error)),
        "false_safe": int(fs.sum()),
        "false_failure": int(ff.sum()),
        "near_boundary_rows": int(near.sum()),
        "near_boundary_MAE": (
            np.mean(np.abs(error[near])) if near.any() else np.nan
        ),
        "near_boundary_false_safe": int((fs & near).sum()),
    })

pd.DataFrame(summary).to_csv(OUT / "criterion_error_summary.csv", index=False)

# Load-case level error summary.
error_cols = [f"{t}__error" for t in TARGETS]
all_rows["mean_absolute_error_across_criteria"] = (
    all_rows[error_cols].abs().mean(axis=1)
)
all_rows["maximum_absolute_error_across_criteria"] = (
    all_rows[error_cols].abs().max(axis=1)
)

all_rows.sort_values(
    "maximum_absolute_error_across_criteria", ascending=False
).to_csv(OUT / "largest_cross_criterion_errors.csv", index=False)

print("\n=== CROSS-CRITERION FALSE-SAFE OVERLAP ===")
print(f"Common test rows: {len(all_rows)}")
print(f"Rows false-safe in 2+ criteria: {len(multi_safe)}")

if len(multi_safe):
    print("\nFalse-safe counts across criteria:")
    print(multi_safe["false_safe_count"].value_counts().sort_index(ascending=False))

    print("\nMost frequently shared false-safe cases:")
    show_cols = ["row_index", "false_safe_count", *LOADS]
    print(multi_safe[show_cols].head(15).to_string(index=False))
else:
    print("No test row is false-safe in two or more criteria.")

print("\n=== PER-CRITERION ERROR SUMMARY ===")
print(pd.DataFrame(summary).round(6).to_string(index=False))

print("\n=== OUTPUT FILES ===")
print(OUT / "multi_criterion_false_safe.csv")
print(OUT / "criterion_error_summary.csv")
print(OUT / "largest_cross_criterion_errors.csv")
print("\nDiagnostic complete. Models and API were not modified.")
