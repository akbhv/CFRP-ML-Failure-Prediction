
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
PRED_DIR = ROOT / "results" / "multitarget_dnn_v3"
DIAG_DIR = PRED_DIR / "cross_criterion_diagnostics"

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

data = pd.read_csv(DATA_PATH)
shared = pd.read_csv(DIAG_DIR / "multi_criterion_false_safe.csv")

# Build one aligned table of predictions by original dataset row_index.
tables = []

for target in TARGETS:
    path = PRED_DIR / target / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    pred = pd.read_csv(path)

    required = {"row_index", "reference_fi", "predicted_fi"}
    if not required.issubset(pred.columns):
        raise ValueError(
            f"{path} must contain columns: {sorted(required)}. "
            f"Found: {list(pred.columns)}"
        )

    pred = pred[["row_index", "reference_fi", "predicted_fi"]].copy()
    pred = pred.rename(columns={
        "reference_fi": f"{target}_actual",
        "predicted_fi": f"{target}_predicted",
    })
    tables.append(pred)

merged = tables[0]
for table in tables[1:]:
    merged = merged.merge(table, on="row_index", how="inner", validate="one_to_one")

if merged["row_index"].nunique() != 3000:
    raise ValueError(
        f"Expected 3000 common test rows, found {merged['row_index'].nunique()}"
    )

# Confirm saved prediction actuals and loads match the source dataset.
for target in TARGETS:
    actual_col = f"{target}_actual"
    source_col = target

    expected = data.iloc[
        merged["row_index"].astype(int).to_numpy()
    ][source_col].to_numpy()

    if not np.allclose(
        merged[actual_col].to_numpy(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(f"Actual values do not align for {target}")

load_values = data.iloc[
    shared["row_index"].astype(int).to_numpy()
][LOADS].reset_index(drop=True)

if not np.allclose(
    shared[LOADS].to_numpy(),
    load_values.to_numpy(),
    rtol=1e-8,
    atol=1e-8,
):
    raise ValueError("Shared false-safe CSV has mismatched load values")

# Keep the 16 shared cases and calculate criterion-level errors.
case = shared[["row_index", "false_safe_count"]].merge(
    merged, on="row_index", how="left", validate="one_to_one"
)
case = case.merge(
    data[["failed"] + TARGETS].reset_index().rename(columns={"index": "row_index"}),
    on="row_index",
    how="left",
    validate="one_to_one",
)

for target in TARGETS:
    actual_col = f"{target}_actual"
    pred_col = f"{target}_predicted"

    # False-safe: actual FI >= 1, predicted FI < 1.
    case[f"{target}_error"] = case[pred_col] - case[actual_col]
    case[f"{target}_false_safe"] = (
        (case[actual_col] >= 1.0) & (case[pred_col] < 1.0)
    )
    case[f"{target}_threshold_margin"] = case[pred_col] - 1.0

    # Positive shortfall means the prediction is below the threshold.
    case[f"{target}_shortfall"] = np.maximum(0.0, 1.0 - case[pred_col])

# Save detailed cases with loads and all criterion predictions.
case = case.merge(
    shared[["row_index"] + LOADS],
    on="row_index",
    how="left",
    validate="one_to_one",
)
case = case.sort_values(
    ["false_safe_count", "row_index"],
    ascending=[False, True],
)

DIAG_DIR.mkdir(parents=True, exist_ok=True)
detail_path = DIAG_DIR / "shared_false_safe_investigation.csv"
case.to_csv(detail_path, index=False)

# Compact report: only criteria that falsely predicted safe for each row.
summary_rows = []
for _, row in case.iterrows():
    affected = [
        target for target in TARGETS
        if bool(row[f"{target}_false_safe"])
    ]

    summary_rows.append({
        "row_index": int(row["row_index"]),
        "false_safe_count": int(row["false_safe_count"]),
        "false_safe_criteria": ", ".join(affected),
        "loads": "; ".join(
            f"{load}={row[load]:.8g}" for load in LOADS
        ),
        "actual_FI_values": "; ".join(
            f"{target}={row[f'{target}_actual']:.6f}"
            for target in affected
        ),
        "predicted_FI_values": "; ".join(
            f"{target}={row[f'{target}_predicted']:.6f}"
            for target in affected
        ),
        "threshold_shortfalls": "; ".join(
            f"{target}={row[f'{target}_shortfall']:.6f}"
            for target in affected
        ),
    })

summary_path = DIAG_DIR / "shared_false_safe_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print("Investigation completed.")
print(f"Shared cases checked: {len(case)}")
print(f"Detailed report: {detail_path}")
print(f"Compact report:  {summary_path}")
print("\nFalse-safe counts within these shared cases:")
for target in TARGETS:
    print(f"  {target}: {int(case[f'{target}_false_safe'].sum())}")
