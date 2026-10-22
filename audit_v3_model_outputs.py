
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "multitarget_dnn_v3"

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

records = []

for target in TARGETS:
    folder = OUT / target
    model_path = folder / f"{target}_model.keras"
    pred_path = folder / "test_predictions.csv"

    if not model_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            f"Missing model or predictions for {target}: {folder}"
        )

    df = pd.read_csv(pred_path)
    actual = df["reference_fi"].to_numpy(dtype=float)
    predicted = df["predicted_fi"].to_numpy(dtype=float)

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"Non-finite predictions found for {target}")

    boundary = (actual >= 0.9) & (actual <= 1.1)
    a = actual[boundary]
    p = predicted[boundary]

    if len(a) == 0:
        raise ValueError(f"No near-boundary samples for {target}")

    actual_failed = a >= 1.0
    predicted_failed = p >= 1.0

    records.append({
        "target": target,
        "test_rows": len(actual),
        "near_boundary_rows": len(a),
        "near_boundary_MAE": mean_absolute_error(a, p),
        "near_boundary_RMSE": np.sqrt(mean_squared_error(a, p)),
        "near_boundary_accuracy": accuracy_score(
            actual_failed, predicted_failed
        ),
        "near_boundary_precision": precision_score(
            actual_failed, predicted_failed, zero_division=0
        ),
        "near_boundary_recall": recall_score(
            actual_failed, predicted_failed, zero_division=0
        ),
        "near_boundary_F1": f1_score(
            actual_failed, predicted_failed, zero_division=0
        ),
        "false_safe": int((actual_failed & ~predicted_failed).sum()),
        "false_failure": int((~actual_failed & predicted_failed).sum()),
        "max_absolute_error": float(np.max(np.abs(actual - predicted))),
    })

result = pd.DataFrame(records)
result.to_csv(OUT / "near_boundary_audit.csv", index=False)

print("=== V3 MODEL OUTPUT AUDIT ===")
print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"\nSaved: {OUT / 'near_boundary_audit.csv'}")
print("\nAll seven model files and prediction files exist.")
