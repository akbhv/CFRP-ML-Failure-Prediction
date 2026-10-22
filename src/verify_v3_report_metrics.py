
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multitarget_dnn_v3"

targets = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

saved_reg = pd.read_csv(RESULTS / "regression_metrics.csv").set_index("target")
saved_cls = pd.read_csv(RESULTS / "failure_boundary_metrics.csv").set_index("target")

rows = []

for target in targets:
    pred_path = RESULTS / target / "test_predictions.csv"
    df = pd.read_csv(pred_path)

    actual = df["reference_fi"].to_numpy(dtype=float)
    predicted = df["predicted_fi"].to_numpy(dtype=float)

    y_true = actual >= 1.0
    y_pred = predicted >= 1.0

    metrics = {
        "target": target,
        "rows": len(df),
        "MAE_recomputed": mean_absolute_error(actual, predicted),
        "RMSE_recomputed": np.sqrt(mean_squared_error(actual, predicted)),
        "R2_recomputed": r2_score(actual, predicted),
        "accuracy_recomputed": accuracy_score(y_true, y_pred),
        "precision_recomputed": precision_score(y_true, y_pred, zero_division=0),
        "recall_recomputed": recall_score(y_true, y_pred, zero_division=0),
        "F1_recomputed": f1_score(y_true, y_pred, zero_division=0),
        "false_safe_count": int(np.sum(y_true & ~y_pred)),
    }

    checks = {
        "MAE_match": np.isclose(metrics["MAE_recomputed"], saved_reg.loc[target, "test_MAE"], atol=1e-8),
        "RMSE_match": np.isclose(metrics["RMSE_recomputed"], saved_reg.loc[target, "test_RMSE"], atol=1e-8),
        "R2_match": np.isclose(metrics["R2_recomputed"], saved_reg.loc[target, "test_R2"], atol=1e-8),
        "accuracy_match": np.isclose(metrics["accuracy_recomputed"], saved_cls.loc[target, "accuracy"], atol=1e-8),
        "F1_match": np.isclose(metrics["F1_recomputed"], saved_cls.loc[target, "F1"], atol=1e-8),
    }

    metrics.update(checks)
    metrics["all_checks_pass"] = all(checks.values())
    rows.append(metrics)

out = pd.DataFrame(rows)
out_path = RESULTS / "report_metric_verification.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print(f"\nAll targets pass: {out['all_checks_pass'].all()}")
print(f"Saved verification: {out_path}")
