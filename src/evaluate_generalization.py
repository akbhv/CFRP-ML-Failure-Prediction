import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/unseen/"
    "unseen_generalization_dataset.csv"
)

SCALER_FILE = Path(
    "data/processed/ml/feature_scaler.pkl"
)

MODEL_FILE = Path(
    "results/architecture_comparison/"
    "baseline_model.keras"
)

OUTPUT_DIR = Path(
    "results/generalization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


feature_columns = [
    "Nx",
    "Ny",
    "Nxy",
    "Mx",
    "My",
    "Mxy"
]


X = df[
    feature_columns
].values

y_actual = df[
    "hashin_fi"
].values


# ============================================================
# LOAD SCALER
# ============================================================

import joblib

scaler = joblib.load(
    SCALER_FILE
)


X_scaled = scaler.transform(
    df[feature_columns]
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

print("-----------------------------------")
print("GENERALIZATION EVALUATION")
print("-----------------------------------")

print()
print(
    f"Samples: {len(X)}"
)

print(
    f"Model: {MODEL_FILE}"
)

print()

model = load_model(
    MODEL_FILE
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_scaled,
    verbose=0
).flatten()


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

r2 = r2_score(
    y_actual,
    y_pred
)

errors = (
    y_pred - y_actual
)

absolute_errors = np.abs(
    errors
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("-----------------------------------")
print("GENERALIZATION RESULTS")
print("-----------------------------------")

print()

print(
    f"MAE          : {mae:.6f}"
)

print(
    f"RMSE         : {rmse:.6f}"
)

print(
    f"R²           : {r2:.6f}"
)

print()

print(
    f"Mean error   : "
    f"{np.mean(errors):.6f}"
)

print(
    f"Median abs error : "
    f"{np.median(absolute_errors):.6f}"
)

print(
    f"Max abs error : "
    f"{np.max(absolute_errors):.6f}"
)


# ============================================================
# ERROR PERCENTILES
# ============================================================

print()

print("-----------------------------------")
print("ABSOLUTE ERROR PERCENTILES")
print("-----------------------------------")

for percentile in [
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        absolute_errors,
        percentile
    )

    print(
        f"{percentile:>2}th percentile : "
        f"{value:.6f}"
    )


# ============================================================
# ERROR BY FI RANGE
# ============================================================

bins = [
    -np.inf,
    0.50,
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.20,
    1.50,
    2.00,
    np.inf
]

labels = [
    "<0.50",
    "0.50–0.80",
    "0.80–0.90",
    "0.90–0.95",
    "0.95–1.00",
    "1.00–1.05",
    "1.05–1.10",
    "1.10–1.20",
    "1.20–1.50",
    "1.50–2.00",
    "≥2.00"
]


fi_bins = pd.cut(
    y_actual,
    bins=bins,
    labels=labels
)


band_results = []


for label in labels:

    mask = (
        fi_bins == label
    )

    count = np.sum(mask)

    if count == 0:
        continue

    band_mae = mean_absolute_error(
        y_actual[mask],
        y_pred[mask]
    )

    band_rmse = np.sqrt(
        mean_squared_error(
            y_actual[mask],
            y_pred[mask]
        )
    )

    band_results.append({
        "FI_range": label,
        "samples": count,
        "MAE": band_mae,
        "RMSE": band_rmse
    })


band_df = pd.DataFrame(
    band_results
)


print()

print("-----------------------------------")
print("ERROR BY HASHIN FI RANGE")
print("-----------------------------------")

print(
    band_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

band_df.to_csv(
    OUTPUT_DIR /
    "generalization_error_by_fi_range.csv",
    index=False
)


results_summary = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2",
        "Mean Error",
        "Median Absolute Error",
        "Maximum Absolute Error"
    ],

    "value": [
        mae,
        rmse,
        r2,
        np.mean(errors),
        np.median(absolute_errors),
        np.max(absolute_errors)
    ]
})


results_summary.to_csv(
    OUTPUT_DIR /
    "generalization_metrics.csv",
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = df.copy()

prediction_df[
    "predicted_hashin_fi"
] = y_pred

prediction_df[
    "prediction_error"
] = errors

prediction_df[
    "absolute_error"
] = absolute_errors


prediction_df.to_csv(
    OUTPUT_DIR /
    "generalization_predictions.csv",
    index=False
)


# ============================================================
# PLOT — ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.5,
    s=12
)

min_value = min(
    y_actual.min(),
    y_pred.min()
)

max_value = max(
    y_actual.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Physics Hashin Failure Index"
)

plt.ylabel(
    "DNN Predicted Failure Index"
)

plt.title(
    "DNN Generalization: Actual vs Predicted Hashin FI"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "generalization_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT — RESIDUALS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    y_actual,
    errors,
    alpha=0.5,
    s=12
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Physics Hashin Failure Index"
)

plt.ylabel(
    "Prediction Error"
)

plt.title(
    "DNN Generalization Residuals"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "generalization_residuals.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print()

print("-----------------------------------")
print("GENERALIZATION EVALUATION COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: "
    f"{OUTPUT_DIR}"
)