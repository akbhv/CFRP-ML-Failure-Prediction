import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed/ml")
MODEL_DIR = Path("results/architecture_comparison")
OUTPUT_DIR = Path("results/regression_error_analysis")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

y_actual = np.load(
    DATA_DIR / "y_reg_test.npy"
)

y_pred = np.load(
    MODEL_DIR / "baseline_y_pred.npy"
)


# ============================================================
# BASIC CHECK
# ============================================================

if len(y_actual) != len(y_pred):
    raise ValueError(
        "Actual and predicted arrays have different lengths."
    )


# ============================================================
# ERROR CALCULATIONS
# ============================================================

errors = y_pred - y_actual
absolute_errors = np.abs(errors)


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


print("-----------------------------------")
print("REGRESSION ERROR ANALYSIS")
print("-----------------------------------")

print()
print(f"Test samples : {len(y_actual)}")
print(f"MAE          : {mae:.6f}")
print(f"RMSE         : {rmse:.6f}")
print(f"R²           : {r2:.6f}")

print()
print(f"Mean error   : {np.mean(errors):.6f}")
print(f"Std error    : {np.std(errors):.6f}")
print(f"Max abs error: {np.max(absolute_errors):.6f}")
print(f"Median abs error: {np.median(absolute_errors):.6f}")


# ============================================================
# ERROR PERCENTILES
# ============================================================

percentiles = [50, 75, 90, 95, 99]

print()
print("-----------------------------------")
print("ABSOLUTE ERROR PERCENTILES")
print("-----------------------------------")

for p in percentiles:

    value = np.percentile(
        absolute_errors,
        p
    )

    print(
        f"{p:>2}th percentile : "
        f"{value:.6f}"
    )


# ============================================================
# ERROR BY HASHIN FI RANGE
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


error_by_band = []

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

    band_max_error = np.max(
        absolute_errors[mask]
    )

    error_by_band.append({
        "FI_range": label,
        "samples": count,
        "MAE": band_mae,
        "RMSE": band_rmse,
        "Max_abs_error": band_max_error
    })


error_by_band_df = pd.DataFrame(
    error_by_band
)


print()
print("-----------------------------------")
print("ERROR BY HASHIN FI RANGE")
print("-----------------------------------")

print(
    error_by_band_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE ERROR TABLE
# ============================================================

error_by_band_df.to_csv(
    OUTPUT_DIR /
    "error_by_fi_range.csv",
    index=False
)


# ============================================================
# NEAR-BOUNDARY ANALYSIS
# ============================================================

dist_from_boundary = np.abs(
    y_actual - 1.0
)

boundary_limits = [
    0.01,
    0.02,
    0.05,
    0.10
]

boundary_results = []

print()
print("-----------------------------------")
print("NEAR-BOUNDARY ERROR ANALYSIS")
print("-----------------------------------")

for limit in boundary_limits:

    mask = (
        dist_from_boundary <= limit
    )

    count = np.sum(mask)

    if count == 0:
        continue

    boundary_mae = mean_absolute_error(
        y_actual[mask],
        y_pred[mask]
    )

    boundary_rmse = np.sqrt(
        mean_squared_error(
            y_actual[mask],
            y_pred[mask]
        )
    )

    boundary_results.append({
        "distance_from_FI_1": limit,
        "samples": count,
        "MAE": boundary_mae,
        "RMSE": boundary_rmse
    })

    print(
        f"|FI-1| <= {limit:.2f}: "
        f"{count} samples, "
        f"MAE = {boundary_mae:.6f}, "
        f"RMSE = {boundary_rmse:.6f}"
    )


boundary_df = pd.DataFrame(
    boundary_results
)

boundary_df.to_csv(
    OUTPUT_DIR /
    "near_boundary_analysis.csv",
    index=False
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

worst_indices = np.argsort(
    absolute_errors
)[-20:][::-1]

worst_predictions = pd.DataFrame({
    "actual_FI": y_actual[worst_indices],
    "predicted_FI": y_pred[worst_indices],
    "error": errors[worst_indices],
    "absolute_error": absolute_errors[worst_indices]
})


print()
print("-----------------------------------")
print("20 LARGEST PREDICTION ERRORS")
print("-----------------------------------")

print(
    worst_predictions.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

worst_predictions.to_csv(
    OUTPUT_DIR /
    "worst_predictions.csv",
    index=False
)


# ============================================================
# PLOT 1 — ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.5,
    s=15
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
    "Actual Hashin Failure Index"
)

plt.ylabel(
    "Predicted Hashin Failure Index"
)

plt.title(
    "Actual vs Predicted Hashin Failure Index"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 — RESIDUALS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    y_actual,
    errors,
    alpha=0.5,
    s=15
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Actual Hashin Failure Index"
)

plt.ylabel(
    "Prediction Error"
)

plt.title(
    "Regression Residuals"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "residuals.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 — ABSOLUTE ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    absolute_errors,
    bins=50
)

plt.xlabel(
    "Absolute Prediction Error"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Distribution of Absolute Prediction Error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "absolute_error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("-----------------------------------")
print("ERROR ANALYSIS COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: {OUTPUT_DIR}"
)