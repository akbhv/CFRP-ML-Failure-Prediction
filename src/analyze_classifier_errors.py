import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path("data/processed/ml")
RESULTS_DIR = Path("results/dnn_classifier")

OUTPUT_DIR = RESULTS_DIR / "error_analysis"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

y_test = np.load(
    RESULTS_DIR / "y_test.npy"
)

y_probability = np.load(
    RESULTS_DIR / "y_probability.npy"
)

y_pred = np.load(
    RESULTS_DIR / "y_pred.npy"
)

# Hashin FI is the regression target.
# We need the original test FI values.
#
# These were saved separately by the regression pipeline.

regression_y_test_path = Path(
    "results/dnn_regression/y_test.npy"
)

if regression_y_test_path.exists():
    hashin_fi = np.load(
        regression_y_test_path
    )
else:
    raise FileNotFoundError(
        "Could not find results/dnn_regression/y_test.npy"
    )


# ==================================================
# BASIC CHECK
# ==================================================

if not (
    len(y_test)
    == len(y_probability)
    == len(y_pred)
    == len(hashin_fi)
):
    raise ValueError(
        "Prediction and test arrays have different lengths."
    )


# ==================================================
# ERROR MASKS
# ==================================================

false_positive = (
    (y_test == 0)
    & (y_pred == 1)
)

false_negative = (
    (y_test == 1)
    & (y_pred == 0)
)

correct_prediction = (
    y_test == y_pred
)


# ==================================================
# SUMMARY
# ==================================================

print("-----------------------------------")
print("CLASSIFIER ERROR ANALYSIS")
print("-----------------------------------")

print()
print(f"Total test samples : {len(y_test)}")

print(
    f"Correct predictions: "
    f"{correct_prediction.sum()}"
)

print(
    f"False positives    : "
    f"{false_positive.sum()}"
)

print(
    f"False negatives    : "
    f"{false_negative.sum()}"
)

print(
    f"Total errors       : "
    f"{(~correct_prediction).sum()}"
)


# ==================================================
# ERROR FI DISTRIBUTIONS
# ==================================================

fp_fi = hashin_fi[false_positive]
fn_fi = hashin_fi[false_negative]

print()
print("-----------------------------------")
print("FALSE POSITIVES")
print("-----------------------------------")

if len(fp_fi) > 0:
    print(
        f"Minimum FI : {fp_fi.min():.6f}"
    )
    print(
        f"Maximum FI : {fp_fi.max():.6f}"
    )
    print(
        f"Mean FI    : {fp_fi.mean():.6f}"
    )
    print(
        f"Median FI  : {np.median(fp_fi):.6f}"
    )

print()
print("-----------------------------------")
print("FALSE NEGATIVES")
print("-----------------------------------")

if len(fn_fi) > 0:
    print(
        f"Minimum FI : {fn_fi.min():.6f}"
    )
    print(
        f"Maximum FI : {fn_fi.max():.6f}"
    )
    print(
        f"Mean FI    : {fn_fi.mean():.6f}"
    )
    print(
        f"Median FI  : {np.median(fn_fi):.6f}"
    )


# ==================================================
# DISTANCE FROM FAILURE BOUNDARY
# ==================================================

distance_from_boundary = np.abs(
    hashin_fi - 1.0
)

error_mask = ~correct_prediction

error_distance = (
    distance_from_boundary[error_mask]
)

print()
print("-----------------------------------")
print("ERROR DISTANCE FROM FI = 1")
print("-----------------------------------")

print(
    f"Minimum distance : "
    f"{error_distance.min():.6f}"
)

print(
    f"Maximum distance : "
    f"{error_distance.max():.6f}"
)

print(
    f"Mean distance    : "
    f"{error_distance.mean():.6f}"
)

print(
    f"Median distance  : "
    f"{np.median(error_distance):.6f}"
)


# ==================================================
# BOUNDARY BANDS
# ==================================================

bands = [
    ("FI < 0.80", hashin_fi < 0.80),

    ("0.80 <= FI < 0.90",
     (hashin_fi >= 0.80)
     & (hashin_fi < 0.90)),

    ("0.90 <= FI < 0.95",
     (hashin_fi >= 0.90)
     & (hashin_fi < 0.95)),

    ("0.95 <= FI < 1.00",
     (hashin_fi >= 0.95)
     & (hashin_fi < 1.00)),

    ("1.00 <= FI < 1.05",
     (hashin_fi >= 1.00)
     & (hashin_fi < 1.05)),

    ("1.05 <= FI < 1.10",
     (hashin_fi >= 1.05)
     & (hashin_fi < 1.10)),

    ("1.10 <= FI < 1.20",
     (hashin_fi >= 1.10)
     & (hashin_fi < 1.20)),

    ("FI >= 1.20",
     hashin_fi >= 1.20)
]


print()
print("-----------------------------------")
print("PERFORMANCE BY FI BAND")
print("-----------------------------------")

band_results = []

for name, mask in bands:

    total = mask.sum()

    if total == 0:
        continue

    correct = (
        (y_test[mask] == y_pred[mask])
    ).sum()

    errors = total - correct

    accuracy = correct / total

    band_results.append({
        "FI_band": name,
        "samples": total,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy
    })

    print()
    print(name)
    print(f"Samples  : {total}")
    print(f"Correct  : {correct}")
    print(f"Errors   : {errors}")
    print(
        f"Accuracy : {accuracy:.6f}"
    )


band_df = pd.DataFrame(
    band_results
)

band_df.to_csv(
    OUTPUT_DIR / "fi_band_performance.csv",
    index=False
)


# ==================================================
# NEAR-BOUNDARY ANALYSIS
# ==================================================

print()
print("-----------------------------------")
print("NEAR-BOUNDARY ERROR ANALYSIS")
print("-----------------------------------")

for tolerance in [
    0.01,
    0.02,
    0.05,
    0.10
]:

    boundary_mask = (
        np.abs(hashin_fi - 1.0)
        <= tolerance
    )

    total = boundary_mask.sum()

    errors = (
        error_mask
        & boundary_mask
    ).sum()

    if total > 0:
        error_rate = errors / total

        print()
        print(
            f"|FI - 1| <= {tolerance:.2f}"
        )

        print(
            f"Samples    : {total}"
        )

        print(
            f"Errors     : {errors}"
        )

        print(
            f"Error rate : {error_rate:.6f}"
        )


# ==================================================
# ERROR TABLE
# ==================================================

error_mask = ~correct_prediction

error_table = pd.DataFrame({
    "actual_failed": y_test[error_mask],
    "actual_hashin_fi": hashin_fi[error_mask],
    "predicted_probability": y_probability[error_mask],
    "predicted_class": y_pred[error_mask],
    "distance_from_boundary": (
        np.abs(
            hashin_fi[error_mask] - 1.0
        )
    )
})

error_table = error_table.sort_values(
    "distance_from_boundary"
)

error_table.to_csv(
    OUTPUT_DIR / "classification_errors.csv",
    index=False
)


# ==================================================
# PLOT 1 — PROBABILITY VS ACTUAL FI
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    hashin_fi,
    y_probability,
    alpha=0.5
)

plt.axvline(
    1.0,
    linestyle="--",
    label="Hashin FI = 1"
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Classification threshold = 0.5"
)

plt.xlabel("Actual Hashin FI")
plt.ylabel("Predicted Failure Probability")

plt.title(
    "Predicted Failure Probability vs Hashin FI"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "probability_vs_hashin_fi.png",
    dpi=300
)

plt.close()


# ==================================================
# PLOT 2 — ERROR DISTANCE
# ==================================================

plt.figure(figsize=(8, 5))

plt.hist(
    error_distance,
    bins=20
)

plt.xlabel(
    "|Hashin FI - 1|"
)

plt.ylabel(
    "Number of Classification Errors"
)

plt.title(
    "Distance of Classification Errors from Failure Boundary"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_distance_histogram.png",
    dpi=300
)

plt.close()


# ==================================================
# PLOT 3 — ACCURACY BY FI BAND
# ==================================================

plt.figure(figsize=(10, 5))

plt.bar(
    band_df["FI_band"],
    band_df["accuracy"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(
    0,
    1.05
)

plt.xlabel("Hashin FI Band")
plt.ylabel("Classification Accuracy")

plt.title(
    "Classifier Accuracy Across Hashin FI Bands"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_by_fi_band.png",
    dpi=300
)

plt.close()


# ==================================================
# FINISHED
# ==================================================

print()
print("-----------------------------------")
print("ERROR ANALYSIS COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: {OUTPUT_DIR}"
)