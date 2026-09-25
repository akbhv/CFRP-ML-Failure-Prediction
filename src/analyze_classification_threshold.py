import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# PATHS
# ==================================================

RESULTS_DIR = Path("results/dnn_classifier")

OUTPUT_DIR = RESULTS_DIR / "threshold_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD TEST PREDICTIONS
# ==================================================

y_test = np.load(
    RESULTS_DIR / "y_test.npy"
)

y_probability = np.load(
    RESULTS_DIR / "y_probability.npy"
)


# ==================================================
# THRESHOLD ANALYSIS
# ==================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

results = []


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    false_positives = (
        (y_test == 0)
        & (y_pred == 1)
    ).sum()

    false_negatives = (
        (y_test == 1)
        & (y_pred == 0)
    ).sum()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    })


results_df = pd.DataFrame(
    results
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("-----------------------------------")
print("CLASSIFICATION THRESHOLD ANALYSIS")
print("-----------------------------------")

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# BEST F1 THRESHOLD
# ==================================================

best_f1_index = (
    results_df["f1_score"].idxmax()
)

best_f1_row = (
    results_df.loc[best_f1_index]
)


print()
print("-----------------------------------")
print("BEST F1 THRESHOLD")
print("-----------------------------------")

print(
    f"Threshold       : "
    f"{best_f1_row['threshold']:.2f}"
)

print(
    f"Accuracy        : "
    f"{best_f1_row['accuracy']:.6f}"
)

print(
    f"Precision       : "
    f"{best_f1_row['precision']:.6f}"
)

print(
    f"Recall          : "
    f"{best_f1_row['recall']:.6f}"
)

print(
    f"F1-score        : "
    f"{best_f1_row['f1_score']:.6f}"
)

print(
    f"False positives : "
    f"{int(best_f1_row['false_positives'])}"
)

print(
    f"False negatives : "
    f"{int(best_f1_row['false_negatives'])}"
)


# ==================================================
# SAVE TABLE
# ==================================================

results_df.to_csv(
    OUTPUT_DIR / "threshold_results.csv",
    index=False
)


# ==================================================
# PLOT — METRICS VS THRESHOLD
# ==================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["threshold"],
    results_df["accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1_score"],
    marker="o",
    label="F1-score"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Metric"
)

plt.title(
    "Classification Performance vs Probability Threshold"
)

plt.ylim(
    0,
    1.05
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "metrics_vs_threshold.png",
    dpi=300
)

plt.close()


# ==================================================
# PLOT — FALSE POSITIVES / NEGATIVES
# ==================================================

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["threshold"],
    results_df["false_positives"],
    marker="o",
    label="False Positives"
)

plt.plot(
    results_df["threshold"],
    results_df["false_negatives"],
    marker="o",
    label="False Negatives"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Number of Errors"
)

plt.title(
    "False Positives and False Negatives vs Threshold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "errors_vs_threshold.png",
    dpi=300
)

plt.close()


# ==================================================
# FINISHED
# ==================================================

print()
print("-----------------------------------")
print("THRESHOLD ANALYSIS COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: {OUTPUT_DIR}"
)