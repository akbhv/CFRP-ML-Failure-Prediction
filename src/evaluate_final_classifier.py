import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# PATHS
# ==================================================

RESULTS_DIR = Path("results/dnn_classifier")

THRESHOLD_DIR = (
    RESULTS_DIR / "threshold_analysis"
)

OUTPUT_DIR = (
    RESULTS_DIR / "final_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD TEST DATA
# ==================================================

y_test = np.load(
    RESULTS_DIR / "y_test.npy"
)

y_probability = np.load(
    RESULTS_DIR / "y_probability.npy"
)


# ==================================================
# LOAD FROZEN THRESHOLD
# ==================================================

selected_threshold = float(
    np.load(
        THRESHOLD_DIR / "selected_threshold.npy"
    )[0]
)


print("-----------------------------------")
print("FINAL CLASSIFIER EVALUATION")
print("-----------------------------------")

print()
print(
    f"Frozen threshold: "
    f"{selected_threshold:.2f}"
)


# ==================================================
# PREDICTIONS — THRESHOLD 0.55
# ==================================================

y_pred_final = (
    y_probability >= selected_threshold
).astype(int)


# ==================================================
# FINAL METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred_final
)

precision = precision_score(
    y_test,
    y_pred_final,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred_final,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred_final,
    zero_division=0
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred_final
)

tn, fp, fn, tp = cm.ravel()


# ==================================================
# PRINT FINAL RESULTS
# ==================================================

print()
print("-----------------------------------")
print("FINAL TEST SET RESULTS")
print("-----------------------------------")

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1-score  : {f1:.6f}"
)

print()
print("-----------------------------------")
print("CONFUSION MATRIX")
print("-----------------------------------")

print(cm)

print()
print(
    f"True negatives  : {tn}"
)

print(
    f"False positives : {fp}"
)

print(
    f"False negatives : {fn}"
)

print(
    f"True positives  : {tp}"
)


# ==================================================
# BASELINE 0.50 COMPARISON
# ==================================================

y_pred_baseline = (
    y_probability >= 0.50
).astype(int)


baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

baseline_precision = precision_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)


# ==================================================
# COMPARISON TABLE
# ==================================================

comparison = pd.DataFrame({
    "threshold": [
        0.50,
        selected_threshold
    ],

    "accuracy": [
        baseline_accuracy,
        accuracy
    ],

    "precision": [
        baseline_precision,
        precision
    ],

    "recall": [
        baseline_recall,
        recall
    ],

    "f1_score": [
        baseline_f1,
        f1
    ]
})


print()
print("-----------------------------------")
print("THRESHOLD COMPARISON")
print("-----------------------------------")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# SAVE RESULTS
# ==================================================

comparison.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)

np.save(
    OUTPUT_DIR / "final_y_pred.npy",
    y_pred_final
)


# ==================================================
# SAVE FINAL CONFUSION MATRIX
# ==================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Safe",
        "Failure"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    f"Final DNN Classifier\n"
    f"Threshold = {selected_threshold:.2f}"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==================================================
# FINISHED
# ==================================================

print()
print("-----------------------------------")
print("FINAL EVALUATION COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: "
    f"{OUTPUT_DIR}"
)