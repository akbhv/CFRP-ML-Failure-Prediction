import numpy as np
import pandas as pd
import tensorflow as tf
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

DATA_DIR = Path("data/processed/ml")
MODEL_PATH = Path(
    "results/dnn_classifier/dnn_failure_classifier.keras"
)

OUTPUT_DIR = Path(
    "results/dnn_classifier/threshold_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD VALIDATION DATA
# ==================================================

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

y_val = np.load(
    DATA_DIR / "y_cls_val.npy"
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==================================================
# VALIDATION PREDICTIONS
# ==================================================

y_probability = model.predict(
    X_val,
    verbose=0
).flatten()


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
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    false_positives = (
        (y_val == 0)
        & (y_pred == 1)
    ).sum()

    false_negatives = (
        (y_val == 1)
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
print("VALIDATION THRESHOLD ANALYSIS")
print("-----------------------------------")

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ==================================================
# SELECT BEST F1 THRESHOLD
# ==================================================

best_f1_index = (
    results_df["f1_score"].idxmax()
)

best_row = (
    results_df.loc[best_f1_index]
)

best_threshold = float(
    best_row["threshold"]
)


print()
print("-----------------------------------")
print("SELECTED THRESHOLD")
print("-----------------------------------")

print(
    f"Threshold       : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation accuracy : "
    f"{best_row['accuracy']:.6f}"
)

print(
    f"Validation precision: "
    f"{best_row['precision']:.6f}"
)

print(
    f"Validation recall   : "
    f"{best_row['recall']:.6f}"
)

print(
    f"Validation F1       : "
    f"{best_row['f1_score']:.6f}"
)

print(
    f"False positives     : "
    f"{int(best_row['false_positives'])}"
)

print(
    f"False negatives     : "
    f"{int(best_row['false_negatives'])}"
)


# ==================================================
# SAVE RESULTS
# ==================================================

results_df.to_csv(
    OUTPUT_DIR / "validation_threshold_results.csv",
    index=False
)

np.save(
    OUTPUT_DIR / "selected_threshold.npy",
    np.array([best_threshold])
)


print()
print("-----------------------------------")
print("VALIDATION ANALYSIS COMPLETE")
print("-----------------------------------")

print(
    f"Selected threshold saved to: "
    f"{OUTPUT_DIR / 'selected_threshold.npy'}"
)