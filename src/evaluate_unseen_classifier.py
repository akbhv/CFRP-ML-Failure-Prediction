import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_FILE = Path("data/processed/unseen/unseen_generalization_dataset.csv")
SCALER_FILE = Path("data/processed/ml/feature_scaler.pkl")
MODEL_FILE = Path("results/dnn_classifier/dnn_failure_classifier.keras")
THRESHOLD_FILE = Path("results/dnn_classifier/threshold_analysis/selected_threshold.npy")
OUTPUT_DIR = Path("results/dnn_classifier/unseen_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y_actual = df["failed"].values

scaler = joblib.load(SCALER_FILE)
X_scaled = scaler.transform(X)

model = load_model(MODEL_FILE)

threshold = float(np.load(THRESHOLD_FILE)[0])

probabilities = model.predict(X_scaled, verbose=0).flatten()
predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_actual, predictions)
precision = precision_score(y_actual, predictions, zero_division=0)
recall = recall_score(y_actual, predictions, zero_division=0)
f1 = f1_score(y_actual, predictions, zero_division=0)
roc_auc = roc_auc_score(y_actual, probabilities)

cm = confusion_matrix(y_actual, predictions)
tn, fp, fn, tp = cm.ravel()

print("-----------------------------------")
print("UNSEEN CLASSIFICATION EVALUATION")
print("-----------------------------------")
print()
print(f"Samples           : {len(df)}")
print(f"Frozen threshold  : {threshold:.2f}")
print(f"Model             : {MODEL_FILE}")
print()
print("-----------------------------------")
print("GENERALIZATION RESULTS")
print("-----------------------------------")
print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")
print()
print("-----------------------------------")
print("CONFUSION MATRIX")
print("-----------------------------------")
print(cm)
print()
print(f"True negatives  : {tn}")
print(f"False positives : {fp}")
print(f"False negatives : {fn}")
print(f"True positives  : {tp}")

results = pd.DataFrame({
    "metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "value": [accuracy, precision, recall, f1, roc_auc]
})

results.to_csv(
    OUTPUT_DIR / "unseen_classification_metrics.csv",
    index=False
)

prediction_df = df.copy()
prediction_df["failure_probability"] = probabilities
prediction_df["predicted_failed"] = predictions
prediction_df["correct"] = (predictions == y_actual).astype(int)

prediction_df.to_csv(
    OUTPUT_DIR / "unseen_classification_predictions.csv",
    index=False
)

print()
print("-----------------------------------")
print("EVALUATION COMPLETE")
print("-----------------------------------")
print(f"Results saved to: {OUTPUT_DIR}")
