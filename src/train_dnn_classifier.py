import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# REPRODUCIBILITY
# ==================================================

np.random.seed(42)
tf.random.set_seed(42)


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path("data/processed/ml")
RESULTS_DIR = Path("results/dnn_classifier")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_cls_train.npy")
y_val = np.load(DATA_DIR / "y_cls_val.npy")
y_test = np.load(DATA_DIR / "y_cls_test.npy")


print("-----------------------------------")
print("DNN BINARY CLASSIFICATION")
print("-----------------------------------")

print()
print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")


# ==================================================
# BUILD MODEL
# ==================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==================================================
# MODEL SUMMARY
# ==================================================

print()
print("-----------------------------------")
print("MODEL ARCHITECTURE")
print("-----------------------------------")

model.summary()


# ==================================================
# CALLBACKS
# ==================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)


# ==================================================
# TRAIN
# ==================================================

print()
print("-----------------------------------")
print("TRAINING")
print("-----------------------------------")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ==================================================
# SAVE MODEL
# ==================================================

model.save(
    RESULTS_DIR / "dnn_failure_classifier.keras"
)


# ==================================================
# PREDICTIONS
# ==================================================

y_probability = model.predict(
    X_test,
    verbose=0
).flatten()

# Classification threshold = 0.5
y_pred = (
    y_probability >= 0.5
).astype(int)


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print()
print("-----------------------------------")
print("TEST SET RESULTS")
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

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("-----------------------------------")
print("CONFUSION MATRIX")
print("-----------------------------------")

print(cm)


# ==================================================
# SAVE CONFUSION MATRIX
# ==================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Failure"]
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "DNN Failure Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==================================================
# TRAINING HISTORY — LOSS
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("DNN Classification Training History")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_loss.png",
    dpi=300
)

plt.close()


# ==================================================
# TRAINING HISTORY — ACCURACY
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DNN Classification Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_accuracy.png",
    dpi=300
)

plt.close()


# ==================================================
# SAVE PREDICTIONS
# ==================================================

np.save(
    RESULTS_DIR / "y_test.npy",
    y_test
)

np.save(
    RESULTS_DIR / "y_probability.npy",
    y_probability
)

np.save(
    RESULTS_DIR / "y_pred.npy",
    y_pred
)


# ==================================================
# FINISHED
# ==================================================

print()
print("-----------------------------------")
print("TRAINING COMPLETE")
print("-----------------------------------")

print(
    f"Best epoch: "
    f"{np.argmin(history.history['val_loss']) + 1}"
)

print(
    f"Total epochs trained: "
    f"{len(history.history['loss'])}"
)

print()
print(
    f"Model saved to: "
    f"{RESULTS_DIR / 'dnn_failure_classifier.keras'}"
)

print(
    f"Results saved to: "
    f"{RESULTS_DIR}"
)