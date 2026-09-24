import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# REPRODUCIBILITY
# ==================================================

np.random.seed(42)
tf.random.set_seed(42)


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path("data/processed/ml")
RESULTS_DIR = Path("results/dnn_regression")

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

y_train = np.load(DATA_DIR / "y_reg_train.npy")
y_val = np.load(DATA_DIR / "y_reg_val.npy")
y_test = np.load(DATA_DIR / "y_reg_test.npy")


print("-----------------------------------")
print("DNN REGRESSION")
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
        activation="linear"
    )
])


# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
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
    RESULTS_DIR / "dnn_hashin_fi.keras"
)


# ==================================================
# TEST PREDICTIONS
# ==================================================

y_pred = model.predict(
    X_test,
    verbose=0
).flatten()


# ==================================================
# METRICS
# ==================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print()
print("-----------------------------------")
print("TEST SET RESULTS")
print("-----------------------------------")

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)

print(
    f"R²   : {r2:.6f}"
)


# ==================================================
# SAVE PREDICTIONS
# ==================================================

np.save(
    RESULTS_DIR / "y_test.npy",
    y_test
)

np.save(
    RESULTS_DIR / "y_pred.npy",
    y_pred
)


# ==================================================
# TRAINING CURVE
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
plt.ylabel("MSE Loss")
plt.title("DNN Regression Training History")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "training_history.png",
    dpi=300
)

plt.close()


# ==================================================
# ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Hashin FI")
plt.ylabel("Predicted Hashin FI")
plt.title("Actual vs Predicted Hashin Failure Index")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ==================================================
# FINISHED
# ==================================================

print()
print("-----------------------------------")
print("TRAINING COMPLETE")
print("-----------------------------------")

print(
    f"Best epoch: {np.argmin(history.history['val_loss']) + 1}"
)

print(
    f"Total epochs trained: {len(history.history['loss'])}"
)

print()
print(
    f"Model saved to: {RESULTS_DIR / 'dnn_hashin_fi.keras'}"
)

print(
    f"Results saved to: {RESULTS_DIR}"
)