import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed/ml")
OUTPUT_DIR = Path("results/architecture_comparison")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

y_train = np.load(DATA_DIR / "y_reg_train.npy")
y_val = np.load(DATA_DIR / "y_reg_val.npy")
y_test = np.load(DATA_DIR / "y_reg_test.npy")


print("-----------------------------------")
print("DNN ARCHITECTURE COMPARISON")
print("-----------------------------------")

print()
print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(hidden_layers):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(X_train.shape[1],)
        )
    )

    for units in hidden_layers:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu"
            )
        )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="linear"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# ARCHITECTURES
# ============================================================

architectures = {
    "Small": [32, 16],

    "Baseline": [64, 64, 32],

    "Large": [128, 128, 64, 32]
}


# ============================================================
# TRAINING
# ============================================================

results = []


for name, architecture in architectures.items():

    print()
    print("===================================")
    print(f"Training: {name}")
    print(f"Architecture: {architecture}")
    print("===================================")

    # Reset random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_model(
        architecture
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test,
        verbose=0
    ).flatten()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    best_epoch = (
        np.argmin(history.history["val_loss"])
        + 1
    )

    total_epochs = len(
        history.history["loss"]
    )

    parameter_count = model.count_params()

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "model": name,
        "architecture": str(architecture),
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model.save(
        OUTPUT_DIR /
        f"{name.lower()}_model.keras"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR /
        f"{name.lower()}_y_pred.npy",
        y_pred
    )

    print()
    print(f"Parameters : {parameter_count}")
    print(f"Best epoch : {best_epoch}")
    print(f"Total epochs : {total_epochs}")
    print(f"MAE        : {mae:.6f}")
    print(f"RMSE       : {rmse:.6f}")
    print(f"R²         : {r2:.6f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


print()
print()
print("-----------------------------------")
print("FINAL ARCHITECTURE COMPARISON")
print("-----------------------------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "architecture_comparison.csv",
    index=False
)


print()
print("-----------------------------------")
print("COMPARISON COMPLETE")
print("-----------------------------------")

print(
    f"Results saved to: {OUTPUT_DIR}"
)