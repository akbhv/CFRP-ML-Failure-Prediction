import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ==================================================
# PATHS
# ==================================================

INPUT_PATH = Path(
    "data/processed/failure_dataset_v2.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# FEATURES AND TARGETS
# ==================================================

FEATURES = [
    "Nx",
    "Ny",
    "Nxy",
    "Mx",
    "My",
    "Mxy"
]

REGRESSION_TARGET = "hashin_fi"
CLASSIFICATION_TARGET = "failed"


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)

X = df[FEATURES].copy()

y_regression = df[
    REGRESSION_TARGET
].copy()

y_classification = df[
    CLASSIFICATION_TARGET
].copy()


# ==================================================
# TRAIN / TEMPORARY SPLIT
# ==================================================

X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = (
    train_test_split(
        X,
        y_regression,
        y_classification,
        test_size=0.30,
        random_state=42,
        stratify=y_classification
    )
)


# ==================================================
# VALIDATION / TEST SPLIT
# ==================================================

X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = (
    train_test_split(
        X_temp,
        y_reg_temp,
        y_cls_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_cls_temp
    )
)


# ==================================================
# STANDARDIZATION
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)


# ==================================================
# SAVE ARRAYS
# ==================================================

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train_scaled
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val_scaled
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test_scaled
)

np.save(
    OUTPUT_DIR / "y_reg_train.npy",
    y_reg_train.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_reg_val.npy",
    y_reg_val.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_reg_test.npy",
    y_reg_test.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_cls_train.npy",
    y_cls_train.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_cls_val.npy",
    y_cls_val.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_cls_test.npy",
    y_cls_test.to_numpy()
)


# ==================================================
# SAVE SCALER
# ==================================================

joblib.dump(
    scaler,
    OUTPUT_DIR / "feature_scaler.pkl"
)


# ==================================================
# SUMMARY
# ==================================================

print("-----------------------------------")
print("ML DATASET PREPARATION")
print("-----------------------------------")

print()
print("Feature columns:")
print(FEATURES)

print()

print(
    f"Training samples   = {len(X_train)}"
)

print(
    f"Validation samples = {len(X_val)}"
)

print(
    f"Test samples       = {len(X_test)}"
)


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

print()
print("-----------------------------------")
print("CLASS DISTRIBUTION")
print("-----------------------------------")

print("Training:")
print(y_cls_train.value_counts())

print()

print("Validation:")
print(y_cls_val.value_counts())

print()

print("Test:")
print(y_cls_test.value_counts())


# ==================================================
# REGRESSION TARGET
# ==================================================

print()
print("-----------------------------------")
print("REGRESSION TARGET")
print("-----------------------------------")

print("Training:")
print(y_reg_train.describe())

print()

print("Validation:")
print(y_reg_val.describe())

print()

print("Test:")
print(y_reg_test.describe())


# ==================================================
# SCALING CHECK
# ==================================================

print()
print("-----------------------------------")
print("SCALING CHECK")
print("-----------------------------------")

print(
    "Training feature means:"
)

print(
    X_train_scaled.mean(axis=0)
)

print()

print(
    "Training feature standard deviations:"
)

print(
    X_train_scaled.std(axis=0)
)


print()
print("-----------------------------------")
print("ML DATASET READY")
print("-----------------------------------")