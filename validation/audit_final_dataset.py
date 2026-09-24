import pandas as pd
import numpy as np


DATASET_PATH = (
    "data/processed/failure_dataset_v2.csv"
)


df = pd.read_csv(DATASET_PATH)


print("-----------------------------------")
print("FINAL DATASET AUDIT")
print("-----------------------------------")


# ==================================================
# BASIC INFORMATION
# ==================================================

print()
print("Shape:")
print(df.shape)

print()
print("Columns:")
print(df.columns.tolist())


# ==================================================
# MISSING VALUES
# ==================================================

print()
print("-----------------------------------")
print("MISSING VALUES")
print("-----------------------------------")

print(df.isna().sum())


# ==================================================
# DUPLICATES
# ==================================================

print()
print("-----------------------------------")
print("DUPLICATES")
print("-----------------------------------")

print(
    "Duplicate rows =",
    df.duplicated().sum()
)


# ==================================================
# FAILURE CLASS
# ==================================================

print()
print("-----------------------------------")
print("FAILURE CLASS")
print("-----------------------------------")

class_counts = df["failed"].value_counts()

print(class_counts)

print()

print(
    "Safe percentage =",
    (df["failed"] == 0).mean() * 100
)

print(
    "Failed percentage =",
    (df["failed"] == 1).mean() * 100
)


# ==================================================
# HASHIN FI
# ==================================================

print()
print("-----------------------------------")
print("HASHIN FI")
print("-----------------------------------")

print(
    df["hashin_fi"].describe()
)


# ==================================================
# FAILURE MODES
# ==================================================

print()
print("-----------------------------------")
print("FAILURE MODES")
print("-----------------------------------")

mode_counts = df["hashin_mode"].value_counts()

print(mode_counts)

print()

print(
    "Failure-mode percentages:"
)

print(
    mode_counts / len(df) * 100
)


# ==================================================
# INPUT FEATURES
# ==================================================

features = [
    "Nx",
    "Ny",
    "Nxy",
    "Mx",
    "My",
    "Mxy"
]


print()
print("-----------------------------------")
print("INPUT FEATURE STATISTICS")
print("-----------------------------------")

print(
    df[features].describe()
)


# ==================================================
# CORRELATION WITH HASHIN FI
# ==================================================

print()
print("-----------------------------------")
print("CORRELATION WITH HASHIN FI")
print("-----------------------------------")

correlations = (
    df[features + ["hashin_fi"]]
    .corr()["hashin_fi"]
    .sort_values()
)

print(correlations)


# ==================================================
# TARGET CONSISTENCY
# ==================================================

print()
print("-----------------------------------")
print("TARGET CONSISTENCY")
print("-----------------------------------")

expected_failed = (
    df["hashin_fi"] >= 1.0
)

actual_failed = (
    df["failed"] == 1
)

mismatches = (
    expected_failed != actual_failed
).sum()

print(
    "FI/classification mismatches =",
    mismatches
)


# ==================================================
# SAMPLING REGIONS
# ==================================================

print()
print("-----------------------------------")
print("SAMPLING REGIONS")
print("-----------------------------------")

print(
    df["sampling_region"].value_counts()
)


# ==================================================
# LOAD SIGN DISTRIBUTION
# ==================================================

print()
print("-----------------------------------")
print("LOAD SIGN DISTRIBUTION")
print("-----------------------------------")

for feature in features:

    positive = (df[feature] > 0).sum()
    negative = (df[feature] < 0).sum()
    zero = (df[feature] == 0).sum()

    print()
    print(feature)

    print("Positive:", positive)
    print("Negative:", negative)
    print("Zero:", zero)


# ==================================================
# FINAL CHECK
# ==================================================

print()
print("-----------------------------------")
print("AUDIT COMPLETE")
print("-----------------------------------")