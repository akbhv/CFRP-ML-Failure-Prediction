import pandas as pd
import numpy as np


DATASET_PATH = "data/processed/failure_dataset_v2_pilot.csv"


df = pd.read_csv(DATASET_PATH)


print("-----------------------------------")
print("DATASET V2 PILOT ANALYSIS")
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
# HASHIN FI STATISTICS
# ==================================================

print()
print("-----------------------------------")
print("HASHIN FI STATISTICS")
print("-----------------------------------")

print(df["hashin_fi"].describe())


# ==================================================
# FI BINS
# ==================================================

print()
print("-----------------------------------")
print("HASHIN FI DISTRIBUTION")
print("-----------------------------------")

bins = [
    0.0,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    1.00,
    1.05,
    1.25,
    1.50,
    2.00,
    np.inf
]

labels = [
    "0.00-0.25",
    "0.25-0.50",
    "0.50-0.75",
    "0.75-0.90",
    "0.90-0.95",
    "0.95-1.00",
    "1.00-1.05",
    "1.05-1.25",
    "1.25-1.50",
    "1.50-2.00",
    "2.00+"
]

fi_bins = pd.cut(
    df["hashin_fi"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(fi_bins.value_counts().sort_index())


# ==================================================
# FAILURE CLASS
# ==================================================

print()
print("-----------------------------------")
print("FAILURE CLASS")
print("-----------------------------------")

print(df["failed"].value_counts())


# ==================================================
# FAILURE MODES
# ==================================================

print()
print("-----------------------------------")
print("FAILURE MODES")
print("-----------------------------------")

print(df["hashin_mode"].value_counts())


# ==================================================
# SAMPLING REGIONS
# ==================================================

print()
print("-----------------------------------")
print("SAMPLING REGIONS")
print("-----------------------------------")

print(df["sampling_region"].value_counts())


# ==================================================
# LOAD MULTIPLIER
# ==================================================

print()
print("-----------------------------------")
print("LOAD MULTIPLIER")
print("-----------------------------------")

print(df["load_multiplier"].describe())


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
# MISSING VALUES
# ==================================================

print()
print("-----------------------------------")
print("MISSING VALUES")
print("-----------------------------------")

print(df.isna().sum())


# ==================================================
# LOAD RANGES
# ==================================================

print()
print("-----------------------------------")
print("LOAD RANGES")
print("-----------------------------------")

load_columns = [
    "Nx",
    "Ny",
    "Nxy",
    "Mx",
    "My",
    "Mxy"
]

print(df[load_columns].describe())