
import numpy as np
import pandas as pd

path = "data/processed/failure_dataset_v3_scaled.csv"
df = pd.read_csv(path)

targets = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

print("=== Failure-index threshold check ===")
for col in targets:
    values = df[col].to_numpy(dtype=float)
    print(
        f"{col:16s} "
        f"FI < 1: {(values < 1).sum():5d} | "
        f"FI = 1 (within 1e-3): "
        f"{np.isclose(values, 1.0, atol=1e-3).sum():5d} | "
        f"FI > 1: {(values > 1).sum():5d}"
    )

print("\n=== Label consistency ===")
if "hashin_mode" in df.columns and "puck_mode" in df.columns:
    print("Hashin mode counts:")
    print(df["hashin_mode"].value_counts().to_string())

    print("\nPuck mode counts:")
    print(df["puck_mode"].value_counts().to_string())

print("\n=== Target correlations ===")
print(df[targets].corr().round(3).to_string())

print("\n=== Load feature ranges ===")
load_cols = [
    c for c in ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
    if c in df.columns
]
if load_cols:
    print(df[load_cols].describe().loc[["min", "max"]].to_string())
else:
    print("Load columns not found under expected names.")
