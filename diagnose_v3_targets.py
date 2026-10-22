
import numpy as np
import pandas as pd

path = "data/processed/failure_dataset_v3_scaled.csv"
df = pd.read_csv(path)

pairs = [
    ("max_stress_fi", "puck_fi"),
    ("tsai_wu_fi", "hoffman_fi"),
]

print("=== Target-pair differences ===")
for a, b in pairs:
    x = df[a].to_numpy(dtype=float)
    y = df[b].to_numpy(dtype=float)
    diff = x - y

    print(f"\n{a} vs {b}")
    print(f"Correlation:          {np.corrcoef(x, y)[0, 1]:.8f}")
    print(f"Mean absolute diff:   {np.mean(np.abs(diff)):.8g}")
    print(f"Maximum abs diff:     {np.max(np.abs(diff)):.8g}")
    print(f"Nearly identical:     {np.mean(np.isclose(x, y, rtol=1e-5, atol=1e-8)) * 100:.2f}%")

print("\n=== Governing-mode combinations ===")
if {"hashin_mode", "puck_mode"}.issubset(df.columns):
    print(pd.crosstab(df["hashin_mode"], df["puck_mode"]).to_string())

print("\n=== Input sign distribution ===")
for col in ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]:
    if col in df.columns:
        values = df[col].to_numpy(dtype=float)
        print(
            f"{col:5s}: negative={(values < 0).sum():5d}, "
            f"positive={(values > 0).sum():5d}, "
            f"zero={(values == 0).sum():5d}"
        )
