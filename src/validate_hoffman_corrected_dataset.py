import pandas as pd
import numpy as np
from pathlib import Path

source = Path("data/processed/failure_dataset_v2_puck_corrected_exact.csv")
output = Path("data/processed/failure_dataset_v2_hoffman_corrected.csv")

a = pd.read_csv(source)
b = pd.read_csv(output)

cols = [
    "hoffman_fi",
    "hoffman_failed",
    "hoffman_ply",
    "hoffman_angle",
    "hoffman_surface",
]

print("=== HOFFMAN CORRECTION VALIDATION ===")
print("Rows:", len(b))
print("Columns:", len(b.columns))
print("Missing values in Hoffman columns:")
print(b[cols].isna().sum().to_string())

fi = b["hoffman_fi"].to_numpy(dtype=float)
labels = b["hoffman_failed"]

expected = (fi >= 1.0).astype(int)
actual = labels.to_numpy(dtype=int)

print("\nFailure-label mismatches against FI >= 1.0:", int(np.sum(expected != actual)))
print("Non-finite Hoffman FI values:", int(np.sum(~np.isfinite(fi))))

print("\nCorrected Hoffman FI statistics:")
print(b["hoffman_fi"].describe().to_string())

print("\nHoffman failure-label counts:")
print(labels.value_counts(dropna=False).sort_index().to_string())

print("\nOriginal vs corrected Hoffman FI:")
delta = b["hoffman_fi"] - a["hoffman_fi"]
print("Changed rows:", int((~np.isclose(
    b["hoffman_fi"], a["hoffman_fi"], rtol=1e-12, atol=1e-12
)).sum()))
print("Maximum absolute change:", float(delta.abs().max()))
print("Mean absolute change:", float(delta.abs().mean()))

print("\nFirst five corrected Hoffman records:")
print(b[cols].head().to_string(index=False))

print("\nOutput:", output.resolve())
