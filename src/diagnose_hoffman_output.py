import pandas as pd
import numpy as np
from pathlib import Path

source = Path("data/processed/failure_dataset_v2_puck_corrected_exact.csv")
output = Path("data/processed/failure_dataset_v2_hoffman_corrected.csv")

a = pd.read_csv(source)
b = pd.read_csv(output)

hoffman = {
    "hoffman_fi", "hoffman_failed", "hoffman_ply",
    "hoffman_angle", "hoffman_surface"
}
cols = [c for c in a.columns if c not in hoffman]

print("Source shape:", a.shape)
print("Output shape:", b.shape)
print("Same column order:", list(a.columns) == list(b.columns))
print("\nNon-Hoffman differences:")

problems = 0
for c in cols:
    if c not in b.columns:
        print(f"MISSING COLUMN: {c}")
        problems += 1
        continue

    x, y = a[c], b[c]

    if x.equals(y):
        continue

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        xv = x.to_numpy(dtype=float)
        yv = y.to_numpy(dtype=float)
        same = np.isclose(xv, yv, rtol=1e-12, atol=1e-12, equal_nan=True)
        count = int((~same).sum())
        print(f"{c}: {count} numeric mismatches; max abs delta = "
              f"{np.nanmax(np.abs(xv-yv)) if len(xv) else 0:.12g}")
        if count:
            idx = np.flatnonzero(~same)[:3]
            for i in idx:
                print(f"  row {i}: source={x.iloc[i]!r}, output={y.iloc[i]!r}")
        else:
            print(f"{c}: values numerically equivalent; dtype source={x.dtype}, output={y.dtype}")
    else:
        mismatch = ~(x.eq(y) | (x.isna() & y.isna()))
        count = int(mismatch.sum())
        print(f"{c}: {count} value mismatches; dtype source={x.dtype}, output={y.dtype}")
        for i in np.flatnonzero(mismatch.to_numpy())[:3]:
            print(f"  row {i}: source={x.iloc[i]!r}, output={y.iloc[i]!r}")

    problems += 1

print("\nColumns with strict comparison differences:", problems)
print("Output file exists:", output.exists())
