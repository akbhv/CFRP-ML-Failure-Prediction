
import pandas as pd

path = "data/processed/failure_dataset_v3_scaled.csv"
df = pd.read_csv(path)

fi_col = "hoffman_fi"
flag_col = "hoffman_failed"

expected = df[fi_col] >= 1.0
actual = df[flag_col].astype(bool)

mismatches = expected != actual

print("=== Hoffman flag audit ===")
print(f"Rows checked: {len(df)}")
print(f"Existing failed flags: {actual.sum()}")
print(f"Expected failed flags: {expected.sum()}")
print(f"Flag mismatches: {mismatches.sum()}")

if mismatches.any():
    print("\nFirst mismatched rows:")
    print(df.loc[mismatches, [fi_col, flag_col]].head(10))

print("\nAudit complete. No dataset changes were made.")
