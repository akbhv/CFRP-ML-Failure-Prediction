
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

flag_columns = [
    "max_stress_failed",
    "max_strain_failed",
    "tsai_hill_failed",
    "tsai_wu_failed",
    "hoffman_failed",
    "hashin_failed",
    "puck_failed",
]

print("=== FINAL V3 DATASET AUDIT ===")
print(f"Shape: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values: {df.isna().sum().sum()}")

missing_targets = [c for c in targets if c not in df.columns]
missing_flags = [c for c in flag_columns if c not in df.columns]

print(f"Missing target columns: {missing_targets}")
print(f"Missing flag columns: {missing_flags}")

if not missing_targets:
    finite = np.isfinite(df[targets].to_numpy(dtype=float)).all()
    print(f"All target values finite: {finite}")

if not missing_flags and not missing_targets:
    mismatches = {}

    for target, flag in zip(targets, flag_columns):
        expected = df[target] >= 1.0
        actual = df[flag].astype(bool)
        mismatches[flag] = int((expected != actual).sum())

    print("\nFlag mismatches:")
    for flag, count in mismatches.items():
        print(f"  {flag}: {count}")

if "failed" in df.columns and "hashin_fi" in df.columns:
    expected_failed = df["hashin_fi"] >= 1.0
    actual_failed = df["failed"].astype(bool)
    print(
        "\nHashin label mismatches:",
        int((expected_failed != actual_failed).sum()),
    )

print("\n=== END AUDIT ===")
