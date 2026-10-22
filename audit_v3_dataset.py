
import pandas as pd
import numpy as np

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

print("=" * 50)
print("V3 DATASET AUDIT")
print("=" * 50)

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nTarget audit:")
for col in targets:
    if col not in df.columns:
        print(f"{col}: MISSING COLUMN")
        continue

    values = df[col].to_numpy(dtype=float)
    print(
        f"{col}: finite={np.isfinite(values).all()}, "
        f"min={values.min():.6f}, "
        f"max={values.max():.6f}"
    )

print("\nSampling regions:")
if "sampling_region" in df.columns:
    print(df["sampling_region"].value_counts().to_string())
else:
    print("Column missing")

print("\nHashin modes:")
if "hashin_mode" in df.columns:
    print(df["hashin_mode"].value_counts().to_string())
else:
    print("Column missing")

print("\nPuck modes:")
if "puck_mode" in df.columns:
    print(df["puck_mode"].value_counts().to_string())
else:
    print("Column missing")

print("\nAudit finished.")
