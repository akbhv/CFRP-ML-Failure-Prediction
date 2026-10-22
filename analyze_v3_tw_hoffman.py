
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/failure_dataset_v3_scaled.csv")

tw = df["tsai_wu_fi"].to_numpy(dtype=float)
hf = df["hoffman_fi"].to_numpy(dtype=float)

diff = hf - tw
abs_diff = np.abs(diff)

print("=== Tsai-Wu vs Hoffman: full dataset ===")
print(f"Rows: {len(df)}")
print(f"Correlation: {np.corrcoef(tw, hf)[0, 1]:.10f}")
print(f"Mean signed difference (Hoffman - Tsai-Wu): {diff.mean():.10f}")
print(f"Mean absolute difference: {abs_diff.mean():.10f}")
print(f"Median absolute difference: {np.median(abs_diff):.10f}")
print(f"95th percentile absolute difference: {np.percentile(abs_diff, 95):.10f}")
print(f"99th percentile absolute difference: {np.percentile(abs_diff, 99):.10f}")
print(f"Maximum absolute difference: {abs_diff.max():.10f}")

for threshold in [1e-6, 1e-4, 1e-3, 1e-2]:
    print(
        f"Rows with |difference| <= {threshold:g}: "
        f"{np.sum(abs_diff <= threshold)} "
        f"({np.mean(abs_diff <= threshold) * 100:.2f}%)"
    )

# Check whether the criteria disagree on the failure classification.
tw_failed = tw >= 1.0
hf_failed = hf >= 1.0
disagree = tw_failed != hf_failed

print("\n=== Failure classification disagreement ===")
print(f"Disagreeing rows: {disagree.sum()} ({disagree.mean() * 100:.3f}%)")

if disagree.any():
    print(df.loc[disagree, [
        "tsai_wu_fi", "hoffman_fi",
        "tsai_wu_failed", "hoffman_failed"
    ]].head(10).to_string(index=True))

print("\n=== Largest differences ===")
largest = np.argsort(abs_diff)[-10:][::-1]
print(df.iloc[largest][[
    "tsai_wu_fi", "hoffman_fi",
    "tsai_wu_ply", "hoffman_ply",
    "tsai_wu_angle", "hoffman_angle",
    "tsai_wu_surface", "hoffman_surface"
]].to_string(index=True))
