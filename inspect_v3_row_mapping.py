
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(
    ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
)
results = ROOT / "results" / "multitarget_dnn_v3"

targets = [
    "max_stress_fi", "max_strain_fi", "tsai_hill_fi",
    "tsai_wu_fi", "hoffman_fi", "hashin_fi", "puck_fi",
]

for target in targets:
    path = results / target / "test_predictions.csv"
    df = pd.read_csv(path)

    print(f"\n=== {target} ===")

    if "row_index" not in df.columns:
        print("Missing row_index column.")
        continue

    idx = df["row_index"].to_numpy(dtype=int)
    actual = df["reference_fi"].to_numpy(dtype=float)

    if len(np.unique(idx)) != len(idx):
        print("WARNING: duplicate row_index values found.")

    if np.any(idx < 0) or np.any(idx >= len(data)):
        print("WARNING: row_index outside source dataset.")
        continue

    source_values = data.iloc[idx][target].to_numpy(dtype=float)
    matches = np.isclose(
        source_values, actual, rtol=1e-6, atol=1e-6
    )

    print(f"Rows matching source: {matches.sum()}/{len(matches)}")

    for i in np.flatnonzero(~matches):
        print(
            f"prediction_row={i}, row_index={idx[i]}, "
            f"saved_reference={actual[i]:.12g}, "
            f"source_reference={source_values[i]:.12g}, "
            f"predicted={df['predicted_fi'].iloc[i]:.12g}"
        )

print("\nRead-only mapping inspection complete.")
