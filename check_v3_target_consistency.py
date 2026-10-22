
import numpy as np
import pandas as pd

from src.failure_criteria import tsai_wu_failure, hoffman_failure
from src.generate_dataset_final import strengths

DATASET = "data/processed/failure_dataset_v3_scaled.csv"

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Find the stress columns if the dataset stores local ply stresses.
stress_candidates = [
    ("sigma_1", "sigma_2", "tau_12"),
    ("sigma1", "sigma2", "tau12"),
]

stress_columns = next(
    (cols for cols in stress_candidates if all(c in df.columns for c in cols)),
    None,
)

if stress_columns is None:
    print(
        "\nNo obvious local-stress columns found. "
        "We need to inspect how the dataset generator computes its targets."
    )
    print("\nFirst 10 column names:")
    print(df.columns.tolist()[:10])
else:
    errors = []
    for _, row in df.iterrows():
        stress = row[list(stress_columns)].to_numpy(dtype=float)
        tw, _ = tsai_wu_failure(stress, strengths)
        ho, _ = hoffman_failure(stress, strengths)
        errors.append((tw, ho))

    calculated = np.asarray(errors)
    print("\nRecomputed Tsai-Wu vs Hoffman")
    print("Mean absolute difference:", np.mean(np.abs(calculated[:, 0] - calculated[:, 1])))
    print("Maximum absolute difference:", np.max(np.abs(calculated[:, 0] - calculated[:, 1])))
