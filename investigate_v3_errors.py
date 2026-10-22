
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(
    ROOT / "data" / "processed" / "failure_dataset_v3_scaled.csv"
)
RESULTS = ROOT / "results" / "multitarget_dnn_v3"

TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]

for target in TARGETS:
    path = RESULTS / target / "test_predictions.csv"
    pred_df = pd.read_csv(path)

    row_idx = pred_df["row_index"].to_numpy(dtype=int)
    actual = pred_df["reference_fi"].to_numpy(dtype=float)
    predicted = pred_df["predicted_fi"].to_numpy(dtype=float)

    error = predicted - actual
    abs_error = np.abs(error)

    false_safe = (actual >= 1.0) & (predicted < 1.0)
    false_failure = (actual < 1.0) & (predicted >= 1.0)

    print(f"\n{'=' * 65}")
    print(f"TARGET: {target}")
    print(f"False-safe: {false_safe.sum()}")
    print(f"False-failure: {false_failure.sum()}")
    print(f"Maximum absolute error: {abs_error.max():.6f}")

    # Show the five largest errors.
    worst = np.argsort(abs_error)[-5:][::-1]
    print("\nFive largest absolute errors:")
    for i in worst:
        print(
            f"row={row_idx[i]}, actual={actual[i]:.6f}, "
            f"predicted={predicted[i]:.6f}, "
            f"error={error[i]:+.6f}"
        )

    # Show all false-safe cases, including the applied load resultants.
    bad = np.where(false_safe)[0]
    if len(bad):
        cols = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
        print("\nFalse-safe cases:")
        for i in bad:
            loads = DATA.iloc[row_idx[i]][cols].to_dict()
            print(
                f"row={row_idx[i]}, actual={actual[i]:.6f}, "
                f"predicted={predicted[i]:.6f}, loads={loads}"
            )

print("\nInvestigation complete. No files were modified.")
