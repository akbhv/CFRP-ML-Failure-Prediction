
import numpy as np
import pandas as pd

from src.generate_dataset_final import (
    E1, E2, G12, nu12, ply_angles, ply_thickness, strengths
)
from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)

df = pd.read_csv("data/processed/failure_dataset_v3_scaled.csv")

max_equation_error = 0.0
max_governing_error = 0.0
checked_surfaces = 0

for row_index, row in df.iterrows():
    N = [row["Nx"], row["Ny"], row["Nxy"]]
    M = [row["Mx"], row["My"], row["Mxy"]]

    result = evaluate_laminate_failure(
        E1, E2, G12, nu12,
        ply_angles, ply_thickness,
        N, M, strengths,
    )

    for ply_result in result["ply_results"]:
        sigma_1, sigma_2, tau_12 = ply_result["stress_local"]

        tw = ply_result["criteria"]["Tsai-Wu"]["failure_index"]
        hf = ply_result["criteria"]["Hoffman"]["failure_index"]

        # With F12=0, this is the expected same-stress difference.
        expected_difference = (
            -sigma_1 * sigma_2 / (strengths["Xt"] * strengths["Xc"])
        )

        equation_error = abs((hf - tw) - expected_difference)
        max_equation_error = max(max_equation_error, equation_error)
        checked_surfaces += 1

    tw_gov = find_governing_failure(result, "Tsai-Wu")
    hf_gov = find_governing_failure(result, "Hoffman")

    max_governing_error = max(
        max_governing_error,
        abs(
            tw_gov["criteria"]["Tsai-Wu"]["failure_index"]
            - row["tsai_wu_fi"]
        ),
        abs(
            hf_gov["criteria"]["Hoffman"]["failure_index"]
            - row["hoffman_fi"]
        ),
    )

    if row_index % 2000 == 0:
        print(f"Checked {row_index} / {len(df)} rows")

print("\n=== Verification summary ===")
print(f"Dataset rows: {len(df)}")
print(f"Ply surfaces checked: {checked_surfaces}")
print(f"Maximum same-stress equation error: {max_equation_error:.3e}")
print(f"Maximum stored-target mismatch: {max_governing_error:.3e}")

if max_equation_error < 1e-10 and max_governing_error < 1e-8:
    print("PASS: equation relationship and stored targets are consistent.")
else:
    print("INVESTIGATE: at least one verification tolerance was exceeded.")
