import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths
from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)

path = "data/processed/failure_dataset_v2_hoffman_corrected.csv"
df = pd.read_csv(path)

fiber = get_fiber("T300")
matrix = get_matrix("Standard Epoxy")

E1, E2, G12, nu12 = calculate_lamina_properties(
    fiber["E"], matrix["E"],
    fiber["G"], matrix["G"],
    fiber["nu"], matrix["nu"], 0.60,
)

strength_data = find_lamina_strengths("T300", "Standard Epoxy")
strengths = {
    k: float(strength_data[k])
    for k in ("Xt", "Xc", "Yt", "Yc", "S")
}

angles = [0, 45, -45, 90, 90, -45, 45, 0]
thickness = 0.125e-3
sample_rows = np.linspace(0, len(df) - 1, 10, dtype=int)

errors = []
location_mismatches = []

for i in sample_rows:
    row = df.iloc[i]
    N = row[["Nx", "Ny", "Nxy"]].to_numpy(dtype=float)
    M = row[["Mx", "My", "Mxy"]].to_numpy(dtype=float)

    result = evaluate_laminate_failure(
        E1, E2, G12, nu12, angles, thickness, N, M, strengths
    )
    governing = find_governing_failure(result, "Hoffman")
    criterion = governing["criteria"]["Hoffman"]

    recalculated_fi = float(criterion["failure_index"])
    saved_fi = float(row["hoffman_fi"])
    errors.append(abs(recalculated_fi - saved_fi))

    expected_location = (
        int(governing["ply"]),
        float(governing["angle"]),
        str(governing["surface"]),
    )
    saved_location = (
        int(row["hoffman_ply"]),
        float(row["hoffman_angle"]),
        str(row["hoffman_surface"]),
    )

    if expected_location != saved_location:
        location_mismatches.append(
            (int(i), expected_location, saved_location)
        )

    print(
        f"Row {i:5d}: saved FI={saved_fi:.12f}, "
        f"recalculated FI={recalculated_fi:.12f}, "
        f"abs error={abs(recalculated_fi-saved_fi):.3e}"
    )

print("\n=== SUMMARY ===")
print("Rows independently checked:", len(sample_rows))
print("Maximum absolute FI difference:", max(errors))
print("Governing-location mismatches:", len(location_mismatches))
for item in location_mismatches:
    print("Location mismatch:", item)

print(
    "RESULT:",
    "PASS" if max(errors) < 1e-10 and not location_mismatches else "INVESTIGATE"
)
