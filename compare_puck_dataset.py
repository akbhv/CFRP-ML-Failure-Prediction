
import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths


DATA_PATH = "data/processed/failure_dataset_v2.csv"
REPORT_PATH = "data/processed/puck_recalculation_comparison.csv"

FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60

PLY_ANGLES = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3


def main():
    df = pd.read_csv(DATA_PATH)

    fiber = get_fiber(FIBER_NAME)
    matrix = get_matrix(MATRIX_NAME)

    E1, E2, G12, nu12 = calculate_lamina_properties(
        fiber["E"],
        matrix["E"],
        fiber["G"],
        matrix["G"],
        fiber["nu"],
        matrix["nu"],
        VF,
    )

    strength_data = find_lamina_strengths(FIBER_NAME, MATRIX_NAME)
    if strength_data is None:
        raise ValueError("Could not find the configured lamina strengths.")

    strengths = {
        key: strength_data[key]
        for key in ("Xt", "Xc", "Yt", "Yc", "S")
    }

    records = []

    for idx, row in df.iterrows():
        N = row[["Nx", "Ny", "Nxy"]].to_numpy(dtype=float)
        M = row[["Mx", "My", "Mxy"]].to_numpy(dtype=float)

        result = evaluate_laminate_failure(
            E1, E2, G12, nu12,
            PLY_ANGLES, PLY_THICKNESS,
            N, M, strengths,
        )

        governing = find_governing_failure(result, "Puck")
        puck = governing["criteria"]["Puck"]

        new_fi = float(puck["failure_index"])
        old_fi = float(row["puck_fi"])

        records.append({
            "row_index": idx,
            "old_puck_fi": old_fi,
            "new_puck_fi": new_fi,
            "absolute_fi_difference": abs(new_fi - old_fi),
            "old_puck_mode": row["puck_mode"],
            "new_puck_mode": puck["failure_mode"],
            "old_puck_failed": bool(row["puck_failed"]),
            "new_puck_failed": bool(puck["failed"]),
            "old_puck_ply": row["puck_ply"],
            "new_puck_ply": governing["ply"],
            "old_puck_angle": row["puck_angle"],
            "new_puck_angle": governing["angle"],
            "old_fracture_angle": row["puck_fracture_angle"],
            "new_fracture_angle": puck["fracture_angle"],
            "new_fi_finite": bool(np.isfinite(new_fi)),
            "old_puck_surface": row["puck_surface"],
            "new_puck_surface": governing["surface"],
        })

        if (idx + 1) % 1000 == 0:
            print(f"Recomputed {idx + 1}/{len(df)} rows")

    report = pd.DataFrame(records)
    report.to_csv(REPORT_PATH, index=False)

    finite = np.isfinite(report["new_puck_fi"])
    abs_diff = report.loc[finite, "absolute_fi_difference"]

    print("\n=== Puck dataset comparison ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Finite recomputed FI: {int(finite.sum())}")
    print(f"Non-finite recomputed FI: {int((~finite).sum())}")
    print(f"Mode changes: {(report['old_puck_mode'] != report['new_puck_mode']).sum()}")
    print(f"Failed-label changes: {(report['old_puck_failed'] != report['new_puck_failed']).sum()}")
    print(f"Governing-ply changes: {(report['old_puck_ply'] != report['new_puck_ply']).sum()}")
    print(f"Mean absolute FI difference: {abs_diff.mean():.8g}")
    print(f"Median absolute FI difference: {abs_diff.median():.8g}")
    print(f"Maximum absolute FI difference: {abs_diff.max():.8g}")

    print("\nLargest differences:")
    print(
        report.nlargest(10, "absolute_fi_difference")[
            [
                "row_index",
                "old_puck_fi",
                "new_puck_fi",
                "absolute_fi_difference",
                "old_puck_mode",
                "new_puck_mode",
                "old_puck_ply",
                "new_puck_ply",
            ]
        ].to_string(index=False)
    )

    print(f"\nRead-only comparison report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
