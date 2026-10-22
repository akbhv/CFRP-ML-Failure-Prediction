
import pandas as pd

from src.generate_dataset_final import (
    E1, E2, G12, nu12, ply_angles, ply_thickness, strengths
)
from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)

DATA_PATH = "data/processed/failure_dataset_v3_scaled.csv"
df = pd.read_csv(DATA_PATH)

# Select representative cases with nonzero combined loading.
indices = [0, 1, 2, 100, 1000]

for idx in indices:
    row = df.iloc[idx]
    N = [row["Nx"], row["Ny"], row["Nxy"]]
    M = [row["Mx"], row["My"], row["Mxy"]]

    result = evaluate_laminate_failure(
        E1, E2, G12, nu12,
        ply_angles, ply_thickness,
        N, M, strengths,
    )

    print(f"\n{'=' * 65}")
    print(f"Dataset row: {idx}")
    print(f"Loads N={N}, M={M}")

    for criterion in ["Tsai-Wu", "Hoffman"]:
        gov = find_governing_failure(result, criterion)
        criterion_data = gov["criteria"][criterion]

        print(
            f"{criterion}: "
            f"FI={criterion_data['failure_index']:.10f}, "
            f"ply={gov['ply']}, "
            f"angle={gov['angle']}, "
            f"surface={gov['surface']}"
        )

    print(
        "Stored values: "
        f"Tsai-Wu={row['tsai_wu_fi']:.10f}, "
        f"Hoffman={row['hoffman_fi']:.10f}"
    )
