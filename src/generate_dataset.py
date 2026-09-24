import numpy as np
import pandas as pd
from pathlib import Path
from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import evaluate_laminate_case


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------
# 2. Material properties
# --------------------------------------------------

Ef = 230e9
Gf = 30e9
nu_f = 0.20

Em = 3.5e9
Gm = 1.3e9
nu_m = 0.35

Vf = 0.60


E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef,
    Em,
    Gf,
    Gm,
    nu_f,
    nu_m,
    Vf
)


# --------------------------------------------------
# 3. Material strengths
# --------------------------------------------------

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# --------------------------------------------------
# 4. Laminate definition
# --------------------------------------------------

ply_angles = [
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0
]

ply_thickness = 0.125e-3


# --------------------------------------------------
# 5. Dataset settings
# --------------------------------------------------

N_SAMPLES = 20000


# --------------------------------------------------
# 6. Generate loading cases
# --------------------------------------------------

rows = []


for _ in range(N_SAMPLES):

    # Random loading direction
    load_direction = rng.normal(
        size=6
    )

    # Normalize the loading vector
    norm = np.linalg.norm(load_direction)

    load_direction = load_direction / norm

    # Random load magnitude
    magnitude = rng.uniform(
        1.0e4,
        5.0e5
    )

    load_vector = (
        load_direction * magnitude
    )

    # Membrane loads
    N = np.array([
        load_vector[0],
        load_vector[1],
        load_vector[2]
    ])

    # Bending/twisting moments
    M = np.array([
        load_vector[3],
        load_vector[4],
        load_vector[5]
    ])

    # --------------------------------------------------
    # Evaluate physics
    # --------------------------------------------------

    result = evaluate_laminate_case(
        E1,
        E2,
        G12,
        nu12,
        strengths,
        ply_angles,
        ply_thickness,
        N,
        M
    )

    # --------------------------------------------------
    # Store dataset row
    # --------------------------------------------------

    row = {
        "Nx": N[0],
        "Ny": N[1],
        "Nxy": N[2],
        "Mx": M[0],
        "My": M[1],
        "Mxy": M[2],

        "max_stress_fi":
            result["max_stress_fi"],

        "tsai_hill_fi":
            result["tsai_hill_fi"],

        "tsai_wu_fi":
            result["tsai_wu_fi"],

        "hashin_fi":
            result["hashin_fi"],

        "hashin_mode":
            result["hashin_mode"],

        "failed":
            result["failed"],

        "hashin_ply":
            result["hashin_ply"],

        "hashin_surface":
            result["hashin_surface"]
    }

    rows.append(row)


# --------------------------------------------------
# 7. Create DataFrame
# --------------------------------------------------

dataset = pd.DataFrame(rows)


# --------------------------------------------------
# 8. Save dataset
# --------------------------------------------------

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "failure_dataset_v1.csv"

dataset.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print("-----------------------------------")
print("DATASET GENERATION COMPLETE")
print("-----------------------------------")

print(f"Samples generated: {len(dataset)}")
print(f"Columns: {len(dataset.columns)}")

print(
    f"Failed samples: "
    f"{dataset['failed'].sum()}"
)

print(
    f"Safe samples: "
    f"{(~dataset['failed']).sum()}"
)

print(
    f"Failure rate: "
    f"{dataset['failed'].mean():.2%}"
)

print(
    f"\nHashin FI range: "
    f"{dataset['hashin_fi'].min():.6f}"
    f" → "
    f"{dataset['hashin_fi'].max():.6f}"
)

print(
    f"\nSaved to: "
    f"{output_path}"
)