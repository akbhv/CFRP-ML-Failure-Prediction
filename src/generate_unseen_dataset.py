import numpy as np
import pandas as pd

from pathlib import Path

from src.dataset_generator import (
    find_hashin_failure_scale,
    evaluate_laminate_case
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 20260926
NUMBER_OF_SAMPLES = 5000


# ============================================================
# MATERIAL PROPERTIES
# ============================================================

from src.materials import get_fiber, get_matrix
from src.micromechanics import calculate_lamina_properties
from src.strengths import find_lamina_strengths


FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
Vf = 0.60


# Get constituent properties from the canonical material database
fiber = get_fiber(FIBER_NAME)
matrix = get_matrix(MATRIX_NAME)


# Calculate effective lamina properties using the same
# micromechanics model used by the main application
E1, E2, G12, nu12 = calculate_lamina_properties(
    fiber["E"],
    matrix["E"],
    fiber["G"],
    matrix["G"],
    fiber["nu"],
    matrix["nu"],
    Vf
)


# Get the verified lamina strength dataset
strength_data = find_lamina_strengths(
    FIBER_NAME,
    MATRIX_NAME
)

if strength_data is None:
    raise ValueError(
        f"No verified lamina strength dataset exists for "
        f"{FIBER_NAME}/{MATRIX_NAME}"
    )


strengths = {
    "Xt": strength_data["Xt"],
    "Xc": strength_data["Xc"],
    "Yt": strength_data["Yt"],
    "Yc": strength_data["Yc"],
    "S": strength_data["S"]
}

# ============================================================
# LAMINATE
# ============================================================

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


# ============================================================
# SAMPLING REGIONS
# ============================================================

regions = {
    "low_safe": (0.10, 0.60),
    "near_boundary_safe": (0.60, 0.95),
    "boundary": (0.95, 1.05),
    "near_boundary_failed": (1.05, 1.20),
    "high_failed": (1.20, 1.50)
}

region_names = list(regions.keys())

region_probabilities = [
    0.15,
    0.25,
    0.20,
    0.25,
    0.15
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "data/processed/unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "unseen_generalization_dataset.csv"
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# GENERATE DATASET
# ============================================================

rows = []

boundary_search_failures = 0


print("-----------------------------------")
print("UNSEEN GENERALIZATION DATASET")
print("-----------------------------------")

print()
print(
    f"Samples requested : "
    f"{NUMBER_OF_SAMPLES}"
)

print(
    f"Random seed       : "
    f"{RANDOM_SEED}"
)

print()


for i in range(NUMBER_OF_SAMPLES):

    # --------------------------------------------------------
    # Random 6D load direction
    # --------------------------------------------------------

    load_direction = rng.normal(
        size=6
    )

    load_direction /= np.linalg.norm(
        load_direction
    )


    # --------------------------------------------------------
    # Find first Hashin failure
    # --------------------------------------------------------

    boundary_scale, boundary_result = (
        find_hashin_failure_scale(
            E1,
            E2,
            G12,
            nu12,
            strengths,
            ply_angles,
            ply_thickness,
            load_direction
        )
    )


    if boundary_scale is None:

        boundary_search_failures += 1

        continue


    # --------------------------------------------------------
    # Choose sampling region
    # --------------------------------------------------------

    region = rng.choice(
        region_names,
        p=region_probabilities
    )

    multiplier_min, multiplier_max = (
        regions[region]
    )

    load_multiplier = rng.uniform(
        multiplier_min,
        multiplier_max
    )


    # --------------------------------------------------------
    # Generate load vector
    # --------------------------------------------------------

    load_vector = (
        boundary_scale
        * load_multiplier
        * load_direction
    )

    N = load_vector[:3]
    M = load_vector[3:]


    # --------------------------------------------------------
    # Evaluate physics model
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    rows.append({
        "Nx": N[0],
        "Ny": N[1],
        "Nxy": N[2],
        "Mx": M[0],
        "My": M[1],
        "Mxy": M[2],

        "boundary_scale": boundary_scale,

        "load_multiplier": load_multiplier,

        "sampling_region": region,

        "hashin_fi": result["hashin_fi"],

        "max_stress_fi":
            result["max_stress_fi"],

        "tsai_hill_fi":
            result["tsai_hill_fi"],

        "tsai_wu_fi":
            result["tsai_wu_fi"],

        "failed":
            int(result["failed"]),

        "hashin_mode":
            result["hashin_mode"],

        "hashin_ply":
            result["hashin_ply"],

        "hashin_surface":
            result["hashin_surface"]
    })


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("-----------------------------------")
print("GENERATION COMPLETE")
print("-----------------------------------")

print()
print(
    f"Generated samples : "
    f"{len(df)}"
)

print(
    f"Boundary failures : "
    f"{boundary_search_failures}"
)

print()

print(
    f"Safe samples      : "
    f"{(df['failed'] == 0).sum()}"
)

print(
    f"Failed samples    : "
    f"{(df['failed'] == 1).sum()}"
)

print(
    f"Failure rate      : "
    f"{df['failed'].mean():.4%}"
)

print()

print(
    f"Hashin FI min     : "
    f"{df['hashin_fi'].min():.6f}"
)

print(
    f"Hashin FI max     : "
    f"{df['hashin_fi'].max():.6f}"
)

print(
    f"Hashin FI mean    : "
    f"{df['hashin_fi'].mean():.6f}"
)

print()

print("Failure modes:")
print(
    df["hashin_mode"]
    .value_counts()
)

print()

print(
    f"Saved to: {OUTPUT_FILE}"
)