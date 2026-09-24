import numpy as np
import pandas as pd
from pathlib import Path

from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import find_hashin_failure_scale
from src.dataset_generator import evaluate_laminate_case


# ==================================================
# MATERIAL
# ==================================================

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

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# ==================================================
# LAMINATE
# ==================================================

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


# ==================================================
# DATASET SETTINGS
# ==================================================

NUMBER_OF_SAMPLES = 20000
RANDOM_SEED = 2026

OUTPUT_PATH = Path(
    "data/processed/failure_dataset_v2.csv"
)


# ==================================================
# RANDOM LOAD DIRECTION
# ==================================================

def generate_random_load_direction(rng):

    direction = rng.normal(size=6)

    norm = np.linalg.norm(direction)

    while norm == 0:
        direction = rng.normal(size=6)
        norm = np.linalg.norm(direction)

    return direction / norm


# ==================================================
# SAMPLE MULTIPLIER
# ==================================================

def generate_multiplier(rng):

    region = rng.choice(
        [
            "low_safe",
            "near_boundary_safe",
            "boundary",
            "near_boundary_failed",
            "high_failed"
        ],
        p=[
            0.15,
            0.25,
            0.20,
            0.25,
            0.15
        ]
    )

    if region == "low_safe":
        multiplier = rng.uniform(0.10, 0.60)

    elif region == "near_boundary_safe":
        multiplier = rng.uniform(0.60, 0.95)

    elif region == "boundary":
        multiplier = rng.uniform(0.95, 1.05)

    elif region == "near_boundary_failed":
        multiplier = rng.uniform(1.05, 1.20)

    else:
        multiplier = rng.uniform(1.20, 1.50)

    return multiplier, region


# ==================================================
# DATASET GENERATION
# ==================================================

def generate_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    attempts = 0
    boundary_failures = 0

    print("-----------------------------------")
    print("GENERATING FINAL DATASET V2")
    print("-----------------------------------")

    while len(rows) < NUMBER_OF_SAMPLES:

        attempts += 1

        # ------------------------------------------
        # Random loading direction
        # ------------------------------------------

        load_direction = generate_random_load_direction(rng)

        # ------------------------------------------
        # Failure boundary
        # ------------------------------------------

        failure_scale, boundary_result = (
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

        if failure_scale is None:
            boundary_failures += 1
            continue

        # ------------------------------------------
        # Sample around boundary
        # ------------------------------------------

        multiplier, sampling_region = (
            generate_multiplier(rng)
        )

        scale = failure_scale * multiplier

        load_vector = scale * load_direction

        N = load_vector[:3]
        M = load_vector[3:]

        # ------------------------------------------
        # Physics evaluation
        # ------------------------------------------

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

        # ------------------------------------------
        # Store sample
        # ------------------------------------------

        rows.append({

            # ------------------------------
            # Inputs
            # ------------------------------

            "Nx": N[0],
            "Ny": N[1],
            "Nxy": N[2],

            "Mx": M[0],
            "My": M[1],
            "Mxy": M[2],

            # ------------------------------
            # Dataset metadata
            # ------------------------------

            "boundary_scale": failure_scale,

            "load_multiplier": multiplier,

            "sampling_region": sampling_region,

            # ------------------------------
            # Regression targets
            # ------------------------------

            "hashin_fi": result["hashin_fi"],

            "max_stress_fi": result["max_stress_fi"],

            "tsai_hill_fi": result["tsai_hill_fi"],

            "tsai_wu_fi": result["tsai_wu_fi"],

            # ------------------------------
            # Classification target
            # ------------------------------

            "failed": int(result["failed"]),

            # ------------------------------
            # Failure information
            # ------------------------------

            "hashin_mode": result["hashin_mode"],

            "hashin_ply": result["hashin_ply"],

            "hashin_surface": result["hashin_surface"],
        })

        if len(rows) % 1000 == 0:

            print(
                f"Generated "
                f"{len(rows)}/{NUMBER_OF_SAMPLES}"
            )

    # ==================================================
    # DATAFRAME
    # ==================================================

    df = pd.DataFrame(rows)

    # ==================================================
    # SAVE
    # ==================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    failed_count = int(df["failed"].sum())

    safe_count = len(df) - failed_count

    print()
    print("-----------------------------------")
    print("FINAL DATASET V2 COMPLETE")
    print("-----------------------------------")

    print(
        f"Samples generated = {len(df)}"
    )

    print(
        f"Total attempts = {attempts}"
    )

    print(
        f"Boundary search failures = "
        f"{boundary_failures}"
    )

    print()

    print(
        f"Safe samples = {safe_count}"
    )

    print(
        f"Failed samples = {failed_count}"
    )

    print(
        f"Failure rate = "
        f"{failed_count / len(df) * 100:.2f}%"
    )

    print()

    print(
        f"Hashin FI minimum = "
        f"{df['hashin_fi'].min():.6f}"
    )

    print(
        f"Hashin FI maximum = "
        f"{df['hashin_fi'].max():.6f}"
    )

    print(
        f"Hashin FI mean = "
        f"{df['hashin_fi'].mean():.6f}"
    )

    print()

    print("Sampling regions:")
    print(
        df["sampling_region"].value_counts()
    )

    print()

    print("Failure modes:")
    print(
        df["hashin_mode"].value_counts()
    )

    print()

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    generate_dataset()