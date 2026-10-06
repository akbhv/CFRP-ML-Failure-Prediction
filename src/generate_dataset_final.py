import numpy as np
import pandas as pd
from pathlib import Path

from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import find_hashin_failure_scale
from src.laminate_failure_comparison import evaluate_laminate_failure, find_governing_failure


# ==================================================
# MATERIAL
# ==================================================

from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths


FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
Vf = 0.60


# Get constituent properties from the canonical material database
fiber = get_fiber(FIBER_NAME)
matrix = get_matrix(MATRIX_NAME)


# Calculate effective lamina properties using the same
# micromechanics model used by the application
E1, E2, G12, nu12 = calculate_lamina_properties(
    fiber["E"],
    matrix["E"],
    fiber["G"],
    matrix["G"],
    fiber["nu"],
    matrix["nu"],
    Vf
)


# Get the verified lamina strength dataset corresponding
# to the selected fiber/matrix combination
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
        p=[0.15, 0.25, 0.20, 0.25, 0.15]
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
    print("GENERATING FINAL DATASET V3")
    print("-----------------------------------")
    print("Seven failure-criterion outputs enabled")
    print("-----------------------------------")

    while len(rows) < NUMBER_OF_SAMPLES:
        attempts += 1

        # ------------------------------------------
        # Random loading direction
        # ------------------------------------------

        load_direction = generate_random_load_direction(rng)

        # ------------------------------------------
        # Hashin failure boundary for boundary-aware sampling
        # ------------------------------------------

        failure_scale, boundary_result = find_hashin_failure_scale(
            E1,
            E2,
            G12,
            nu12,
            strengths,
            ply_angles,
            ply_thickness,
            load_direction
        )

        if failure_scale is None:
            boundary_failures += 1
            continue

        # ------------------------------------------
        # Sample around Hashin boundary
        # ------------------------------------------

        multiplier, sampling_region = generate_multiplier(rng)
        scale = failure_scale * multiplier
        load_vector = scale * load_direction

        N = load_vector[:3]
        M = load_vector[3:]

        # ------------------------------------------
        # Full physics evaluation: ALL 7 criteria
        # ------------------------------------------

        laminate_result = evaluate_laminate_failure(
            E1,
            E2,
            G12,
            nu12,
            ply_angles,
            ply_thickness,
            N,
            M,
            strengths,
        )

        criteria = [
            "Maximum Stress",
            "Maximum Strain",
            "Tsai-Hill",
            "Tsai-Wu",
            "Hoffman",
            "Hashin",
            "Puck",
        ]

        governing = {
            criterion: find_governing_failure(laminate_result, criterion)
            for criterion in criteria
        }

        # ------------------------------------------
        # Store sample
        # ------------------------------------------

        row = {
            # Inputs
            "Nx": N[0],
            "Ny": N[1],
            "Nxy": N[2],
            "Mx": M[0],
            "My": M[1],
            "Mxy": M[2],

            # Dataset metadata
            "boundary_scale": failure_scale,
            "load_multiplier": multiplier,
            "sampling_region": sampling_region,
        }

        # Store FI + classification + governing location for every criterion.
        criterion_columns = {
            "Maximum Stress": "max_stress",
            "Maximum Strain": "max_strain",
            "Tsai-Hill": "tsai_hill",
            "Tsai-Wu": "tsai_wu",
            "Hoffman": "hoffman",
            "Hashin": "hashin",
            "Puck": "puck",
        }

        for criterion, prefix in criterion_columns.items():
            result = governing[criterion]
            criterion_data = result["criteria"][criterion]

            row[f"{prefix}_fi"] = float(criterion_data["failure_index"])
            row[f"{prefix}_failed"] = int(criterion_data["failed"])
            row[f"{prefix}_ply"] = int(result["ply"])
            row[f"{prefix}_angle"] = float(result["angle"])
            row[f"{prefix}_surface"] = result["surface"]

            # Hashin and Puck expose mode information.
            if criterion == "Hashin":
                row["hashin_mode"] = criterion_data["failure_mode"]
            elif criterion == "Puck":
                row["puck_mode"] = criterion_data["failure_mode"]
                row["puck_fracture_angle"] = float(criterion_data["fracture_angle"])

        # Preserve Hashin as the primary classification target for the ML pipeline.
        row["failed"] = row["hashin_failed"]

        rows.append(row)

        if len(rows) % 1000 == 0:
            print(f"Generated {len(rows)}/{NUMBER_OF_SAMPLES}")

    # ==================================================
    # DATAFRAME
    # ==================================================

    df = pd.DataFrame(rows)

    # ==================================================
    # SAVE
    # ==================================================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # ==================================================
    # SUMMARY
    # ==================================================

    failed_count = int(df["failed"].sum())
    safe_count = len(df) - failed_count

    print()
    print("-----------------------------------")
    print("FINAL DATASET V3 COMPLETE")
    print("-----------------------------------")
    print(f"Samples generated = {len(df)}")
    print(f"Total attempts = {attempts}")
    print(f"Boundary search failures = {boundary_failures}")
    print()
    print(f"Safe samples = {safe_count}")
    print(f"Failed samples = {failed_count}")
    print(f"Failure rate = {failed_count / len(df) * 100:.2f}%")
    print()

    for criterion, prefix in {
        "Maximum Stress": "max_stress",
        "Maximum Strain": "max_strain",
        "Tsai-Hill": "tsai_hill",
        "Tsai-Wu": "tsai_wu",
        "Hoffman": "hoffman",
        "Hashin": "hashin",
        "Puck": "puck",
    }.items():
        print(
            f"{criterion:16s} FI: "
            f"min={df[f'{prefix}_fi'].min():.6f}, "
            f"max={df[f'{prefix}_fi'].max():.6f}, "
            f"mean={df[f'{prefix}_fi'].mean():.6f}"
        )

    print()
    print("Sampling regions:")
    print(df["sampling_region"].value_counts())

    print()
    print("Hashin failure modes:")
    print(df["hashin_mode"].value_counts())

    print()
    print("Puck failure modes:")
    print(df["puck_mode"].value_counts())

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
