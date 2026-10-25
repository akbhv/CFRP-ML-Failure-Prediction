"""Generate a versioned CFRP failure dataset with improved low-load coverage.

Run from the project root:
    python -m src.generate_dataset_v4b

This script creates a NEW dataset and refuses to overwrite existing files.
It does not modify V4 or any trained model artifacts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "failure_dataset_v4b.csv"

N_SAMPLES = 20_000
SEED = 20261025

FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60
PLY_ANGLES = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
CRITERIA = {
    "Maximum Stress": "max_stress",
    "Maximum Strain": "max_strain",
    "Tsai-Hill": "tsai_hill",
    "Tsai-Wu": "tsai_wu",
    "Hoffman": "hoffman",
    "Hashin": "hashin",
    "Puck": "puck",
}

# Explicit coverage of near-zero load magnitudes. Each band is sampled
# independently and symmetrically about zero.
LOW_LOAD_BANDS = [
    # label, number of samples, force component limit (N/m), moment component limit (N)
    ("very_low_load", 2_000, 10.0, 1.0),
    ("low_load_100", 2_000, 100.0, 10.0),
    ("low_load_1000", 2_000, 1_000.0, 10.0),
]


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {OUTPUT_PATH}"
        )

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
        raise ValueError(
            f"No strength dataset for {FIBER_NAME}/{MATRIX_NAME}"
        )
    strengths = {
        key: float(strength_data[key])
        for key in ["Xt", "Xc", "Yt", "Yc", "S"]
    }

    rng = np.random.default_rng(SEED)
    load_cases = [(np.zeros(6, dtype=float), "zero_load")]

    # Guarantee a large number of genuinely very-small loads rather than
    # hoping uniform samples in a broad interval land near zero.
    for region, count, force_limit, moment_limit in LOW_LOAD_BANDS:
        for _ in range(count):
            load = np.array(
                [
                    rng.uniform(-force_limit, force_limit),
                    rng.uniform(-force_limit, force_limit),
                    rng.uniform(-force_limit, force_limit),
                    rng.uniform(-moment_limit, moment_limit),
                    rng.uniform(-moment_limit, moment_limit),
                    rng.uniform(-moment_limit, moment_limit),
                ],
                dtype=float,
            )
            load_cases.append((load, region))

    # Keep the remaining samples focused on the failure boundary and beyond.
    n_boundary = N_SAMPLES - len(load_cases)
    load_scales = np.array(
        [100_000.0, 50_000.0, 50_000.0, 100.0, 100.0, 100.0]
    )

    def evaluate(load):
        return evaluate_laminate_failure(
            E1,
            E2,
            G12,
            nu12,
            PLY_ANGLES,
            PLY_THICKNESS,
            load[:3],
            load[3:],
            strengths,
        )

    def hashin_fi(result):
        governing = find_governing_failure(result, "Hashin")
        return float(governing["criteria"]["Hashin"]["failure_index"])

    boundary_regions = [
        ("low_safe", 0.10, 0.60),
        ("near_boundary_safe", 0.60, 0.95),
        ("boundary", 0.95, 1.05),
        ("near_boundary_failed", 1.05, 1.20),
        ("high_failed", 1.20, 1.50),
    ]
    region_weights = np.array([1, 1, 1, 1, 1], dtype=float)
    region_weights /= region_weights.sum()

    for sample_number in range(n_boundary):
        direction = rng.normal(size=6)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([1.0, 0, 0, 0, 0, 0])
        else:
            direction /= norm
        direction *= load_scales

        lo, hi = 0.0, 1.0
        result_hi = evaluate(hi * direction)

        while hashin_fi(result_hi) < 1.0 and hi < 1e6:
            lo = hi
            hi = min(hi * 2.0, 1e6)
            result_hi = evaluate(hi * direction)

        if hashin_fi(result_hi) < 1.0:
            load_cases.append((0.5 * direction, "boundary_search_fallback"))
            continue

        for _ in range(80):
            mid = (lo + hi) / 2.0
            if hashin_fi(evaluate(mid * direction)) < 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-5 * max(hi, 1e-12):
                break

        # Select one region first, then draw a multiplier from its range.
        region_index = int(rng.choice(len(boundary_regions), p=region_weights))
        region, multiplier_low, multiplier_high = boundary_regions[region_index]
        multiplier = rng.uniform(multiplier_low, multiplier_high)
        load_cases.append((hi * multiplier * direction, region))

    if len(load_cases) != N_SAMPLES:
        raise ValueError(
            f"Expected {N_SAMPLES} load cases before evaluation; "
            f"got {len(load_cases)}"
        )

    rows = []
    for idx, (load, region) in enumerate(load_cases, start=1):
        result = evaluate(load)
        row = dict(zip(FEATURES, map(float, load)))
        row["sampling_region"] = region

        for criterion, prefix in CRITERIA.items():
            governing = find_governing_failure(result, criterion)
            data = governing["criteria"][criterion]
            fi = float(data["failure_index"])
            if not np.isfinite(fi):
                raise ValueError(
                    f"Non-finite {criterion} FI at sample {idx}: {fi}"
                )

            row[f"{prefix}_fi"] = fi
            row[f"{prefix}_failed"] = int(bool(data["failed"]))
            row[f"{prefix}_ply"] = int(governing["ply"])
            row[f"{prefix}_angle"] = float(governing["angle"])
            row[f"{prefix}_surface"] = str(governing["surface"])

            if criterion == "Hashin":
                row["hashin_mode"] = str(data.get("failure_mode", "unknown"))
            elif criterion == "Puck":
                row["puck_mode"] = str(data.get("failure_mode", "unknown"))
                if "fracture_angle" in data:
                    row["puck_fracture_angle"] = float(data["fracture_angle"])

        row["failed"] = row["hashin_failed"]
        rows.append(row)

        if idx % 1000 == 0:
            print(f"Evaluated {idx}/{N_SAMPLES} samples")

    df = pd.DataFrame(rows)
    if len(df) != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} rows; got {len(df)}")

    target_columns = [f"{prefix}_fi" for prefix in CRITERIA.values()]
    numeric = df[FEATURES + target_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Dataset contains NaN or infinite inputs/targets")

    zero = df.loc[df["sampling_region"] == "zero_load"]
    if len(zero) != 1:
        raise ValueError(f"Expected one zero-load anchor, found {len(zero)}")

    zero_fis = zero[target_columns].iloc[0].to_numpy(dtype=float)
    if not np.allclose(zero_fis, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError(
            f"Zero-load failure indices are not all zero: {zero_fis}"
        )

    # Check labels against the documented FI >= 1 convention.
    for prefix in CRITERIA.values():
        expected = (df[f"{prefix}_fi"] >= 1.0).astype(int)
        if not np.array_equal(expected.to_numpy(), df[f"{prefix}_failed"].to_numpy()):
            raise ValueError(f"Failure-label mismatch for {prefix}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {OUTPUT_PATH}"
        )
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nV4B DATASET CREATED")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("\nSampling regions:")
    print(df["sampling_region"].value_counts().to_string())

    print("\nTarget summary (min/median/mean/max):")
    print(df[target_columns].agg(["min", "median", "mean", "max"]).to_string())

    print("\nLow-load coverage:")
    load_force_max = df[["Nx", "Ny", "Nxy"]].abs().max(axis=1)
    load_moment_max = df[["Mx", "My", "Mxy"]].abs().max(axis=1)
    for force_limit, moment_limit in [(10, 1), (100, 10), (1000, 10)]:
        mask = (load_force_max <= force_limit) & (load_moment_max <= moment_limit)
        print(
            f"forces <= {force_limit:g} N/m and moments <= "
            f"{moment_limit:g} N: {int(mask.sum())}/{len(df)}"
        )


if __name__ == "__main__":
    main()