
"""
V6 pilot dataset generator for CFRP laminate failure prediction.

Run from the project root:
    python -m src.generate_dataset_v6

Creates:
    data/processed/failure_dataset_v6_pilot.csv
    data/processed/failure_dataset_v6_pilot_summary.csv

This is a computational pilot, not experimental validation.
It does not modify V5 files or models.
"""

from __future__ import annotations

from pathlib import Path
import itertools

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

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "failure_dataset_v6_pilot.csv"
)
SUMMARY_PATH = (
    ROOT / "data" / "processed" / "failure_dataset_v6_pilot_summary.csv"
)

SEED = 20261025
SAMPLES_PER_CONFIGURATION = 8

# Only material combinations with documented strengths in strengths.py.
MATERIAL_SYSTEMS = [
    ("T300", "Standard Epoxy"),
    ("AS4", "3501-6 Epoxy"),
    ("IM7", "Toughened Epoxy"),
]

# Include the existing V5 layup and two additional configurations.
LAYUPS = [
    (0, 45, -45, 90, 90, -45, 45, 0),
    (0, 0, 45, -45, 90, 90, -45, 45, 0, 0),
    (0, 45, 90, -45, 0, -45, 90, 45),
]

FIBER_VOLUME_FRACTIONS = [0.50, 0.60]
PLY_THICKNESSES_M = [0.000100, 0.000125]

CRITERIA = {
    "Maximum Stress": "max_stress",
    "Maximum Strain": "max_strain",
    "Tsai-Hill": "tsai_hill",
    "Tsai-Wu": "tsai_wu",
    "Hoffman": "hoffman",
    "Hashin": "hashin",
    "Puck": "puck",
}

TARGETS = [f"{prefix}_fi" for prefix in CRITERIA.values()]


def get_material_data(fiber_name, matrix_name, vf):
    """Calculate effective lamina properties and retrieve strengths."""

    fiber = get_fiber(fiber_name)
    matrix = get_matrix(matrix_name)

    E1, E2, G12, nu12 = calculate_lamina_properties(
        Ef=fiber["E"],
        Em=matrix["E"],
        Gf=fiber["G"],
        Gm=matrix["G"],
        nu_f=fiber["nu"],
        nu_m=matrix["nu"],
        Vf=vf,
    )

    strength_data = find_lamina_strengths(fiber_name, matrix_name)

    if strength_data is None:
        raise ValueError(
            f"No documented strength data for "
            f"{fiber_name} / {matrix_name}"
        )

    strengths = {
        key: float(strength_data[key])
        for key in ("Xt", "Xc", "Yt", "Yc", "S")
    }

    properties = {
        "E1": float(E1),
        "E2": float(E2),
        "G12": float(G12),
        "nu12": float(nu12),
    }

    return properties, strengths, strength_data


def make_load_case(rng, scale_group):
    """
    Generate a six-component load vector.

    Force and moment components use separate scales.
    This pilot tests the pipeline; it does not guarantee
    balanced coverage of all seven failure-index ranges.
    """

    force_scale = 10.0 ** rng.uniform(3.5, 6.0) * scale_group
    moment_scale = 10.0 ** rng.uniform(0.0, 3.0) * scale_group

    return np.array(
        [
            rng.uniform(-1, 1) * force_scale,
            rng.uniform(-1, 1) * force_scale,
            rng.uniform(-1, 1) * force_scale,
            rng.uniform(-1, 1) * moment_scale,
            rng.uniform(-1, 1) * moment_scale,
            rng.uniform(-1, 1) * moment_scale,
        ],
        dtype=float,
    )


def evaluate_case(properties, strengths, layup, thickness, load):
    """Run the classical solver and collect governing failure indices."""

    result = evaluate_laminate_failure(
        E1=properties["E1"],
        E2=properties["E2"],
        G12=properties["G12"],
        nu12=properties["nu12"],
        ply_angles=list(layup),
        ply_thickness=thickness,
        N=load[:3],
        M=load[3:],
        strengths=strengths,
    )

    values = {}

    for criterion, prefix in CRITERIA.items():
        governing = find_governing_failure(result, criterion)
        criterion_result = governing["criteria"][criterion]
        fi = float(criterion_result["failure_index"])

        if not np.isfinite(fi):
            raise ValueError(
                f"Non-finite FI for {criterion}: {fi}"
            )

        values[f"{prefix}_fi"] = fi
        values[f"{prefix}_failed"] = int(fi >= 1.0)
        values[f"{prefix}_governing_ply"] = int(governing["ply"])
        values[f"{prefix}_governing_angle"] = float(
            governing["angle"]
        )
        values[f"{prefix}_governing_surface"] = str(
            governing["surface"]
        )

        if "failure_mode" in criterion_result:
            values[f"{prefix}_failure_mode"] = str(
                criterion_result["failure_mode"]
            )

    return values


def main():
    # Never overwrite existing pilot outputs.
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "V6 pilot output already exists. Nothing was overwritten:\n"
            f"{OUTPUT_PATH}\n{SUMMARY_PATH}"
        )

    rng = np.random.default_rng(SEED)
    rows = []
    sample_id = 0

    configurations = list(
        itertools.product(
            MATERIAL_SYSTEMS,
            LAYUPS,
            FIBER_VOLUME_FRACTIONS,
            PLY_THICKNESSES_M,
        )
    )

    print(f"Configurations: {len(configurations)}")
    print(
        "Expected random samples:",
        len(configurations) * SAMPLES_PER_CONFIGURATION,
    )

    for config_number, (
        material_system,
        layup,
        vf,
        thickness,
    ) in enumerate(configurations, start=1):

        fiber_name, matrix_name = material_system

        properties, strengths, strength_data = get_material_data(
            fiber_name, matrix_name, vf
        )

        config_id = (
            f"{fiber_name}_{matrix_name.replace(' ', '')}"
            f"_vf{vf:.2f}"
            f"_t{thickness * 1000:.3f}mm"
            f"_layup{config_number:03d}"
        )

        print(
            f"[{config_number}/{len(configurations)}] "
            f"{config_id}",
            flush=True,
        )

        # One exact zero-load anchor for each configuration.
        # These rows help verify the solver's zero-load behaviour.
        zero_load = np.zeros(6, dtype=float)
        zero_values = evaluate_case(
            properties, strengths, layup, thickness, zero_load
        )

        sample_id += 1
        zero_row = {
            "sample_id": sample_id,
            "configuration_id": config_id,
            "sampling_region": "zero_load_anchor",
            "fiber": fiber_name,
            "matrix": matrix_name,
            "strength_system_id": strength_data["system_id"],
            "Vf": vf,
            **properties,
            **strengths,
            "ply_thickness_m": thickness,
            "number_of_plies": len(layup),
            "layup_angles_deg": ",".join(map(str, layup)),
            "Nx": 0.0,
            "Ny": 0.0,
            "Nxy": 0.0,
            "Mx": 0.0,
            "My": 0.0,
            "Mxy": 0.0,
            **zero_values,
        }
        rows.append(zero_row)

        # Random load cases, using reproducible sampling.
        for case_index in range(SAMPLES_PER_CONFIGURATION):
            # Spread load magnitudes over several orders of magnitude.
            scale_group = 10.0 ** rng.uniform(-0.5, 0.5)
            load = make_load_case(rng, scale_group)

            target_values = evaluate_case(
                properties, strengths, layup, thickness, load
            )

            sample_id += 1
            row = {
                "sample_id": sample_id,
                "configuration_id": config_id,
                "sampling_region": "pilot_random_load",
                "fiber": fiber_name,
                "matrix": matrix_name,
                "strength_system_id": strength_data["system_id"],
                "Vf": vf,
                **properties,
                **strengths,
                "ply_thickness_m": thickness,
                "number_of_plies": len(layup),
                "layup_angles_deg": ",".join(map(str, layup)),
                "Nx": float(load[0]),
                "Ny": float(load[1]),
                "Nxy": float(load[2]),
                "Mx": float(load[3]),
                "My": float(load[4]),
                "Mxy": float(load[5]),
                **target_values,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Basic dataset integrity checks.
    if df.empty:
        raise ValueError("Generated dataset is empty.")

    numeric_columns = [
        "Vf", "E1", "E2", "G12", "nu12",
        "ply_thickness_m", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy",
        *TARGETS,
    ]

    numeric_data = df[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_data).all():
        raise ValueError("Dataset contains NaN or infinite numeric values.")

    for prefix in CRITERIA.values():
        expected_failed = (df[f"{prefix}_fi"] >= 1.0).astype(int)
        if not np.array_equal(
            expected_failed.to_numpy(),
            df[f"{prefix}_failed"].to_numpy(),
        ):
            raise ValueError(f"Failure label mismatch for {prefix}")

    zero_rows = df[df["sampling_region"] == "zero_load_anchor"]
    if len(zero_rows) != len(configurations):
        raise ValueError("Unexpected number of zero-load anchor rows.")

    if not np.allclose(
        zero_rows[TARGETS].to_numpy(dtype=float),
        0.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "At least one zero-load anchor has a non-zero failure index. "
            "Inspect the solver before using this dataset."
        )

    # Summary statistics per target, including failed cases.
    summary_rows = []

    for target in TARGETS:
        values = df[target]
        summary_rows.append(
            {
                "target": target,
                "min": float(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
                "count_ge_1": int((values >= 1.0).sum()),
                "count_below_1": int((values < 1.0).sum()),
                "count_near_boundary": int(
                    values.between(0.90, 1.10).sum()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Create output only after all checks pass.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "A V6 pilot output appeared during generation; refusing to overwrite."
        )

    df.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nV6 PILOT DATASET CREATED")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Unique configurations: {df['configuration_id'].nunique()}")

    print("\nRows per material system:")
    print(df.groupby(["fiber", "matrix"]).size().to_string())

    print("\nFailure-index summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print(
        "\nImportant: this is a computational pilot. "
        "Review failure-index coverage and grouped validation "
        "before training any V6 model."
    )


if __name__ == "__main__":
    main()
