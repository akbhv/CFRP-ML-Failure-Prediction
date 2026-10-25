
"""Validate selected V4 dataset samples against the running classical API.

Run from the project root:
    python -m src.validate_dataset_v4

This script is read-only: it does not modify the dataset or models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.materials import get_fiber, get_matrix
from src.micromechanics import calculate_lamina_properties
from src.strengths import find_lamina_strengths


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v4.csv"
API_URL = "http://127.0.0.1:8000/analyze"

FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

TARGET_TO_CRITERION = {
    "max_stress_fi": "Maximum Stress",
    "max_strain_fi": "Maximum Strain",
    "tsai_hill_fi": "Tsai-Hill",
    "tsai_wu_fi": "Tsai-Wu",
    "hoffman_fi": "Hoffman",
    "hashin_fi": "Hashin",
    "puck_fi": "Puck",
}

# Strict tolerance because dataset generation and API use the same solver.
ABS_TOLERANCE = 1e-8
MAX_SAMPLES = 15


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + list(TARGET_TO_CRITERION) + ["sampling_region"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

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

    # Select representative cases from different sampling regions.
    selected_indices = []

    zero_indices = df.index[
        (df[FEATURES] == 0).all(axis=1)
    ].tolist()
    if zero_indices:
        selected_indices.append(zero_indices[0])

    for region in [
        "low_load",
        "low_safe",
        "near_boundary_safe",
        "boundary",
        "near_boundary_failed",
        "high_failed",
    ]:
        candidates = df.index[df["sampling_region"] == region].tolist()
        if candidates:
            # Select a deterministic middle row from each region.
            selected_indices.append(candidates[len(candidates) // 2])

    # Add random deterministic samples to broaden the comparison.
    remaining = df.index[~df.index.isin(selected_indices)].to_numpy()
    rng = np.random.default_rng(2026)
    extra_count = min(
        MAX_SAMPLES - len(selected_indices),
        len(remaining),
    )
    if extra_count > 0:
        selected_indices.extend(
            rng.choice(remaining, size=extra_count, replace=False).tolist()
        )

    selected_indices = list(dict.fromkeys(selected_indices))[:MAX_SAMPLES]

    session = requests.Session()
    comparisons = []

    print(f"Dataset: {DATA_PATH}")
    print(f"API: {API_URL}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Absolute tolerance: {ABS_TOLERANCE:g}\n")

    for index in selected_indices:
        row = df.loc[index]

        request_body = {
            "material": {
                "E1": float(E1),
                "E2": float(E2),
                "G12": float(G12),
                "nu12": float(nu12),
            },
            "strengths": strengths,
            "layup": LAYUP,
            "ply_thickness": PLY_THICKNESS,
            "loads": {
                key: float(row[key])
                for key in FEATURES
            },
        }

        response = session.post(
            API_URL,
            json=request_body,
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                f"API failed for dataset row {index}: "
                f"HTTP {response.status_code}\n{response.text}"
            )

        result = response.json()

        # The API response contains governing_results.
        governing = result.get("governing_results")
        if governing is None:
            raise KeyError(
                "API response has no 'governing_results' key. "
                f"Available keys: {list(result.keys())}"
            )

        row_errors = {}

        for target, criterion in TARGET_TO_CRITERION.items():
            if criterion not in governing:
                raise KeyError(
                    f"Criterion {criterion!r} missing from API response."
                )

            expected = float(row[target])
            actual = float(governing[criterion]["failure_index"])
            error = abs(expected - actual)

            row_errors[target] = error

            comparisons.append({
                "row_index": int(index),
                "sampling_region": str(row["sampling_region"]),
                "target": target,
                "dataset_fi": expected,
                "api_fi": actual,
                "absolute_error": error,
            })

        print(
            f"Row {index:5d} | {str(row['sampling_region']):22s} | "
            f"max absolute error = {max(row_errors.values()):.3e}"
        )

    report = pd.DataFrame(comparisons)
    print("\nMaximum absolute error by criterion:")
    summary = report.groupby("target")["absolute_error"].agg(
        ["max", "mean"]
    )
    print(summary.to_string(float_format=lambda value: f"{value:.3e}"))

    max_error = float(report["absolute_error"].max())
    print(f"\nOverall maximum absolute error: {max_error:.3e}")

    if max_error > ABS_TOLERANCE:
        raise AssertionError(
            "V4 dataset targets do not match the API within tolerance. "
            "Do not train until the discrepancy is understood."
        )

    print("\nPASS: selected V4 targets match the classical API.")


if __name__ == "__main__":
    main()