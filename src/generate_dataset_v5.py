"""
Generate a V5 CFRP failure dataset with additional high-FI and boundary coverage.

Run from the project root:
    python -m src.generate_dataset_v5

Creates:
    data/processed/failure_dataset_v5.csv
    data/processed/failure_dataset_v5_summary.csv

Refuses to overwrite either output. Does not modify V4B data or models.
Computational dataset generation only; this is not experimental validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.laminate_failure_comparison import evaluate_laminate_failure, find_governing_failure
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "failure_dataset_v5.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "failure_dataset_v5_summary.csv"

N_SAMPLES = 25_001
SEED = 20261026

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
TARGETS = [f"{prefix}_fi" for prefix in CRITERIA.values()]

# Preserve meaningful coverage near zero while adding coverage for high-FI values.
ZERO_COUNT = 1
LOW_LOAD_BANDS = [
    ("very_low_load", 1_000, 10.0, 1.0),
    ("low_load_100", 1_000, 100.0, 10.0),
    ("low_load_1000", 1_000, 1_000.0, 10.0),
]

# Remaining rows are allocated by Hashin FI bands, but every generated row
# retains all seven criterion targets. This makes the sampling emphasis explicit.
HASHIN_BANDS = [
    ("low_safe", 0.10, 0.60, 3_000),
    ("moderate_safe", 0.60, 0.90, 2_500),
    ("near_boundary_safe", 0.90, 0.99, 3_000),
    ("boundary", 0.99, 1.01, 3_000),
    ("near_boundary_failed", 1.01, 1.10, 2_500),
    ("moderate_failed", 1.10, 2.25, 3_000),
    ("high_failed", 2.25, 4.50, 5_000),
]
SEARCH_MAX_MULTIPLIER = 1e7
BISECTION_STEPS = 90
MAX_ATTEMPTS_PER_ROW = 200


def setup_material():
    fiber = get_fiber(FIBER_NAME)
    matrix = get_matrix(MATRIX_NAME)
    E1, E2, G12, nu12 = calculate_lamina_properties(
        fiber["E"], matrix["E"], fiber["G"], matrix["G"],
        fiber["nu"], matrix["nu"], VF,
    )
    strength_data = find_lamina_strengths(FIBER_NAME, MATRIX_NAME)
    if strength_data is None:
        raise ValueError(f"No strength dataset for {FIBER_NAME}/{MATRIX_NAME}")
    strengths = {key: float(strength_data[key]) for key in ["Xt", "Xc", "Yt", "Yc", "S"]}
    return (float(E1), float(E2), float(G12), float(nu12)), strengths


def evaluate_factory(material_props, strengths):
    E1, E2, G12, nu12 = material_props

    def evaluate(load):
        result = evaluate_laminate_failure(
            E1, E2, G12, nu12, PLY_ANGLES, PLY_THICKNESS,
            load[:3], load[3:], strengths,
        )
        values = {}
        for criterion, prefix in CRITERIA.items():
            governing = find_governing_failure(result, criterion)
            item = governing["criteria"][criterion]
            fi = float(item["failure_index"])
            # Preserve the classical solver's raw failure index. Some
            # interaction criteria (notably Tsai-Wu) can return negative values
            # for safe load combinations in this implementation. Reject only
            # non-finite values; do not clamp training targets.
            if not np.isfinite(fi):
                raise ValueError(f"Non-finite FI for {criterion}: {fi}")
            values[f"{prefix}_fi"] = fi
            values[f"{prefix}_failed"] = int(fi >= 1.0)
            values[f"{prefix}_ply"] = int(governing["ply"])
            values[f"{prefix}_angle"] = float(governing["angle"])
            values[f"{prefix}_surface"] = str(governing["surface"])
            if criterion == "Hashin":
                values["hashin_mode"] = str(item.get("failure_mode", "unknown"))
            if criterion == "Puck":
                values["puck_mode"] = str(item.get("failure_mode", "unknown"))
                if "fracture_angle" in item:
                    values["puck_fracture_angle"] = float(item["fracture_angle"])
        return values

    return evaluate


def make_row(load: np.ndarray, region: str, evaluate, sample_id: int) -> dict:
    row = dict(zip(FEATURES, map(float, load)))
    row["sampling_region"] = region
    row["sample_id"] = sample_id
    row.update(evaluate(load))
    return row


def sample_direction(rng: np.random.Generator) -> np.ndarray:
    # Random directions with component-wise scales comparable to V4B, but with
    # independently varied force and moment scales to broaden load combinations.
    direction = rng.normal(size=6)
    direction[:3] *= np.array([1.0, 0.7, 0.9])
    direction[3:] *= np.array([1.0, 0.7, 0.9])
    direction /= max(np.linalg.norm(direction), 1e-12)
    base_scales = np.array([100_000.0, 50_000.0, 50_000.0, 100.0, 100.0, 100.0])
    # Randomly alter force/moment balance while retaining dimensional scales.
    force_scale = 10 ** rng.uniform(-0.5, 0.5)
    moment_scale = 10 ** rng.uniform(-0.5, 0.5)
    direction[:3] *= force_scale
    direction[3:] *= moment_scale
    return direction * base_scales


def find_hashin_threshold(direction: np.ndarray, evaluate) -> tuple[float, float] | None:
    """Bracket Hashin FI=1 with coarse geometric probes, then refine briefly."""
    multipliers = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=float)
    values = [evaluate(multiplier * direction)["hashin_fi"] for multiplier in multipliers]
    bracket = None
    for i in range(len(multipliers) - 1):
        if values[i] < 1.0 <= values[i + 1]:
            bracket = (multipliers[i], multipliers[i + 1])
            break
    if bracket is None:
        # If the initial scale was too small/large, resample the direction.
        return None
    lo, hi = bracket
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if evaluate(mid * direction)["hashin_fi"] < 1.0:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2.0
    return threshold, float(evaluate(threshold * direction)["hashin_fi"])


def main() -> None:
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite existing V5 output(s):\n"
            f"dataset: {OUTPUT_PATH}\nsummary: {SUMMARY_PATH}"
        )

    material_props, strengths = setup_material()
    evaluate = evaluate_factory(material_props, strengths)
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    sample_id = 0

    # Exact physical zero-load anchor.
    sample_id += 1
    rows.append(make_row(np.zeros(6, dtype=float), "zero_load", evaluate, sample_id))

    # Preserve a smaller, explicit low-load subset.
    for region, count, force_limit, moment_limit in LOW_LOAD_BANDS:
        for low_idx in range(count):
            load = np.array([
                rng.uniform(-force_limit, force_limit),
                rng.uniform(-force_limit, force_limit),
                rng.uniform(-force_limit, force_limit),
                rng.uniform(-moment_limit, moment_limit),
                rng.uniform(-moment_limit, moment_limit),
                rng.uniform(-moment_limit, moment_limit),
            ], dtype=float)
            sample_id += 1
            rows.append(make_row(load, region, evaluate, sample_id))
            if (low_idx + 1) % 250 == 0:
                print(f"{region}: generated {low_idx + 1}/{count}", flush=True)

    # Generate target-specific Hashin FI bands by scaling random load directions.
    # All criteria are calculated for every case; other criteria may land in
    # different FI bands, which is why validation reports must remain per-target.
    boundary_records = []
    for region, target_low, target_high, count in HASHIN_BANDS:
        made = 0
        attempts = 0
        while made < count and attempts < count * MAX_ATTEMPTS_PER_ROW:
            attempts += 1
            if attempts % 100 == 0:
                print(f"{region}: accepted {made}/{count}, attempts {attempts}", flush=True)
            direction = sample_direction(rng)
            threshold = find_hashin_threshold(direction, evaluate)
            if threshold is None:
                continue
            threshold_multiplier, threshold_fi = threshold
            # Choose a target FI in the requested band, then solve for its load
            # multiplier with bisection. This avoids assuming FI scales linearly
            # or quadratically with proportional load.
            target_fi = float(rng.uniform(target_low, target_high))
            if target_fi < 1.0:
                lo, hi = 0.0, threshold_multiplier
            else:
                lo, hi = threshold_multiplier, max(2.0 * threshold_multiplier, threshold_multiplier + 1e-12)
                fi_hi = evaluate(hi * direction)["hashin_fi"]
                while fi_hi < target_fi and hi < SEARCH_MAX_MULTIPLIER:
                    lo = hi
                    hi *= 2.0
                    fi_hi = evaluate(hi * direction)["hashin_fi"]
                if fi_hi < target_fi:
                    continue

            for _ in range(BISECTION_STEPS):
                mid = (lo + hi) / 2.0
                fi_mid = evaluate(mid * direction)["hashin_fi"]
                if fi_mid < target_fi:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-7 * max(hi, 1e-12):
                    break

            multiplier = (lo + hi) / 2.0
            load = multiplier * direction
            achieved_fi = evaluate(load)["hashin_fi"]
            if not (target_low <= achieved_fi <= target_high):
                continue
            target_fi = achieved_fi
            sample_id += 1
            row = make_row(load, region, evaluate, sample_id)
            rows.append(row)
            boundary_records.append({
                "sample_id": sample_id, "sampling_region": region,
                "target_hashin_fi": target_fi,
                "threshold_hashin_fi": threshold_fi,
                "threshold_multiplier": threshold_multiplier,
                "sample_multiplier": multiplier,
            })
            made += 1
            if made % 250 == 0:
                print(f"{region}: accepted {made}/{count}", flush=True)
        if made != count:
            raise RuntimeError(
                f"Could generate only {made}/{count} rows for {region} after {attempts} attempts. "
                "No output has been written; inspect solver behavior before retrying."
            )

    df = pd.DataFrame(rows)
    if len(df) != 25_001:
        raise ValueError(f"Expected 25,001 rows including zero anchor, got {len(df)}")
    numeric = df[FEATURES + TARGETS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("V5 data contains NaN or infinite values.")

    zero_rows = df[(df[FEATURES] == 0.0).all(axis=1)]
    if len(zero_rows) != 1:
        raise ValueError(f"Expected exactly one zero-load anchor, found {len(zero_rows)}")
    if not np.allclose(zero_rows[TARGETS].to_numpy(float), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("Zero-load anchor targets are not all zero.")

    for prefix in CRITERIA.values():
        expected = (df[f"{prefix}_fi"] >= 1.0).astype(int)
        if not np.array_equal(expected.to_numpy(), df[f"{prefix}_failed"].to_numpy()):
            raise ValueError(f"Failure label mismatch for {prefix}")

    # Save only after all validations succeed. Both paths were checked at startup.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("V5 output appeared during generation; refusing to overwrite.")
    df.to_csv(OUTPUT_PATH, index=False)

    summary_rows = []
    for target in TARGETS:
        summary_rows.append({
            "target": target,
            "min": float(df[target].min()),
            "median": float(df[target].median()),
            "mean": float(df[target].mean()),
            "p90": float(df[target].quantile(0.90)),
            "p95": float(df[target].quantile(0.95)),
            "p99": float(df[target].quantile(0.99)),
            "max": float(df[target].max()),
            "count_ge_1": int((df[target] >= 1.0).sum()),
            "count_ge_2_25": int((df[target] >= 2.25).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nV5 DATASET CREATED", flush=True)
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Rows: {len(df)}")
    print("\nSampling regions:")
    print(df["sampling_region"].value_counts().to_string())
    print("\nTarget summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nImportant: inspect target distributions and per-target coverage before training.")
    print("This dataset is computationally generated, not experimentally validated.")


if __name__ == "__main__":
    main()
