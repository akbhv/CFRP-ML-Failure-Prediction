"""Generate a versioned CFRP failure dataset with explicit low-load coverage.

Run from the project root:
    python -m src.generate_dataset_v4

This script refuses to overwrite an existing V4 CSV. It uses the same
classical laminate solver and material/strength database as the application.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.laminate_failure_comparison import evaluate_laminate_failure, find_governing_failure
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "failure_dataset_v4.csv"
N_SAMPLES = 20000
SEED = 2026
FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
Vf = 0.60
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


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT_PATH}")

    fiber = get_fiber(FIBER_NAME)
    matrix = get_matrix(MATRIX_NAME)
    E1, E2, G12, nu12 = calculate_lamina_properties(
        fiber["E"], matrix["E"], fiber["G"], matrix["G"],
        fiber["nu"], matrix["nu"], Vf
    )
    strength_data = find_lamina_strengths(FIBER_NAME, MATRIX_NAME)
    if strength_data is None:
        raise ValueError(f"No strength dataset for {FIBER_NAME}/{MATRIX_NAME}")
    strengths = {k: float(strength_data[k]) for k in ["Xt", "Xc", "Yt", "Yc", "S"]}

    rng = np.random.default_rng(SEED)
    rows = []

    # Exactly one zero-load anchor; repeated copies would leak across random splits.
    # Low-load cases deliberately span the small-load region absent from V3.
    n_zero = 1
    n_low = int(N_SAMPLES * 0.20)
    n_boundary = N_SAMPLES - n_zero - n_low

    load_cases = [(np.zeros(6, dtype=float), "zero_load")]
    for _ in range(n_low):
        # Independent signed components; forces in N/m and moments in N.
        load = np.array([
            rng.uniform(-1000, 1000),
            rng.uniform(-1000, 1000),
            rng.uniform(-1000, 1000),
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
        ], dtype=float)
        load_cases.append((load, "low_load"))

    load_scales = np.array([100_000.0, 50_000.0, 50_000.0, 100.0, 100.0, 100.0])

    def evaluate(load):
        return evaluate_laminate_failure(
            E1, E2, G12, nu12, PLY_ANGLES, PLY_THICKNESS,
            load[:3], load[3:], strengths
        )

    def hashin_fi(result):
        governing = find_governing_failure(result, "Hashin")
        return float(governing["criteria"]["Hashin"]["failure_index"])

    # Boundary-oriented cases retained for useful failure-onset coverage.
    for _ in range(n_boundary):
        direction = rng.normal(size=6)
        direction /= np.linalg.norm(direction)
        direction *= load_scales
        lo, hi = 0.0, 1.0
        result_hi = evaluate(hi * direction)
        while hashin_fi(result_hi) < 1.0 and hi < 1e6:
            lo = hi
            hi = min(hi * 2.0, 1e6)
            result_hi = evaluate(hi * direction)
        if hashin_fi(result_hi) < 1.0:
            # Unusual direction that did not reach Hashin failure; retain a broad safe sample.
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
        multiplier = rng.choice([rng.uniform(0.10, 0.60), rng.uniform(0.60, 0.95),
                                  rng.uniform(0.95, 1.05), rng.uniform(1.05, 1.20),
                                  rng.uniform(1.20, 1.50)])
        # Label by multiplier relative to the Hashin boundary.
        region = ("low_safe" if multiplier < 0.60 else
                  "near_boundary_safe" if multiplier < 0.95 else
                  "boundary" if multiplier <= 1.05 else
                  "near_boundary_failed" if multiplier <= 1.20 else "high_failed")
        load_cases.append((hi * multiplier * direction, region))

    for idx, (load, region) in enumerate(load_cases, start=1):
        result = evaluate(load)
        row = dict(zip(FEATURES, map(float, load)))
        row["sampling_region"] = region
        for criterion, prefix in CRITERIA.items():
            governing = find_governing_failure(result, criterion)
            data = governing["criteria"][criterion]
            fi = float(data["failure_index"])
            if not np.isfinite(fi):
                raise ValueError(f"Non-finite {criterion} FI at sample {idx}: {fi}")
            row[f"{prefix}_fi"] = fi
            row[f"{prefix}_failed"] = int(bool(data["failed"]))
            row[f"{prefix}_ply"] = int(governing["ply"])
            row[f"{prefix}_angle"] = float(governing["angle"])
            row[f"{prefix}_surface"] = str(governing["surface"])
            if criterion == "Hashin":
                row["hashin_mode"] = str(data.get("failure_mode", "unknown"))
            if criterion == "Puck":
                row["puck_mode"] = str(data.get("failure_mode", "unknown"))
                if "fracture_angle" in data:
                    row["puck_fracture_angle"] = float(data["fracture_angle"])
        row["failed"] = row["hashin_failed"]
        rows.append(row)
        if idx % 1000 == 0:
            print(f"Evaluated {idx}/{len(load_cases)} samples")

    df = pd.DataFrame(rows)
    if len(df) != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} rows; got {len(df)}")
    numeric = df[FEATURES + [f"{p}_fi" for p in CRITERIA.values()]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Dataset contains NaN or infinite inputs/targets")

    # Physical anchor: zero load must yield zero FI for all seven criteria.
    zero = df.loc[df["sampling_region"] == "zero_load"]
    if len(zero) != 1:
        raise ValueError(f"Expected one zero-load anchor, found {len(zero)}")
    zero_fis = zero[[f"{p}_fi" for p in CRITERIA.values()]].iloc[0].to_numpy(dtype=float)
    if not np.allclose(zero_fis, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"Zero-load failure indices are not all zero: {zero_fis}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Re-check immediately before writing to avoid accidental overwrite.
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nV4 DATASET CREATED")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("Sampling regions:\n" + df["sampling_region"].value_counts().to_string())
    print("\nTarget summary (min/median/mean/max):")
    target_cols = [f"{p}_fi" for p in CRITERIA.values()]
    print(df[target_cols].agg(["min", "median", "mean", "max"]).to_string())
    print("\nLow-load coverage:")
    for f_lim, m_lim in [(10, 1), (100, 10), (1000, 10)]:
        mask = (df[["Nx", "Ny", "Nxy"]].abs().max(axis=1) <= f_lim) & (df[["Mx", "My", "Mxy"]].abs().max(axis=1) <= m_lim)
        print(f"forces <= {f_lim} N/m and moments <= {m_lim} N: {int(mask.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
