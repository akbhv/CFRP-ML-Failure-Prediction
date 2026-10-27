"""
Benchmark CLT versus the seven-model V5 surrogate on identical load cases.

Run from the project root:
    python benchmark_clt_vs_ml.py

Outputs:
    results/runtime_benchmark_v5/runtime_benchmark_raw.csv
    results/runtime_benchmark_v5/runtime_benchmark_summary.csv
    results/runtime_benchmark_v5/runtime_comparison.png

Important:
- Uses the project's V5 test split if available; otherwise samples from the V5 dataset.
- CLT uses the fixed configuration documented for V5:
  T300 / Standard Epoxy, Vf=0.60, [0,45,-45,90,90,-45,45,0], 0.125 mm per ply.
- This is a local computation benchmark. It excludes HTTP, network, browser, and rendering time.
- The ML model only receives six load resultants; it does not receive material, layup,
  thickness, or strength inputs. The comparison is only meaningful for the fixed V5 setup.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api import main as api_main
from src.laminate_failure_comparison import evaluate_laminate_failure
from src.materials import get_fiber, get_matrix
from src.micromechanics import calculate_lamina_properties
from src.strengths import find_lamina_strengths


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v5.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "ml_multitarget_v5"
OUT_DIR = ROOT / "results" / "runtime_benchmark_v5"

# Keep the run manageable on a student laptop. Increase for a more stable estimate.
N_CASES = 250
WARMUP_CASES = 10
REPEATS = 5
RANDOM_SEED = 20261028

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = list(api_main.ML_TARGETS.keys())

# Fixed V5 configuration; ply thickness is in metres.
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS_M = 0.125e-3
FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VOLUME_FRACTION = 0.60


def prepare_cases() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run this script from the project root and ensure V5 data is present."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"V5 dataset missing required features: {missing}")

    # Prefer the held-out V5 test split to avoid selecting only easy examples.
    test_indices_path = SPLIT_DIR / "test_indices.npy"
    if test_indices_path.exists():
        indices = np.load(test_indices_path).astype(int)
        candidate = df.iloc[indices].copy()
    else:
        candidate = df.copy()

    candidate = candidate.loc[np.isfinite(candidate[FEATURES]).all(axis=1)]
    if len(candidate) < N_CASES:
        n = min(N_CASES, len(candidate))
        return candidate.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    return candidate.sample(n=N_CASES, random_state=RANDOM_SEED).reset_index(drop=True)


def get_fixed_clt_inputs():
    fiber = get_fiber(FIBER_NAME)
    matrix = get_matrix(MATRIX_NAME)
    E1, E2, G12, nu12 = calculate_lamina_properties(
        Ef=fiber["E"],
        Em=matrix["E"],
        Gf=fiber["G"],
        Gm=matrix["G"],
        nu_f=fiber["nu"],
        nu_m=matrix["nu"],
        Vf=VOLUME_FRACTION,
    )

    strength_record = find_lamina_strengths(FIBER_NAME, MATRIX_NAME)
    if strength_record is None:
        raise ValueError(
            f"No strength record found for {FIBER_NAME} / {MATRIX_NAME}. "
            "Check src/strengths.py and the fixed V5 configuration."
        )
    strengths = {
        key: float(strength_record[key])
        for key in ["Xt", "Xc", "Yt", "Yc", "S"]
    }
    material = {
        "E1": float(E1),
        "E2": float(E2),
        "G12": float(G12),
        "nu12": float(nu12),
    }
    return material, strengths


def run_clt(load_row, material, strengths):
    N = np.array([load_row["Nx"], load_row["Ny"], load_row["Nxy"]], dtype=float)
    M = np.array([load_row["Mx"], load_row["My"], load_row["Mxy"]], dtype=float)
    result = evaluate_laminate_failure(
        E1=material["E1"],
        E2=material["E2"],
        G12=material["G12"],
        nu12=material["nu12"],
        ply_angles=LAYUP,
        ply_thickness=PLY_THICKNESS_M,
        N=N,
        M=M,
        strengths=strengths,
    )
    # Ensure all seven criteria were evaluated, and consume the governing values.
    governing = {}
    for criterion in [
        "Maximum Stress", "Maximum Strain", "Tsai-Hill", "Tsai-Wu",
        "Hoffman", "Hashin", "Puck"
    ]:
        governing[criterion] = max(
            ply["criteria"][criterion]["failure_index"]
            for ply in result["ply_results"]
        )
    return governing


def run_ml(load_row):
    features = np.asarray(
        [[load_row[c] for c in FEATURES]], dtype=np.float32
    )
    scaled = api_main.ml_scaler.transform(features)
    outputs = {}
    for target, display_name in api_main.ML_TARGETS.items():
        pred = float(
            api_main.ml_models[target](scaled, training=False)
            .numpy().reshape(-1)[0]
        )
        outputs[display_name] = max(0.0, pred)
    return outputs


def time_call(fn, row):
    start = time.perf_counter()
    fn(row)
    return (time.perf_counter() - start) * 1000.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = prepare_cases()
    material, strengths = get_fixed_clt_inputs()

    print(f"Cases: {len(cases)} | warm-up: {WARMUP_CASES} | repeats: {REPEATS}")
    print("Warming up CLT and ML paths...")
    for i in range(min(WARMUP_CASES, len(cases))):
        row = cases.iloc[i]
        run_clt(row, material, strengths)
        run_ml(row)

    records = []
    for case_number, (_, row) in enumerate(cases.iterrows(), start=1):
        clt_times = []
        ml_times = []
        for _ in range(REPEATS):
            clt_times.append(time_call(
                lambda r: run_clt(r, material, strengths), row
            ))
            ml_times.append(time_call(run_ml, row))

        for repeat, elapsed in enumerate(clt_times, start=1):
            records.append({
                "case_index": case_number,
                "repeat": repeat,
                "path": "CLT_all_7_criteria",
                "time_ms": elapsed,
            })
        for repeat, elapsed in enumerate(ml_times, start=1):
            records.append({
                "case_index": case_number,
                "repeat": repeat,
                "path": "V5_ML_7_models",
                "time_ms": elapsed,
            })

        if case_number % 25 == 0 or case_number == len(cases):
            print(f"Timed {case_number}/{len(cases)} cases")

    raw = pd.DataFrame(records)
    raw_path = OUT_DIR / "runtime_benchmark_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby("path")["time_ms"]
        .agg(
            count="count",
            median_ms="median",
            mean_ms="mean",
            std_ms="std",
            p95_ms=lambda s: s.quantile(0.95),
            min_ms="min",
            max_ms="max",
        )
        .reset_index()
    )
    summary["cases"] = len(cases)
    summary["repeats_per_case"] = REPEATS
    summary_path = OUT_DIR / "runtime_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    medians = summary.set_index("path")["median_ms"]
    clt_median = float(medians["CLT_all_7_criteria"])
    ml_median = float(medians["V5_ML_7_models"])
    ratio = clt_median / ml_median if ml_median > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["CLT: all 7 criteria", "V5 ML: 7 models"]
    values = [clt_median, ml_median]
    bars = ax.bar(labels, values)
    ax.set_ylabel("Median computation time per case (ms)")
    ax.set_title("Local runtime comparison: fixed V5 laminate configuration")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f} ms",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom"
        )
    fig.tight_layout()
    chart_path = OUT_DIR / "runtime_comparison.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    print("\n=== Runtime summary (milliseconds per case) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMedian CLT / median ML time ratio: {ratio:.3f}x")
    print("Interpretation: ratio > 1 means ML inference was faster in this local test.")
    print("This is not a web/API latency comparison and does not prove experimental validity.")
    print(f"\nRaw measurements: {raw_path}")
    print(f"Summary:          {summary_path}")
    print(f"Chart:            {chart_path}")


if __name__ == "__main__":
    main()
