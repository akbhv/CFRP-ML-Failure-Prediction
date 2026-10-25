"""Validate V4B surrogate predictions against the classical API.

Run from the project root while the API is running:
    python -m src.validate_v4b_against_classical

This script is read-only. It loads V4B artifacts, calls /analyze for reference
values, and writes a report to a new validation directory. It does not change
api/main.py or the currently active V2 model artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v4b.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "ml_multitarget_v4b"
MODEL_DIR = ROOT / "results" / "multitarget_dnn_v4b"
REPORT_DIR = ROOT / "results" / "validation_v4b_vs_classical"
API_URL = "http://127.0.0.1:8000/analyze"

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = [
    "max_stress_fi",
    "max_strain_fi",
    "tsai_hill_fi",
    "tsai_wu_fi",
    "hoffman_fi",
    "hashin_fi",
    "puck_fi",
]
DISPLAY_NAMES = {
    "max_stress_fi": "Maximum Stress",
    "max_strain_fi": "Maximum Strain",
    "tsai_hill_fi": "Tsai-Hill",
    "tsai_wu_fi": "Tsai-Wu",
    "hoffman_fi": "Hoffman",
    "hashin_fi": "Hashin",
    "puck_fi": "Puck",
}

# Match the material, layup, thickness, and strengths used to generate V4B.
FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3

# Selected cases include strict low-loads, boundary points, and random held-out
# dataset rows. The script also constructs deterministic small-load checks.
MAX_DATASET_CASES = 60
SEED = 20261025
FAILURE_THRESHOLD = 1.0


def load_material_payload() -> tuple[dict, dict]:
    from src.materials import get_fiber, get_matrix
    from src.micromechanics import calculate_lamina_properties
    from src.strengths import find_lamina_strengths

    fiber = get_fiber(FIBER_NAME)
    matrix = get_matrix(MATRIX_NAME)
    E1, E2, G12, nu12 = calculate_lamina_properties(
        fiber["E"], matrix["E"], fiber["G"], matrix["G"],
        fiber["nu"], matrix["nu"], VF
    )
    strengths_data = find_lamina_strengths(FIBER_NAME, MATRIX_NAME)
    if strengths_data is None:
        raise ValueError(f"No strength data for {FIBER_NAME}/{MATRIX_NAME}")

    material = {
        "E1": float(E1),
        "E2": float(E2),
        "G12": float(G12),
        "nu12": float(nu12),
    }
    strengths = {
        key: float(strengths_data[key])
        for key in ["Xt", "Xc", "Yt", "Yc", "S"]
    }
    return material, strengths


def choose_dataset_rows(df: pd.DataFrame) -> list[int]:
    chosen: list[int] = []
    # Always include the zero row and one deterministic row from each region.
    zero_mask = (df[FEATURES] == 0.0).all(axis=1)
    zero_rows = df.index[zero_mask].tolist()
    if zero_rows:
        chosen.append(int(zero_rows[0]))

    for region in [
        "very_low_load",
        "low_load_100",
        "low_load_1000",
        "low_safe",
        "near_boundary_safe",
        "boundary",
        "near_boundary_failed",
        "high_failed",
    ]:
        region_rows = df.index[df["sampling_region"] == region].to_numpy()
        if len(region_rows):
            chosen.append(int(region_rows[len(region_rows) // 2]))

    # Add a deterministic random sample of remaining rows.
    remaining = df.index[~df.index.isin(chosen)].to_numpy()
    rng = np.random.default_rng(SEED)
    extra_count = min(MAX_DATASET_CASES - len(chosen), len(remaining))
    if extra_count > 0:
        chosen.extend(
            rng.choice(remaining, size=extra_count, replace=False).astype(int).tolist()
        )
    return list(dict.fromkeys(chosen))[:MAX_DATASET_CASES]


def create_cases(df: pd.DataFrame) -> list[dict]:
    cases: list[dict] = []
    for idx in choose_dataset_rows(df):
        row = df.loc[idx]
        cases.append({
            "case_id": f"dataset_row_{idx}",
            "source": "V4B dataset row",
            "sampling_region": str(row["sampling_region"]),
            "loads": {feature: float(row[feature]) for feature in FEATURES},
            "dataset_row_index": int(idx),
        })

    # Add fresh, deterministic very-small load cases not copied from the dataset.
    # These use fixed fractions and sign patterns to probe near-zero behaviour.
    small_cases = [
        ("tiny_positive_Nx", [1.0, 0, 0, 0, 0, 0]),
        ("tiny_negative_Nx", [-1.0, 0, 0, 0, 0, 0]),
        ("tiny_mixed_forces", [2.0, -3.0, 1.0, 0, 0, 0]),
        ("tiny_moments", [0, 0, 0, 0.1, -0.2, 0.05]),
        ("small_mixed_loads", [5.0, -4.0, 3.0, 0.5, -0.25, 0.1]),
        ("moderate_low_loads", [50.0, -25.0, 10.0, 2.0, -1.0, 0.5]),
    ]
    for case_id, values in small_cases:
        cases.append({
            "case_id": case_id,
            "source": "constructed low-load check",
            "sampling_region": "independent_low_load_check",
            "loads": dict(zip(FEATURES, map(float, values))),
            "dataset_row_index": None,
        })

    # Also add load vectors scaled from selected dataset cases to probe response
    # as load magnitude increases in the same direction.
    seed_row = df.loc[choose_dataset_rows(df)[-1], FEATURES].to_numpy(dtype=float)
    for factor in [0.25, 0.5, 1.0, 1.5]:
        values = seed_row * factor
        cases.append({
            "case_id": f"scaled_direction_x{factor:g}",
            "source": "scaled load-direction check",
            "sampling_region": "scaled_direction",
            "loads": dict(zip(FEATURES, map(float, values))),
            "dataset_row_index": None,
        })
    return cases


def main() -> None:
    for path in [DATA_PATH, SPLIT_DIR / "feature_scaler.joblib", MODEL_DIR]:
        if not path.exists():
            raise FileNotFoundError(f"Required V4B artifact not found: {path}")
    if REPORT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing validation report directory: {REPORT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + TARGETS + ["sampling_region"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"V4B dataset missing columns: {missing}")

    scaler = joblib.load(SPLIT_DIR / "feature_scaler.joblib")
    models = {}
    for target in TARGETS:
        model_path = MODEL_DIR / target / f"{target}_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        models[target] = keras.models.load_model(model_path)

    material, strengths = load_material_payload()
    cases = create_cases(df)
    session = requests.Session()
    records = []

    print(f"V4B dataset: {DATA_PATH}")
    print(f"Classical reference API: {API_URL}")
    print(f"Cases to compare: {len(cases)}")
    print("Predictions are clamped to non-negative values for comparison, matching the API's inference rule.\n")

    for case in cases:
        loads = case["loads"]
        raw_features = np.array([[loads[key] for key in FEATURES]], dtype=np.float32)
        scaled = scaler.transform(raw_features).astype(np.float32)
        request_body = {
            "material": material,
            "strengths": strengths,
            "layup": LAYUP,
            "ply_thickness": PLY_THICKNESS,
            "loads": loads,
        }
        response = session.post(API_URL, json=request_body, timeout=60)
        if not response.ok:
            raise RuntimeError(
                f"Classical API failed for {case['case_id']}: "
                f"HTTP {response.status_code}\n{response.text}"
            )
        classical = response.json().get("governing_results")
        if classical is None:
            raise KeyError(
                f"No governing_results in /analyze response for {case['case_id']}"
            )

        for target in TARGETS:
            predicted = float(models[target].predict(scaled, verbose=0).reshape(-1)[0])
            predicted = max(0.0, predicted)
            criterion = DISPLAY_NAMES[target]
            reference = float(classical[criterion]["failure_index"])
            record = {
                "case_id": case["case_id"],
                "source": case["source"],
                "sampling_region": case["sampling_region"],
                "dataset_row_index": case["dataset_row_index"],
                **loads,
                "target": target,
                "criterion": criterion,
                "classical_fi": reference,
                "predicted_fi": predicted,
                "absolute_error": abs(predicted - reference),
                "classical_failed": int(reference >= FAILURE_THRESHOLD),
                "predicted_failed": int(predicted >= FAILURE_THRESHOLD),
            }
            records.append(record)

        # Compact per-case diagnostic using the worst criterion error.
        case_records = records[-len(TARGETS):]
        worst = max(case_records, key=lambda item: item["absolute_error"])
        print(
            f"{case['case_id']:28s} | {case['sampling_region']:28s} | "
            f"worst abs error={worst['absolute_error']:.6g}"
        )

    results = pd.DataFrame(records)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    results.to_csv(REPORT_DIR / "case_predictions.csv", index=False)

    metric_rows = []
    for target, group in results.groupby("target", sort=False):
        actual = group["classical_fi"].to_numpy(dtype=float)
        predicted = group["predicted_fi"].to_numpy(dtype=float)
        r2 = (
            float(r2_score(actual, predicted))
            if len(actual) >= 2 and not np.allclose(actual, actual[0])
            else np.nan
        )
        actual_failed = group["classical_failed"].to_numpy(dtype=int)
        predicted_failed = group["predicted_failed"].to_numpy(dtype=int)
        metric_rows.append({
            "target": target,
            "criterion": DISPLAY_NAMES[target],
            "cases": int(len(group)),
            "MAE_vs_classical": float(mean_absolute_error(actual, predicted)),
            "RMSE_vs_classical": float(np.sqrt(mean_squared_error(actual, predicted))),
            "max_absolute_error": float(np.max(np.abs(predicted - actual))),
            "R2_vs_classical": r2,
            "failure_classification_accuracy": float(np.mean(actual_failed == predicted_failed)),
            "false_safe_count": int(np.sum((actual_failed == 1) & (predicted_failed == 0))),
            "false_failure_count": int(np.sum((actual_failed == 0) & (predicted_failed == 1))),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT_DIR / "summary_metrics.csv", index=False)

    # Focused summaries for independently constructed low-load checks and boundary rows.
    subgroup_rows = []
    for (target, subgroup), group in results.groupby(
        ["target", "sampling_region"], sort=False
    ):
        actual = group["classical_fi"].to_numpy(dtype=float)
        predicted = group["predicted_fi"].to_numpy(dtype=float)
        subgroup_rows.append({
            "target": target,
            "criterion": DISPLAY_NAMES[target],
            "subgroup": subgroup,
            "cases": len(group),
            "MAE_vs_classical": float(mean_absolute_error(actual, predicted)),
            "max_absolute_error": float(np.max(np.abs(predicted - actual))),
            "classical_max_fi": float(np.max(actual)),
            "predicted_min_fi": float(np.min(predicted)),
            "predicted_max_fi": float(np.max(predicted)),
        })
    pd.DataFrame(subgroup_rows).to_csv(
        REPORT_DIR / "subgroup_metrics.csv", index=False
    )

    report_meta = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "reference": "running /analyze classical solver",
        "case_count": len(cases),
        "targets": TARGETS,
        "failure_threshold": FAILURE_THRESHOLD,
        "prediction_clamp": "max(0, raw model prediction), for comparison with current API rule",
        "interpretation": (
            "Computational comparison only; this is not experimental validation. "
            "Selected dataset rows and constructed low-load checks are used; "
            "the report does not replace a larger external test suite."
        ),
    }
    (REPORT_DIR / "validation_config.json").write_text(
        json.dumps(report_meta, indent=2), encoding="utf-8"
    )

    print("\n=== V4B vs classical solver summary ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nReports saved to: {REPORT_DIR}")
    print("Review false-safe counts and low-load subgroup results before switching the API to V4B.")


if __name__ == "__main__":
    main()