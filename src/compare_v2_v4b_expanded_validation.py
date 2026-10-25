"""Expanded comparison of V2 and V4B surrogate models against the classical solver.

Save as src/compare_v2_v4b_expanded_validation.py and run from the project root:
    python -m src.compare_v2_v4b_expanded_validation

This is computational validation against the existing classical solver, not
experimental validation. It does not modify api/main.py or switch active models.
It preserves the previous 70-case report by writing to a separate directory.
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

V2_SCALER_PATH = (
    ROOT / "data" / "processed" / "ml_multitarget_v2_hoffman_corrected"
    / "feature_scaler.joblib"
)
V2_MODEL_DIR = ROOT / "results" / "multitarget_dnn_v2_hoffman_corrected"

V4B_SCALER_PATH = (
    ROOT / "data" / "processed" / "ml_multitarget_v4b"
    / "feature_scaler.joblib"
)
V4B_MODEL_DIR = ROOT / "results" / "multitarget_dnn_v4b"

REPORT_DIR = ROOT / "results" / "validation_v2_v4b_expanded"
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

# Must match the dataset generation material configuration.
FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3

MAX_DATASET_CASES = 280
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
        fiber["nu"], matrix["nu"], VF,
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
    """Select a reproducible, region-stratified sample plus the zero-load anchor."""
    chosen: list[int] = []
    zero_mask = (df[FEATURES] == 0.0).all(axis=1)
    zero_rows = df.index[zero_mask].tolist()
    if zero_rows:
        chosen.append(int(zero_rows[0]))

    regions = [
        "very_low_load",
        "low_load_100",
        "low_load_1000",
        "low_safe",
        "near_boundary_safe",
        "boundary",
        "near_boundary_failed",
        "high_failed",
    ]
    rng = np.random.default_rng(SEED)
    per_region = max(1, (MAX_DATASET_CASES - len(chosen)) // len(regions))
    for region in regions:
        region_rows = df.index[
            (df["sampling_region"] == region) & (~df.index.isin(chosen))
        ].to_numpy()
        take = min(per_region, len(region_rows))
        if take:
            selected = rng.choice(region_rows, size=take, replace=False)
            chosen.extend(int(i) for i in selected)

    remaining = df.index[~df.index.isin(chosen)].to_numpy()
    extra_count = min(max(0, MAX_DATASET_CASES - len(chosen)), len(remaining))
    if extra_count:
        chosen.extend(
            int(i) for i in rng.choice(remaining, size=extra_count, replace=False)
        )
    return list(dict.fromkeys(chosen))[:MAX_DATASET_CASES]


def create_cases(df: pd.DataFrame) -> list[dict]:
    cases: list[dict] = []
    selected = choose_dataset_rows(df)

    for idx in selected:
        row = df.loc[idx]
        cases.append({
            "case_id": f"dataset_row_{idx}",
            "source": "V4B dataset stratified sample",
            "sampling_region": str(row["sampling_region"]),
            "loads": {feature: float(row[feature]) for feature in FEATURES},
            "dataset_row_index": int(idx),
        })

    # Constructed low-load probes not selected from the dataset.
    small_cases = [
        ("tiny_positive_Nx", [1.0, 0, 0, 0, 0, 0]),
        ("tiny_negative_Nx", [-1.0, 0, 0, 0, 0, 0]),
        ("tiny_mixed_forces", [2.0, -3.0, 1.0, 0, 0, 0]),
        ("tiny_moments", [0, 0, 0, 0.1, -0.2, 0.05]),
        ("small_mixed_loads", [5.0, -4.0, 3.0, 0.5, -0.25, 0.1]),
        ("moderate_low_loads", [50.0, -25.0, 10.0, 2.0, -1.0, 0.5]),
        ("low_load_Nx_only_100", [100.0, 0, 0, 0, 0, 0]),
        ("low_load_Ny_only_100", [0, -100.0, 0, 0, 0, 0]),
        ("low_load_Nxy_only_100", [0, 0, 100.0, 0, 0, 0]),
    ]
    for case_id, values in small_cases:
        cases.append({
            "case_id": case_id,
            "source": "constructed low-load check",
            "sampling_region": "independent_low_load_check",
            "loads": dict(zip(FEATURES, map(float, values))),
            "dataset_row_index": None,
        })

    # Probe response along six sampled load directions at several scales.
    # These are stress tests, not independent random validation samples.
    rng = np.random.default_rng(SEED + 1)
    seed_indices = rng.choice(selected, size=min(6, len(selected)), replace=False)
    factors = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    for seed_number, idx in enumerate(seed_indices, start=1):
        seed_row = df.loc[idx, FEATURES].to_numpy(dtype=float)
        for factor in factors:
            values = seed_row * factor
            cases.append({
                "case_id": f"scaled_seed_{seed_number}_x{factor:g}",
                "source": "scaled load-direction check",
                "sampling_region": "scaled_direction",
                "loads": dict(zip(FEATURES, map(float, values))),
                "dataset_row_index": None,
            })
    return cases


def load_model_bundle(version: str, scaler_path: Path, model_dir: Path):
    if not scaler_path.is_file():
        raise FileNotFoundError(f"{version} scaler not found: {scaler_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{version} model directory not found: {model_dir}")

    scaler = joblib.load(scaler_path)
    models = {}
    for target in TARGETS:
        model_path = model_dir / target / f"{target}_model.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"{version} model not found: {model_path}")
        models[target] = keras.models.load_model(model_path)
    return scaler, models


def predict_all(scaler, models, loads: dict[str, float]) -> dict[str, float]:
    raw = np.array([[loads[key] for key in FEATURES]], dtype=np.float32)
    scaled = scaler.transform(raw).astype(np.float32)
    output = {}
    for target in TARGETS:
        raw_prediction = float(models[target].predict(scaled, verbose=0).reshape(-1)[0])
        # Match the current API inference post-processing.
        output[target] = max(0.0, raw_prediction)
    return output


def compute_metrics(group: pd.DataFrame, model_label: str, target: str) -> dict:
    actual = group["classical_fi"].to_numpy(dtype=float)
    predicted = group[f"{model_label}_fi"].to_numpy(dtype=float)
    actual_failed = group["v2_classical_failed"].to_numpy(dtype=int)
    predicted_failed = group[f"{model_label}_failed"].to_numpy(dtype=int)
    r2 = (
        float(r2_score(actual, predicted))
        if len(actual) >= 2 and not np.allclose(actual, actual[0])
        else np.nan
    )
    return {
        "model": model_label.upper(),
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
    }


def main() -> None:
    required_paths = [
        DATA_PATH, V2_SCALER_PATH, V2_MODEL_DIR,
        V4B_SCALER_PATH, V4B_MODEL_DIR,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required validation artifacts:\n - " + "\n - ".join(missing_paths)
        )
    if REPORT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report directory: {REPORT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = [
        column for column in FEATURES + ["sampling_region"] if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"V4B dataset is missing required columns: {missing_columns}")

    print("Loading V2 and V4B artifacts without modifying active API configuration...")
    v2_scaler, v2_models = load_model_bundle("V2", V2_SCALER_PATH, V2_MODEL_DIR)
    v4b_scaler, v4b_models = load_model_bundle("V4B", V4B_SCALER_PATH, V4B_MODEL_DIR)
    material, strengths = load_material_payload()
    cases = create_cases(df)

    session = requests.Session()
    records: list[dict] = []
    print(f"Classical reference API: {API_URL}")
    print(f"Shared test cases: {len(cases)} (stratified dataset sample plus constructed checks)")
    print("Each case uses one classical reference response and both model versions.")
    print("Predictions are clamped to non-negative values to match API inference.\n")

    for case in cases:
        loads = case["loads"]
        request_body = {
            "material": material,
            "strengths": strengths,
            "layup": LAYUP,
            "ply_thickness": PLY_THICKNESS,
            "loads": loads,
        }
        try:
            response = session.post(API_URL, json=request_body, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach classical API at {API_URL}. "
                "Start the API and retry."
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f"Classical API failed for {case['case_id']}: "
                f"HTTP {response.status_code}\n{response.text}"
            )
        payload = response.json()
        classical = payload.get("governing_results")
        if classical is None:
            raise KeyError(f"No governing_results in /analyze response for {case['case_id']}")

        v2_predictions = predict_all(v2_scaler, v2_models, loads)
        v4b_predictions = predict_all(v4b_scaler, v4b_models, loads)

        for target in TARGETS:
            criterion = DISPLAY_NAMES[target]
            reference = float(classical[criterion]["failure_index"])
            v2_fi = v2_predictions[target]
            v4b_fi = v4b_predictions[target]
            records.append({
                "case_id": case["case_id"],
                "source": case["source"],
                "sampling_region": case["sampling_region"],
                "dataset_row_index": case["dataset_row_index"],
                **loads,
                "target": target,
                "criterion": criterion,
                "classical_fi": reference,
                "v2_fi": v2_fi,
                "v4b_fi": v4b_fi,
                "v2_absolute_error": abs(v2_fi - reference),
                "v4b_absolute_error": abs(v4b_fi - reference),
                "v2_classical_failed": int(reference >= FAILURE_THRESHOLD),
                "v2_failed": int(v2_fi >= FAILURE_THRESHOLD),
                "v4b_failed": int(v4b_fi >= FAILURE_THRESHOLD),
                "v2_false_safe": int(reference >= FAILURE_THRESHOLD and v2_fi < FAILURE_THRESHOLD),
                "v2_false_failure": int(reference < FAILURE_THRESHOLD and v2_fi >= FAILURE_THRESHOLD),
                "v4b_false_safe": int(reference >= FAILURE_THRESHOLD and v4b_fi < FAILURE_THRESHOLD),
                "v4b_false_failure": int(reference < FAILURE_THRESHOLD and v4b_fi >= FAILURE_THRESHOLD),
                "v2_v4b_absolute_difference": abs(v2_fi - v4b_fi),
            })

        latest = records[-len(TARGETS):]
        worst = max(latest, key=lambda item: max(item["v2_absolute_error"], item["v4b_absolute_error"]))
        print(
            f"{case['case_id']:28s} | {case['sampling_region']:28s} | "
            f"worst model/reference error={max(worst['v2_absolute_error'], worst['v4b_absolute_error']):.6g}"
        )

    results = pd.DataFrame(records)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    results.to_csv(REPORT_DIR / "case_predictions.csv", index=False)

    metric_rows = []
    for target in TARGETS:
        group = results[results["target"] == target]
        metric_rows.append(compute_metrics(group, "v2", target))
        metric_rows.append(compute_metrics(group, "v4b", target))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT_DIR / "summary_metrics.csv", index=False)

    subgroup_rows = []
    for (target, subgroup), group in results.groupby(["target", "sampling_region"], sort=False):
        for model_label in ["v2", "v4b"]:
            actual = group["classical_fi"].to_numpy(dtype=float)
            predicted = group[f"{model_label}_fi"].to_numpy(dtype=float)
            subgroup_rows.append({
                "model": model_label.upper(),
                "target": target,
                "criterion": DISPLAY_NAMES[target],
                "subgroup": subgroup,
                "cases": int(len(group)),
                "MAE_vs_classical": float(mean_absolute_error(actual, predicted)),
                "max_absolute_error": float(np.max(np.abs(predicted - actual))),
                "false_safe_count": int(group[f"{model_label}_false_safe"].sum()),
                "false_failure_count": int(group[f"{model_label}_false_failure"].sum()),
                "classical_min_fi": float(np.min(actual)),
                "classical_max_fi": float(np.max(actual)),
                "predicted_min_fi": float(np.min(predicted)),
                "predicted_max_fi": float(np.max(predicted)),
            })
    pd.DataFrame(subgroup_rows).to_csv(REPORT_DIR / "subgroup_metrics.csv", index=False)

    disagreement_rows = results[
        (results["v2_false_safe"] == 1)
        | (results["v2_false_failure"] == 1)
        | (results["v4b_false_safe"] == 1)
        | (results["v4b_false_failure"] == 1)
    ].copy()
    disagreement_rows.to_csv(REPORT_DIR / "classification_disagreements.csv", index=False)

    boundary_rows = results[
        (results["classical_fi"] - FAILURE_THRESHOLD).abs() <= 0.10
    ].copy()
    boundary_rows.to_csv(REPORT_DIR / "near_threshold_cases.csv", index=False)

    metadata = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "v2_scaler": str(V2_SCALER_PATH.relative_to(ROOT)),
        "v2_model_dir": str(V2_MODEL_DIR.relative_to(ROOT)),
        "v4b_scaler": str(V4B_SCALER_PATH.relative_to(ROOT)),
        "v4b_model_dir": str(V4B_MODEL_DIR.relative_to(ROOT)),
        "reference": "running /analyze classical solver",
        "case_count": len(cases),
        "targets": TARGETS,
        "failure_threshold": FAILURE_THRESHOLD,
        "postprocessing": "max(0, raw model prediction), matching active API inference rule",
        "active_api_modified": False,
        "sampling_design": (
            "Reproducible region-stratified sample of up to 280 rows from the V4B dataset, "
            "plus constructed low-load probes and scaled versions of six sampled load directions."
        ),
        "interpretation": (
            "Computational comparison only, not experimental validation. Both model versions "
            "were evaluated on identical cases against the running classical solver. Dataset "
            "rows are drawn from the same generated dataset family, so this is stronger "
            "diagnostic coverage, not a fully independent external validation. Scaled cases "
            "are probes, not independent random draws. V2 remains the active API model."
        ),
    }
    (REPORT_DIR / "validation_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n=== V2 vs V4B vs classical solver ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nClassification disagreements:", len(disagreement_rows))
    print("Cases within ±0.10 of FI=1.0:", len(boundary_rows))
    print(f"Reports saved to: {REPORT_DIR}")
    print("Do not switch the active API model based on this computational sample alone.")


if __name__ == "__main__":
    main()