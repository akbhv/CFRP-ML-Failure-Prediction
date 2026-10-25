"""
Targeted V4B validation against the classical solver.

Run from the project root:
    python -m src.validate_v4b_targeted

Uses only held-out V4B test rows as seed directions, then evaluates selected
load combinations and scaled probes against the running classical /analyze API.
Writes to a new report directory and does not change api/main.py or model files.

Important: this is computational validation, not experimental validation.
Scaled probes are stress tests and are not independent random samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "failure_dataset_v4b.csv"
TEST_INDICES_PATH = ROOT / "data" / "processed" / "ml_multitarget_v4b" / "test_indices.npy"
SCALER_PATH = ROOT / "data" / "processed" / "ml_multitarget_v4b" / "feature_scaler.joblib"
MODEL_DIR = ROOT / "results" / "multitarget_dnn_v4b"
REPORT_DIR = ROOT / "results" / "validation_v4b_targeted"
API_URL = "http://127.0.0.1:8000/analyze"

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
TARGETS = [
    "max_stress_fi", "max_strain_fi", "tsai_hill_fi", "tsai_wu_fi",
    "hoffman_fi", "hashin_fi", "puck_fi",
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
FAILURE_THRESHOLD = 1.0
SEED = 20261026
MAX_ROWS_PER_TARGET_BAND = 12
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]

# Must match the material/layup used by the existing V4B dataset generator.
FIBER_NAME = "T300"
MATRIX_NAME = "Standard Epoxy"
VF = 0.60
LAYUP = [0, 45, -45, 90, 90, -45, 45, 0]
PLY_THICKNESS = 0.125e-3


def material_payload() -> tuple[dict, dict]:
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
        "E1": float(E1), "E2": float(E2), "G12": float(G12), "nu12": float(nu12)
    }
    strengths = {key: float(strengths_data[key]) for key in ["Xt", "Xc", "Yt", "Yc", "S"]}
    return material, strengths


def load_models():
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    models = {}
    for target in TARGETS:
        path = MODEL_DIR / target / f"{target}_model.keras"
        if not path.is_file():
            raise FileNotFoundError(f"V4B model not found: {path}")
        models[target] = keras.models.load_model(path)
    return scaler, models


def predict(scaler, models, loads: dict[str, float]) -> dict[str, float]:
    x = np.asarray([[loads[name] for name in FEATURES]], dtype=np.float32)
    xs = scaler.transform(x).astype(np.float32)
    return {
        target: max(0.0, float(models[target].predict(xs, verbose=0).reshape(-1)[0]))
        for target in TARGETS
    }


def select_seed_rows(df: pd.DataFrame, test_indices: np.ndarray) -> list[int]:
    """Select held-out test rows by target FI band, then deduplicate."""
    rng = np.random.default_rng(SEED)
    test_df = df.loc[test_indices]
    # Distinct FI bands target the safe region, threshold, and high-FI region.
    bands = [
        ("low_safe", lambda s: s < 0.5),
        ("moderate_safe", lambda s: (s >= 0.5) & (s < 0.9)),
        ("near_boundary_safe", lambda s: (s >= 0.9) & (s < 1.0)),
        ("near_boundary_failed", lambda s: (s >= 1.0) & (s <= 1.1)),
        ("moderate_failed", lambda s: (s > 1.1) & (s <= 2.25)),
        ("high_fi", lambda s: s > 2.25),
    ]
    chosen: list[int] = []
    for target in TARGETS:
        values = test_df[target]
        for _band_name, mask_fn in bands:
            candidates = test_df.index[mask_fn(values)].to_numpy()
            candidates = np.asarray([i for i in candidates if int(i) not in chosen], dtype=int)
            if len(candidates):
                take = min(MAX_ROWS_PER_TARGET_BAND, len(candidates))
                chosen.extend(int(i) for i in rng.choice(candidates, size=take, replace=False))
    return list(dict.fromkeys(chosen))


def build_cases(df: pd.DataFrame, test_indices: np.ndarray) -> list[dict]:
    cases: list[dict] = []
    selected = select_seed_rows(df, test_indices)
    for idx in selected:
        row = df.loc[idx]
        cases.append({
            "case_id": f"heldout_test_row_{idx}",
            "source": "held-out test row",
            "seed_row_index": int(idx),
            "scale_factor": 1.0,
            "loads": {feature: float(row[feature]) for feature in FEATURES},
        })

    # Scale held-out test directions to probe interpolation and higher FI values.
    # Exclude 1.0 because the unscaled case has already been added.
    rng = np.random.default_rng(SEED + 1)
    seed_count = min(12, len(selected))
    seeds = rng.choice(selected, size=seed_count, replace=False) if seed_count else []
    for seed_number, idx in enumerate(seeds, start=1):
        base = df.loc[int(idx), FEATURES].to_numpy(dtype=float)
        for factor in SCALES:
            if np.isclose(factor, 1.0):
                continue
            values = base * factor
            cases.append({
                "case_id": f"scaled_test_seed_{seed_number}_x{factor:g}",
                "source": "scaled held-out test direction",
                "seed_row_index": int(idx),
                "scale_factor": float(factor),
                "loads": dict(zip(FEATURES, map(float, values))),
            })

    # Explicit low-load probes; these are constructed diagnostics, not test-set rows.
    low_loads = [
        ("tiny_positive_Nx", [1.0, 0, 0, 0, 0, 0]),
        ("tiny_negative_Nx", [-1.0, 0, 0, 0, 0, 0]),
        ("tiny_mixed_forces", [2.0, -3.0, 1.0, 0, 0, 0]),
        ("tiny_moments", [0, 0, 0, 0.1, -0.2, 0.05]),
        ("small_mixed_loads", [5.0, -4.0, 3.0, 0.5, -0.25, 0.1]),
    ]
    for case_id, values in low_loads:
        cases.append({
            "case_id": case_id, "source": "constructed low-load probe",
            "seed_row_index": None, "scale_factor": None,
            "loads": dict(zip(FEATURES, map(float, values))),
        })
    return cases


def main() -> None:
    required = [DATA_PATH, TEST_INDICES_PATH, SCALER_PATH, MODEL_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n - " + "\n - ".join(missing))
    if REPORT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report directory: {REPORT_DIR}"
        )

    df = pd.read_csv(DATA_PATH)
    required_cols = FEATURES + TARGETS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    test_indices = np.load(TEST_INDICES_PATH).astype(int)
    if len(test_indices) != 3000:
        raise ValueError(f"Expected 3,000 held-out test indices, found {len(test_indices)}")
    if np.any(test_indices < 0) or np.any(test_indices >= len(df)):
        raise ValueError("Test indices contain invalid dataset row indices.")

    scaler, models = load_models()
    material, strengths = material_payload()
    cases = build_cases(df, test_indices)
    session = requests.Session()
    records = []

    print(f"V4B targeted validation: {len(cases)} cases")
    print(f"Classical solver endpoint: {API_URL}")
    print("This does not modify api/main.py or any model artifacts.\n")

    for n, case in enumerate(cases, start=1):
        loads = case["loads"]
        payload = {
            "material": material, "strengths": strengths,
            "layup": LAYUP, "ply_thickness": PLY_THICKNESS, "loads": loads,
        }
        try:
            response = session.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach classical API at {API_URL}. Start the backend and retry."
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f"API failed for {case['case_id']}: HTTP {response.status_code}\n{response.text}"
            )
        response_data = response.json()
        classical = response_data.get("governing_results")
        if classical is None:
            raise KeyError(f"No governing_results returned for {case['case_id']}")
        predicted = predict(scaler, models, loads)

        for target in TARGETS:
            reference = float(classical[DISPLAY_NAMES[target]]["failure_index"])
            estimate = predicted[target]
            records.append({
                "case_id": case["case_id"],
                "source": case["source"],
                "seed_row_index": case["seed_row_index"],
                "scale_factor": case["scale_factor"],
                **loads,
                "target": target,
                "criterion": DISPLAY_NAMES[target],
                "classical_fi": reference,
                "v4b_fi": estimate,
                "absolute_error": abs(estimate - reference),
                "signed_error_predicted_minus_classical": estimate - reference,
                "classical_failed": int(reference >= FAILURE_THRESHOLD),
                "v4b_failed": int(estimate >= FAILURE_THRESHOLD),
                "false_safe": int(reference >= FAILURE_THRESHOLD and estimate < FAILURE_THRESHOLD),
                "false_failure": int(reference < FAILURE_THRESHOLD and estimate >= FAILURE_THRESHOLD),
                "classical_fi_over_training_target_max": bool(
                    reference > float(df[target].max())
                ),
                "classical_fi_over_2_25": bool(reference > 2.25),
            })
        if n % 10 == 0 or n == len(cases):
            print(f"Completed {n}/{len(cases)} cases")

    results = pd.DataFrame(records)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    results.to_csv(REPORT_DIR / "case_predictions.csv", index=False)

    summary_rows = []
    for target in TARGETS:
        group = results[results["target"] == target]
        actual = group["classical_fi"].to_numpy(float)
        predicted = group["v4b_fi"].to_numpy(float)
        summary_rows.append({
            "target": target,
            "criterion": DISPLAY_NAMES[target],
            "cases": len(group),
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "max_absolute_error": float(np.max(np.abs(predicted - actual))),
            "false_safe_count": int(group["false_safe"].sum()),
            "false_failure_count": int(group["false_failure"].sum()),
            "count_classical_fi_over_training_target_max": int(
                group["classical_fi_over_training_target_max"].sum()
            ),
            "count_classical_fi_over_2_25": int(group["classical_fi_over_2_25"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "summary_metrics.csv", index=False)

    # Detailed errors, grouped by FI range, and near-threshold records.
    results["classical_fi_band"] = pd.cut(
        results["classical_fi"],
        bins=[-np.inf, 0.5, 0.9, 1.0, 1.1, 2.25, np.inf],
        labels=["<0.5", "0.5-0.9", "0.9-1.0", "1.0-1.1", "1.1-2.25", ">2.25"],
        right=True,
        include_lowest=True,
    ).astype(str)
    band_rows = []
    for (target, band), group in results.groupby(["target", "classical_fi_band"], observed=False):
        actual = group["classical_fi"].to_numpy(float)
        predicted = group["v4b_fi"].to_numpy(float)
        band_rows.append({
            "target": target, "criterion": DISPLAY_NAMES[target],
            "classical_fi_band": band, "cases": len(group),
            "MAE": float(mean_absolute_error(actual, predicted)) if len(group) else np.nan,
            "max_absolute_error": float(np.max(np.abs(predicted - actual))) if len(group) else np.nan,
            "false_safe_count": int(group["false_safe"].sum()),
            "false_failure_count": int(group["false_failure"].sum()),
        })
    pd.DataFrame(band_rows).to_csv(REPORT_DIR / "fi_band_metrics.csv", index=False)
    results[results["false_safe"].eq(1) | results["false_failure"].eq(1)].to_csv(
        REPORT_DIR / "classification_disagreements.csv", index=False
    )
    results.sort_values("absolute_error", ascending=False).groupby(
        "target", as_index=False, group_keys=False
    ).head(10).to_csv(REPORT_DIR / "worst_errors_by_target.csv", index=False)
    results[results["classical_fi"].sub(1.0).abs().le(0.1)].to_csv(
        REPORT_DIR / "near_threshold_cases.csv", index=False
    )

    config = {
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "test_indices": str(TEST_INDICES_PATH.relative_to(ROOT)),
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "reference": "running classical /analyze endpoint",
        "case_count": len(cases),
        "test_seed_rows": len(select_seed_rows(df, test_indices)),
        "scaled_seed_count": min(12, len(select_seed_rows(df, test_indices))),
        "scale_factors": SCALES,
        "targets": TARGETS,
        "threshold": FAILURE_THRESHOLD,
        "material": {"fiber": FIBER_NAME, "matrix": MATRIX_NAME, "Vf": VF},
        "layup": LAYUP,
        "limitations": [
            "Computational validation only; no experimental validation.",
            "Held-out test rows were not used to fit the saved V4B model, but the same dataset family was used for training.",
            "Scaled test directions are dependent stress probes, not independent samples.",
            "The script does not modify api/main.py, model artifacts, or previous validation reports.",
        ],
    }
    (REPORT_DIR / "validation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n=== V4B targeted validation summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nV4B classification disagreements:", int(
        results["false_safe"].sum() + results["false_failure"].sum()
    ))
    print("Cases with classical FI within ±0.10 of 1.0:", int(
        results["classical_fi"].sub(1.0).abs().le(0.1).sum()
    ))
    print(f"Reports saved to: {REPORT_DIR}")
    print("Keep V2 active until these results are reviewed.")


if __name__ == "__main__":
    main()