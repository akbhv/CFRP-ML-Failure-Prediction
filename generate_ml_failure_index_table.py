"""
Generate a six-loading-case failure-index heatmap from the trained V5 ML surrogate.

Expected model files (relative to project root):
  data/processed/ml_multitarget_v5/feature_scaler.joblib
  results/multitarget_dnn_v5/<target>/<target>_model.keras

Input CSV:
  results/loading_study/loading_cases_inputs.csv

Required CSV columns:
  Loading Case, Nx, Ny, Nxy, Mx, My, Mxy

IMPORTANT:
- Fill in real, documented load resultants for each case. Do not use zeros as
  placeholder values for a real results slide.
- Units expected by the API/model: Nx, Ny, Nxy in N/m; Mx, My, Mxy in N.
- The surrogate takes only these six load resultants as features. It does not
  receive material, layup, ply angle, thickness, or stacking sequence.
- This script reports FI predictions only; it does not calculate critical
  first-ply-failure load factors or validate predictions against experiments.
"""

from pathlib import Path
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOADING_ORDER = [
    "Longitudinal tension",
    "Transverse tension",
    "In-plane shear",
    "Biaxial tension",
    "Tension-compression",
    "Combined loading",
]

FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

# Must match the V5 training target/model folder names.
TARGETS = [
    ("Maximum Stress", "maximum_stress"),
    ("Maximum Strain", "maximum_strain"),
    ("Tsai-Hill", "tsai_hill"),
    ("Tsai-Wu", "tsai_wu"),
    ("Hoffman", "hoffman"),
    ("Hashin", "hashin"),
    ("Puck", "puck"),
]


def load_v5_models(project_root: Path):
    """Load the V5 feature scaler and seven independently trained Keras models."""
    scaler_path = project_root / "data/processed/ml_multitarget_v5/feature_scaler.joblib"
    model_root = project_root / "results/multitarget_dnn_v5"

    if not scaler_path.exists():
        raise FileNotFoundError(f"V5 scaler not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    models = {}

    for display_name, folder_name in TARGETS:
        model_path = model_root / folder_name / f"{folder_name}_model.keras"
        if not model_path.exists():
            # Some project versions use the display target as folder/file naming.
            alternatives = [
                model_root / display_name / f"{display_name}_model.keras",
                model_root / folder_name / f"{display_name}_model.keras",
            ]
            model_path = next((p for p in alternatives if p.exists()), None)

        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model for {display_name!r} not found. Checked expected path "
                f"{model_root / folder_name / (folder_name + '_model.keras')} "
                "and common alternatives. Update TARGETS/model paths to match "
                "your actual V5 model directory names."
            )

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required. Run this script in the same Python "
                "environment used to train/load the V5 models."
            ) from exc

        models[display_name] = load_model(model_path, compile=False)

    return scaler, models


def predict_failure_indices(inputs: pd.DataFrame, scaler, models) -> pd.DataFrame:
    """Scale the six load features and predict seven failure indices."""
    X = inputs[FEATURES].astype(float).to_numpy()
    X_scaled = scaler.transform(X)
    out = pd.DataFrame(index=inputs.index)

    for display_name, _ in TARGETS:
        predictions = np.asarray(models[display_name].predict(X_scaled, verbose=0)).reshape(-1)
        out[display_name] = predictions

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Path to CFRP-ML-Failure-Prediction project root (default: current directory).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/loading_study/loading_cases_inputs.csv"),
        help="CSV containing six loading cases and six load resultants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/loading_study/ml_failure_index"),
        help="Output directory for predicted values and heatmap.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    input_path = args.input if args.input.is_absolute() else project_root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Create it from loading_cases_inputs_template.csv and enter the actual "
            "load vectors for all six cases. The critical-load-factor CSV does not "
            "contain enough information to reconstruct those vectors by itself."
        )

    cases = pd.read_csv(input_path)
    required = ["Loading Case", *FEATURES]
    missing = [col for col in required if col not in cases.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    cases["Loading Case"] = cases["Loading Case"].astype(str).str.strip()
    if cases["Loading Case"].duplicated().any():
        duplicates = cases.loc[cases["Loading Case"].duplicated(), "Loading Case"].tolist()
        raise ValueError(f"Duplicate loading case names: {duplicates}")

    missing_cases = [name for name in LOADING_ORDER if name not in set(cases["Loading Case"])]
    if missing_cases:
        raise ValueError(f"Input CSV is missing loading cases: {missing_cases}")

    cases = cases.set_index("Loading Case").loc[LOADING_ORDER].reset_index()
    cases[FEATURES] = cases[FEATURES].apply(pd.to_numeric, errors="coerce")
    if cases[FEATURES].isna().any().any():
        raise ValueError("All six load-resultant fields must contain numeric values.")
    if not np.isfinite(cases[FEATURES].to_numpy()).all():
        raise ValueError("Load-resultant inputs must be finite numbers.")

    if (cases[FEATURES].abs().sum(axis=1) == 0).any():
        zero_cases = cases.loc[cases[FEATURES].abs().sum(axis=1) == 0, "Loading Case"].tolist()
        raise ValueError(
            f"All-zero load vector found for {zero_cases}. Enter meaningful, documented "
            "load vectors; all-zero inputs are not suitable for the results slide."
        )

    scaler, models = load_v5_models(project_root)
    predicted = predict_failure_indices(cases, scaler, models)

    results = pd.concat([cases[["Loading Case", *FEATURES]], predicted], axis=1)
    csv_path = output_dir / "ml_failure_index_by_loading_case.csv"
    results.to_csv(csv_path, index=False, float_format="%.8g")

    matrix = predicted.copy()
    matrix.index = cases["Loading Case"]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    # Shared continuous scale; the threshold FI=1 is shown as a contour when in range.
    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Model returned no finite predictions.")
    vmin = min(0.0, float(finite_values.min()))
    vmax = max(1.0, float(finite_values.max()))
    if vmax <= vmin:
        vmax = vmin + 1.0

    image = ax.imshow(values, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Failure Criterion")
    ax.set_ylabel("Loading Case")
    ax.set_title("ML Surrogate Failure-Index Predictions Across Loading Cases")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            # Make text readable on both ends of the colour scale.
            text_color = "white" if value > (vmin + vmax) / 2 else "black"
            label = f"{value:.3f}"
            if value >= 1.0:
                label += " *"
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Predicted Failure Index (FI)")
    fig.text(
        0.5, 0.01,
        "* FI ≥ 1 indicates predicted threshold reached/exceeded for that criterion. "
        "Predictions apply to the trained V5 configuration and input domain.",
        ha="center", fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    png_path = output_dir / "ml_failure_index_heatmap.png"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    # A row-level risk summary, based on the largest FI within each loading case.
    summary_rows = []
    for i, row in matrix.iterrows():
        criterion = row.idxmax()
        max_fi = float(row.max())
        summary_rows.append({
            "Loading Case": i,
            "Highest predicted FI": max_fi,
            "Highest-FI criterion": criterion,
            "Threshold status": "FI >= 1" if max_fi >= 1.0 else "All listed FIs < 1",
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "ml_failure_index_risk_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8g")

    print("\nGenerated ML failure-index results:")
    print(f"  Heatmap: {png_path}")
    print(f"  Full results: {csv_path}")
    print(f"  Risk summary: {summary_path}")
    print("\nHighest predicted FI by loading case:")
    print(summary.to_string(index=False))
    print(
        "\nInterpretation: compare each FI with its own threshold of 1. "
        "The largest FI identifies the largest numerical index for that case, "
        "but different criteria represent different failure formulations and "
        "should not be treated as interchangeable physical failure modes."
    )


if __name__ == "__main__":
    main()
