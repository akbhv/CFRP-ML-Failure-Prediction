"""
WWFE Case 3 quantitative validation.

This script reproduces the Case 3 comparison using the project's actual
CLT and Hashin implementations from src/.

Run from the repository root:
    python -m validation.wwfe_cases.case3_quantitative_validation

Inputs:
    validation/wwfe_cases/case3_experimental_data.csv

Outputs:
    results/wwfe_case3/
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make repository-root imports robust when the file is run directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lamina import calculate_Q
from src.clt import (
    calculate_ABD,
    solve_laminate_response,
    global_to_local_strain,
    local_stress_from_strain,
)
from src.failure_criteria import hashin_failure


# WWFE Case 3: AS4/3501-6 quasi-isotropic laminate.
E1 = 126e9
E2 = 11e9
G12 = 6.6e9
nu12 = 0.28

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}

# WWFE sequence: 90/+45/-45/0/0/-45/+45/90
PLY_ANGLES = [90, 45, -45, 0, 0, -45, 45, 90]
TOTAL_THICKNESS = 1.1e-3
PLY_THICKNESS = TOTAL_THICKNESS / len(PLY_ANGLES)

DATA_PATH = Path(__file__).with_name("case3_experimental_data.csv")
RESULTS_DIR = ROOT / "results" / "wwfe_case3"


def evaluate_hashin_fi(sigma_x_mpa, sigma_y_mpa):
    """Return the maximum Hashin FI for a laminate stress state."""
    Q = calculate_Q(E1, E2, G12, nu12)
    A, B, D, z = calculate_ABD(Q, PLY_ANGLES, PLY_THICKNESS)

    # WWFE section stresses are converted to CLT force resultants:
    # N = sigma * total laminate thickness.
    N = np.array([
        sigma_x_mpa * 1e6 * TOTAL_THICKNESS,
        sigma_y_mpa * 1e6 * TOTAL_THICKNESS,
        0.0,
    ])
    M = np.zeros(3)

    mid_plane_strain, curvature = solve_laminate_response(A, B, D, N, M)

    max_fi = 0.0
    governing_mode = None
    governing_ply = None
    governing_surface = None

    for k, angle in enumerate(PLY_ANGLES):
        for surface_name, z_value in (
            ("bottom", z[k]),
            ("top", z[k + 1]),
        ):
            strain_global = mid_plane_strain + z_value * curvature
            strain_local = global_to_local_strain(strain_global, angle)
            stress_local = local_stress_from_strain(Q, strain_local)

            _, fi, mode, _ = hashin_failure(stress_local, strengths)

            if fi > max_fi:
                max_fi = fi
                governing_mode = mode
                governing_ply = k + 1
                governing_surface = surface_name

    return max_fi, governing_mode, governing_ply, governing_surface


def first_ply_failure_scale(sigma_x_mpa, sigma_y_mpa):
    """
    Find the proportional load magnitude at Hashin FI = 1
    along the experimental point's stress direction.
    """
    radius = np.hypot(sigma_x_mpa, sigma_y_mpa)

    if radius == 0:
        return np.nan

    dx = sigma_x_mpa / radius
    dy = sigma_y_mpa / radius

    lo = 0.0
    hi = 100.0

    while evaluate_hashin_fi(hi * dx, hi * dy)[0] < 1.0:
        hi *= 1.5
        if hi > 20000:
            raise RuntimeError("Could not bracket Hashin FI = 1.")

    for _ in range(60):
        mid = 0.5 * (lo + hi)

        if evaluate_hashin_fi(mid * dx, mid * dy)[0] >= 1.0:
            hi = mid
        else:
            lo = mid

    return hi


def generate_failure_envelope():
    """Generate the CLT–Hashin first-ply-failure envelope."""
    directions = np.linspace(0.0, 2.0 * np.pi, 721)
    envelope = []

    for theta in directions:
        dx = np.cos(theta)
        dy = np.sin(theta)

        scale = first_ply_failure_scale(dx, dy)
        envelope.append((scale * dx, scale * dy))

    return pd.DataFrame(envelope, columns=["sigma_x_MPa", "sigma_y_MPa"])


def quadrant(sigma_x, sigma_y):
    if sigma_x >= 0 and sigma_y >= 0:
        return "Tension–Tension"
    if sigma_x < 0 and sigma_y >= 0:
        return "Compression–Tension"
    if sigma_x < 0 and sigma_y < 0:
        return "Compression–Compression"
    return "Tension–Compression"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experimental = pd.read_csv(DATA_PATH)

    comparison_rows = []

    for point_id, row in experimental.iterrows():
        sx = float(row["sigma_x_MPa"])
        sy = float(row["sigma_y_MPa"])

        point_fi, mode, ply, surface = evaluate_hashin_fi(sx, sy)

        experimental_radius = np.hypot(sx, sy)
        predicted_radius = first_ply_failure_scale(sx, sy)

        ratio = experimental_radius / predicted_radius
        percent_difference = (ratio - 1.0) * 100.0

        comparison_rows.append({
            "id": point_id + 1,
            "experimental_set": row["experimental_set"],
            "sigma_x_MPa": sx,
            "sigma_y_MPa": sy,
            "experimental_resultant_MPa": experimental_radius,
            "predicted_first_ply_resultant_MPa": predicted_radius,
            "experimental_to_predicted_ratio": ratio,
            "percent_difference": percent_difference,
            "point_hashin_FI": point_fi,
            "governing_failure_mode": mode,
            "governing_ply": ply,
            "governing_surface": surface,
            "loading_quadrant": quadrant(sx, sy),
        })

    comparison = pd.DataFrame(comparison_rows)

    # Generate and save theoretical envelope.
    envelope = generate_failure_envelope()
    envelope.to_csv(
        RESULTS_DIR / "first_ply_failure_envelope.csv",
        index=False,
    )

    comparison.to_csv(
        RESULTS_DIR / "quantitative_comparison.csv",
        index=False,
    )

    by_set = comparison.groupby("experimental_set").agg(
        points=("id", "count"),
        mean_ratio=("experimental_to_predicted_ratio", "mean"),
        median_ratio=("experimental_to_predicted_ratio", "median"),
        mean_percent_difference=("percent_difference", "mean"),
        median_percent_difference=("percent_difference", "median"),
        points_FI_ge_1=("point_hashin_FI", lambda x: int((x >= 1).sum())),
    ).reset_index()

    by_set.to_csv(
        RESULTS_DIR / "comparison_by_set.csv",
        index=False,
    )

    by_quadrant = comparison.groupby("loading_quadrant").agg(
        points=("id", "count"),
        mean_ratio=("experimental_to_predicted_ratio", "mean"),
        median_ratio=("experimental_to_predicted_ratio", "median"),
        mean_percent_difference=("percent_difference", "mean"),
        median_percent_difference=("percent_difference", "median"),
        FI_ge_1=("point_hashin_FI", lambda x: int((x >= 1).sum())),
        FI_lt_1=("point_hashin_FI", lambda x: int((x < 1).sum())),
    ).reset_index()

    by_quadrant.to_csv(
        RESULTS_DIR / "comparison_by_quadrant.csv",
        index=False,
    )

    # Experimental points + theoretical envelope.
    plt.figure(figsize=(9, 8))

    for name, group in experimental.groupby("experimental_set"):
        plt.scatter(
            group["sigma_x_MPa"],
            group["sigma_y_MPa"],
            s=35,
            alpha=0.8,
            label=name,
        )

    plt.plot(
        envelope["sigma_x_MPa"],
        envelope["sigma_y_MPa"],
        linewidth=2.2,
        label="CLT–Hashin predicted first-ply-failure envelope",
    )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("sigma_x (MPa)")
    plt.ylabel("sigma_y (MPa)")
    plt.title(
        "WWFE Case 3: Experimental Failure Data vs "
        "CLT–Hashin First-Ply-Failure Envelope"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "experimental_vs_hashin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Experimental/predicted radial-load ratio.
    plt.figure(figsize=(11, 5))

    plt.axhline(
        1.0,
        linewidth=1.5,
        label="Ratio = 1",
    )

    plt.scatter(
        np.arange(1, len(comparison) + 1),
        comparison["experimental_to_predicted_ratio"],
        s=28,
    )

    plt.xlabel("Experimental data point")
    plt.ylabel("Experimental / predicted first-ply load ratio")
    plt.title("WWFE Case 3: Experimental-to-CLT–Hashin First-Ply Load Ratio")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "experimental_predicted_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nWWFE CASE 3 VALIDATION")
    print("=" * 60)
    print(f"Experimental points: {len(comparison)}")
    print(
        "Points with Hashin FI >= 1: "
        f"{int((comparison['point_hashin_FI'] >= 1).sum())}"
    )
    print(
        "Points with Hashin FI < 1: "
        f"{int((comparison['point_hashin_FI'] < 1).sum())}"
    )
    print(
        "Mean experimental/predicted ratio: "
        f"{comparison['experimental_to_predicted_ratio'].mean():.4f}"
    )
    print(
        "Median experimental/predicted ratio: "
        f"{comparison['experimental_to_predicted_ratio'].median():.4f}"
    )
    print(
        "Mean percentage difference: "
        f"{comparison['percent_difference'].mean():.2f}%"
    )
    print(
        "Median percentage difference: "
        f"{comparison['percent_difference'].median():.2f}%"
    )

    print("\nResults written to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
