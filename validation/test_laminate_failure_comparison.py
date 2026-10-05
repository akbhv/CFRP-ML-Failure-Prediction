"""
Validation of laminate-level failure-criteria comparison.

Uses the project's development CFRP material and
[0/45/-45/90/90/-45/45/0] laminate.
"""

import numpy as np

from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)


# ============================================================
# Material properties
# ============================================================

E1 = 139.4e9
E2 = 8.554729e9
G12 = 3.051643e9
nu12 = 0.26

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}


# ============================================================
# Puck parameters
# ============================================================

puck_parameters = {
    "p_perp_parallel_t": 0.30,
    "p_perp_parallel_c": 0.25,
    "p_perp_perp_c": 0.25,
    "m_sigma_f": 1.10,
    "E1": E1,
    "Ef": 230e9,
    "nu12": nu12,
    "nuf": 0.20,
}


# ============================================================
# Laminate
# ============================================================

ply_angles = [
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0,
]

ply_thickness = 0.125e-3


# ============================================================
# Applied laminate loads
# ============================================================

# Units:
# N/m for membrane loads
# N for moment loads

N = np.array([
    1.0e5,
    5.0e4,
    2.0e4,
])

M = np.array([
    0.0,
    0.0,
    0.0,
])


# ============================================================
# Evaluate laminate
# ============================================================

results = evaluate_laminate_failure(
    E1=E1,
    E2=E2,
    G12=G12,
    nu12=nu12,
    ply_angles=ply_angles,
    ply_thickness=ply_thickness,
    N=N,
    M=M,
    strengths=strengths,
    puck_parameters=puck_parameters,
)


# ============================================================
# Basic structural checks
# ============================================================

assert len(results["ply_results"]) == 16

assert results["A"].shape == (3, 3)
assert results["B"].shape == (3, 3)
assert results["D"].shape == (3, 3)

assert results["mid_plane_strain"].shape == (3,)
assert results["curvature"].shape == (3,)


# ============================================================
# Display ply results
# ============================================================

print("=" * 90)
print("LAMINATE FAILURE CRITERIA COMPARISON")
print("=" * 90)

print("\nLaminate")
print("-" * 90)
print(
    "[0/45/-45/90/90/-45/45/0]"
)

print("\nApplied membrane loads")
print("-" * 90)
print(f"N = {N}")

print("\nPly-surface results")
print("-" * 90)

for result in results["ply_results"]:

    print(
        f"Ply {result['ply']:>2} "
        f"({result['angle']:>6.1f} deg) "
        f"{result['surface']:<6} | "
        f"sigma_local = "
        f"["
        f"{result['stress_local'][0] / 1e6:>8.2f}, "
        f"{result['stress_local'][1] / 1e6:>8.2f}, "
        f"{result['stress_local'][2] / 1e6:>8.2f}"
        f"] MPa"
    )


# ============================================================
# Governing result for each criterion
# ============================================================

criteria = [
    "Maximum Stress",
    "Maximum Strain",
    "Tsai-Hill",
    "Tsai-Wu",
    "Hoffman",
    "Hashin",
    "Puck",
]

print("\n")
print("GOVERNING FAILURE INDEX BY CRITERION")
print("-" * 90)

for criterion in criteria:

    governing = find_governing_failure(
        results,
        criterion,
    )

    criterion_result = governing["criteria"][criterion]

    print(
        f"{criterion:<20} "
        f"FI = {criterion_result['failure_index']:.6f}   "
        f"Ply = {governing['ply']}   "
        f"Angle = {governing['angle']:>6.1f} deg   "
        f"Surface = {governing['surface']}"
    )

    if "failure_mode" in criterion_result:

        print(
            f"{'':<20}"
            f"Mode = {criterion_result['failure_mode']}"
        )


print("\n" + "=" * 90)
print("LAMINATE COMPARISON TEST PASSED")
print("=" * 90)