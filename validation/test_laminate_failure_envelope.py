"""
Validation of first-ply-failure load-factor analysis.
"""

import numpy as np

from src.laminate_failure_envelope import (
    find_first_ply_failure,
)


# ============================================================
# Material
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
# Puck
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
# Base loading
# ============================================================

N_base = np.array([
    1.0e5,
    5.0e4,
    2.0e4,
])

M_base = np.array([
    0.0,
    0.0,
    0.0,
])


# ============================================================
# First-ply-failure analysis
# ============================================================

results = find_first_ply_failure(
    E1=E1,
    E2=E2,
    G12=G12,
    nu12=nu12,
    ply_angles=ply_angles,
    ply_thickness=ply_thickness,
    N_base=N_base,
    M_base=M_base,
    strengths=strengths,
    puck_parameters=puck_parameters,
)


# ============================================================
# Display
# ============================================================

print("=" * 100)
print("FIRST-PLY-FAILURE LOAD-FACTOR COMPARISON")
print("=" * 100)

print("\nBase load")
print("-" * 100)
print(f"N_base = {N_base}")

print("\nCritical load factors")
print("-" * 100)

for criterion, result in results.items():

    print(
        f"{criterion:<20} "
        f"lambda = {result['critical_load_factor']:.6f}   "
        f"FI = {result['failure_index']:.6f}   "
        f"Ply = {result['ply']}   "
        f"Angle = {result['angle']:>6.1f} deg   "
        f"Surface = {result['surface']}"
    )

    if "failure_mode" in result:
        print(
            f"{'':<20}"
            f"Mode = {result['failure_mode']}"
        )


# ============================================================
# Validation
# ============================================================

assert len(results) == 7

for criterion, result in results.items():

    assert np.isfinite(
        result["critical_load_factor"]
    )

    assert result["critical_load_factor"] > 0.0

    assert np.isfinite(
        result["failure_index"]
    )

    assert np.isclose(
        result["failure_index"],
        1.0,
        atol=2e-5,
    ), (
        f"{criterion}: critical FI is "
        f"{result['failure_index']}"
    )


print("\n" + "=" * 100)
print("FIRST-PLY-FAILURE ANALYSIS PASSED")
print("=" * 100)