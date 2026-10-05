import numpy as np

from src.micromechanics import calculate_lamina_properties
from src.laminate_loading_study import run_loading_study


# ------------------------------------------------------------------
# Material definition
# ------------------------------------------------------------------

Ef = 230e9
Em = 3.5e9

Gf = 30e9
Gm = 1.3e9

nu_f = 0.20
nu_m = 0.35

Vf = 0.60

E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef,
    Em,
    Gf,
    Gm,
    nu_f,
    nu_m,
    Vf,
)


# ------------------------------------------------------------------
# Strengths
# ------------------------------------------------------------------

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}


# ------------------------------------------------------------------
# Laminate
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Run loading study
# ------------------------------------------------------------------

results = run_loading_study(
    ply_angles=ply_angles,
    ply_thickness=ply_thickness,
    strengths=strengths,
    E1=E1,
    E2=E2,
    G12=G12,
    nu12=nu12,
)


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

print("=" * 110)
print("LAMINATE MULTI-LOADING FIRST-PLY-FAILURE STUDY")
print("=" * 110)

print()
print("Laminate")
print("-" * 110)
print(ply_angles)

print()
print("Results")
print("-" * 110)

for _, row in results.iterrows():

    print(
        f"{row['loading_case']:<25}"
        f"{row['criterion']:<20}"
        f"lambda = {row['critical_load_factor']:9.4f}   "
        f"Ply = {int(row['ply'])}   "
        f"Angle = {row['angle_deg']:6.1f} deg   "
        f"Mode = {row['failure_mode']}"
    )


# ------------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------------

expected_cases = 6
expected_criteria = 7

assert len(results) == expected_cases * expected_criteria

assert results["critical_load_factor"].notna().all()
assert (results["critical_load_factor"] > 0).all()

assert results["failure_index"].notna().all()

assert results["ply"].notna().all()
assert results["angle_deg"].notna().all()

assert set(results["loading_case"]) == {
    "Longitudinal tension",
    "Transverse tension",
    "In-plane shear",
    "Biaxial tension",
    "Tension-compression",
    "Combined loading",
}

assert set(results["criterion"]) == {
    "Maximum Stress",
    "Maximum Strain",
    "Tsai-Hill",
    "Tsai-Wu",
    "Hoffman",
    "Hashin",
    "Puck",
}

print()
print("=" * 110)
print("MULTI-LOADING STUDY VALIDATION PASSED")
print("=" * 110)