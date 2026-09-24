import numpy as np

from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import find_hashin_failure_scale


# --------------------------------------------------
# Material
# --------------------------------------------------

Ef = 230e9
Gf = 30e9
nu_f = 0.20

Em = 3.5e9
Gm = 1.3e9
nu_m = 0.35

Vf = 0.60

E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef,
    Em,
    Gf,
    Gm,
    nu_f,
    nu_m,
    Vf
)

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}

ply_angles = [0, 45, -45, 90, 90, -45, 45, 0]
ply_thickness = 0.125e-3


load_cases = {
    "Pure Nx": [1, 0, 0, 0, 0, 0],
    "Pure Ny": [0, 1, 0, 0, 0, 0],
    "Pure Nxy": [0, 0, 1, 0, 0, 0],
    "Pure Mx": [0, 0, 0, 1, 0, 0],
    "Pure My": [0, 0, 0, 0, 1, 0],
    "Pure Mxy": [0, 0, 0, 0, 0, 1],
    "Combined membrane": [1, 0.5, 0.25, 0, 0, 0],
    "Combined membrane+bending": [1, 0.5, 0.25, 0.2, -0.1, 0.1],
}


print("-----------------------------------")
print("MULTIPLE HASHIN FAILURE BOUNDARIES")
print("-----------------------------------")

for name, direction in load_cases.items():

    failure_scale, result = find_hashin_failure_scale(
        E1,
        E2,
        G12,
        nu12,
        strengths,
        ply_angles,
        ply_thickness,
        direction
    )

    print(f"\n{name}")
    print(f"Failure scale = {failure_scale:.6f}")
    print(f"Hashin FI = {result['hashin_fi']:.6f}")
    print(f"Failure mode = {result['hashin_mode']}")
    print(f"Ply = {result['hashin_ply']}")
    print(f"Surface = {result['hashin_surface']}")

print("E1 =", E1)
print("E2 =", E2)
print("G12 =", G12)
print("nu12 =", nu12)
print("strengths =", strengths)
print("ply_angles =", ply_angles)
print("ply_thickness =", ply_thickness)