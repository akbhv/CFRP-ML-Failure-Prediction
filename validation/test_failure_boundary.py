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


# --------------------------------------------------
# Laminate
# --------------------------------------------------

ply_angles = [
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0
]

ply_thickness = 0.125e-3


# --------------------------------------------------
# Loading direction
# --------------------------------------------------

load_direction = np.array([
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])


# --------------------------------------------------
# Find failure boundary
# --------------------------------------------------

failure_scale, result = find_hashin_failure_scale(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    load_direction
)


# --------------------------------------------------
# Print result
# --------------------------------------------------

print("-----------------------------------")
print("HASHIN FAILURE BOUNDARY")
print("-----------------------------------")

print(
    f"Failure scale = "
    f"{failure_scale:.6f}"
)

print(
    f"Hashin FI = "
    f"{result['hashin_fi']:.6f}"
)

print(
    f"Failure mode = "
    f"{result['hashin_mode']}"
)

print(
    f"Ply = "
    f"{result['hashin_ply']}"
)

print(
    f"Surface = "
    f"{result['hashin_surface']}"
)

print("E1 =", E1)
print("E2 =", E2)
print("G12 =", G12)
print("nu12 =", nu12)
print("strengths =", strengths)
print("ply_angles =", ply_angles)
print("ply_thickness =", ply_thickness)
print("load_direction =", load_direction)