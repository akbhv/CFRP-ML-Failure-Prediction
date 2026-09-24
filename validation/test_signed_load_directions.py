import numpy as np

from src.dataset_generator import find_hashin_failure_scale


# ==================================================
# MATERIAL
# ==================================================

E1 = 139.4e9
E2 = 8.554729011689692e9
G12 = 3.051643192488263e9
nu12 = 0.26

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# ==================================================
# LAMINATE
# ==================================================

ply_angles = [0, 45, -45, 90, 90, -45, 45, 0]
ply_thickness = 0.125e-3


# ==================================================
# LOAD DIRECTIONS
# ==================================================

load_cases = {

    "Nx+":  [1, 0, 0, 0, 0, 0],
    "Nx-":  [-1, 0, 0, 0, 0, 0],

    "Ny+":  [0, 1, 0, 0, 0, 0],
    "Ny-":  [0, -1, 0, 0, 0, 0],

    "Nxy+": [0, 0, 1, 0, 0, 0],
    "Nxy-": [0, 0, -1, 0, 0, 0],

    "Mx+":  [0, 0, 0, 1, 0, 0],
    "Mx-":  [0, 0, 0, -1, 0, 0],

    "My+":  [0, 0, 0, 0, 1, 0],
    "My-":  [0, 0, 0, 0, -1, 0],

    "Mxy+": [0, 0, 0, 0, 0, 1],
    "Mxy-": [0, 0, 0, 0, 0, -1],
}


# ==================================================
# RUN
# ==================================================

print("-----------------------------------")
print("SIGNED LOAD DIRECTION TEST")
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

    print()
    print(name)
    print(f"Failure scale = {failure_scale:.6f}")
    print(f"Hashin FI = {result['hashin_fi']:.6f}")
    print(f"Failure mode = {result['hashin_mode']}")
    print(f"Ply = {result['hashin_ply']}")
    print(f"Surface = {result['hashin_surface']}")