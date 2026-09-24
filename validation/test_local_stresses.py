import numpy as np

from src.dataset_generator import inspect_ply_stresses


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


# ==================================================
# LOAD CASES
# ==================================================

load_cases = {

    "Nx compression": (
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0])
    ),

    "Ny compression": (
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 0.0])
    ),

    "Mx": (
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 0.0, 0.0])
    ),

    "My": (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 100.0, 0.0])
    )
}


# ==================================================
# RUN
# ==================================================

for name, (N, M) in load_cases.items():

    print()
    print("===================================")
    print(name)
    print("===================================")

    results = inspect_ply_stresses(
        E1,
        E2,
        G12,
        nu12,
        strengths,
        ply_angles,
        ply_thickness,
        N,
        M
    )

    for result in results:

        print(
            f"Ply {result['ply']:2d} "
            f"{result['surface']:6s} "
            f"θ={result['angle']:3d}° | "
            f"σ1={result['sigma_1']/1e6:9.3f} MPa | "
            f"σ2={result['sigma_2']/1e6:9.3f} MPa | "
            f"τ12={result['tau_12']/1e6:9.3f} MPa | "
            f"FI={result['hashin_fi']:.6f} | "
            f"{result['hashin_mode']}"
        )