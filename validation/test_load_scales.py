import numpy as np

from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import evaluate_laminate_case


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
    Ef, Em, Gf, Gm, nu_f, nu_m, Vf
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
    0, 45, -45, 90,
    90, -45, 45, 0
]

ply_thickness = 0.125e-3


# --------------------------------------------------
# Test cases
# --------------------------------------------------

cases = {
    "Nx = 10,000": (
        np.array([10000.0, 0.0, 0.0]),
        np.zeros(3)
    ),

    "Mx = 1": (
        np.zeros(3),
        np.array([1.0, 0.0, 0.0])
    ),

    "Mx = 10": (
        np.zeros(3),
        np.array([10.0, 0.0, 0.0])
    ),

    "Mx = 100": (
        np.zeros(3),
        np.array([100.0, 0.0, 0.0])
    ),
}


# --------------------------------------------------
# Run cases
# --------------------------------------------------

for name, (N, M) in cases.items():

    result = evaluate_laminate_case(
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

    print("\n" + "-" * 45)
    print(name)
    print("-" * 45)

    print(f"Maximum Stress FI : {result['max_stress_fi']:.6e}")
    print(f"Tsai-Hill FI      : {result['tsai_hill_fi']:.6e}")
    print(f"Tsai-Wu FI        : {result['tsai_wu_fi']:.6e}")
    print(f"Hashin FI         : {result['hashin_fi']:.6e}")
    print(f"Hashin mode       : {result['hashin_mode']}")
    print(f"Failed            : {result['failed']}")