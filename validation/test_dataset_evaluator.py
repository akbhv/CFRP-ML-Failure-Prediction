import numpy as np

from src.micromechanics import calculate_lamina_properties
from src.dataset_generator import evaluate_laminate_case


# --------------------------------------------------
# Material development values
# --------------------------------------------------

Ef = 230e9
Gf = 30e9
nu_f = 0.20

Em = 3.5e9
Gm = 1.3e9
nu_m = 0.35

Vf = 0.60


# --------------------------------------------------
# Calculate lamina properties
# --------------------------------------------------

E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef,
    Em,
    Gf,
    Gm,
    nu_f,
    nu_m,
    Vf
)


# --------------------------------------------------
# Material strengths
# --------------------------------------------------

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
# Loading
# --------------------------------------------------

N = np.array([
    10000.0,
    0.0,
    0.0
])

M = np.array([
    0.0,
    0.0,
    0.0
])


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

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


# --------------------------------------------------
# Print
# --------------------------------------------------

print("Dataset evaluator result")
print("------------------------")

for key, value in result.items():
    print(f"{key}: {value}")