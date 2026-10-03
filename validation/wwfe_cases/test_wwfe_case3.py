import numpy as np

from src.clt import calculate_ABD
from src.dataset_generator import evaluate_laminate_case


# ============================================================
# WWFE CASE 3
# AS4/3501-6 QUASI-ISOTROPIC LAMINATE
# ============================================================

print("-----------------------------------")
print("WWFE CASE 3 VALIDATION")
print("-----------------------------------")


# ============================================================
# MATERIAL PROPERTIES
# Source:
# Soden, Hinton & Kaddour (1998), Table 1
# ============================================================

E1 = 126e9
E2 = 11e9
G12 = 6.6e9
nu12 = 0.28

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# ============================================================
# LAMINATE
# ============================================================

ply_angles = [
    90,
    45,
    -45,
    0,
    0,
    -45,
    45,
    90
]

total_thickness = 1.1e-3

ply_thickness = (
    total_thickness /
    len(ply_angles)
)


# ============================================================
# PRINT INPUT DATA
# ============================================================

print()
print("Material: AS4/3501-6")

print(
    f"E1       = {E1 / 1e9:.3f} GPa"
)

print(
    f"E2       = {E2 / 1e9:.3f} GPa"
)

print(
    f"G12      = {G12 / 1e9:.3f} GPa"
)

print(
    f"nu12     = {nu12:.3f}"
)

print()
print("Strengths:")

for key, value in strengths.items():
    print(
        f"{key:>3}      = "
        f"{value / 1e6:.1f} MPa"
    )

print()
print(
    f"Stacking sequence = "
    f"{ply_angles}"
)

print(
    f"Total thickness   = "
    f"{total_thickness * 1e3:.4f} mm"
)

print(
    f"Ply thickness     = "
    f"{ply_thickness * 1e3:.4f} mm"
)


# ============================================================
# CHECK ABD MATRIX
# ============================================================

# Calculate reduced stiffness matrix Q first
from src.lamina import calculate_Q

Q = calculate_Q(
    E1,
    E2,
    G12,
    nu12
)

A, B, D, z = calculate_ABD(
    Q,
    ply_angles,
    ply_thickness
)


print()
print("-----------------------------------")
print("ABD VALIDATION")
print("-----------------------------------")

print()
print("A matrix [N/m]:")
print(A)

print()
print("B matrix [N]:")
print(B)

print()
print("D matrix [N*m]:")
print(D)


# ============================================================
# SYMMETRY CHECK
# ============================================================

print()
print("-----------------------------------")
print("SYMMETRY CHECK")
print("-----------------------------------")

print(
    f"Maximum |B| = "
    f"{np.max(np.abs(B)):.6e}"
)

if np.max(np.abs(B)) < 1e-8:
    print(
        "PASS: B matrix is effectively zero."
    )
else:
    print(
        "WARNING: B matrix is not zero."
    )


# ============================================================
# SIMPLE UNAXIAL TEST
#
# WWFE section stresses are converted to
# membrane resultants using:
#
# N = sigma * h
#
# ============================================================

section_stress_x = 100e6
section_stress_y = 0.0

Nx = section_stress_x * total_thickness
Ny = section_stress_y * total_thickness

N = np.array([
    Nx,
    Ny,
    0.0
])

M = np.zeros(3)


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


print()
print("-----------------------------------")
print("100 MPa UNIAXIAL SIGMA-X TEST")
print("-----------------------------------")

print(
    f"sigma_x = "
    f"{section_stress_x / 1e6:.1f} MPa"
)

print(
    f"Nx      = "
    f"{Nx:.3f} N/m"
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
    f"Failure = "
    f"{result['failed']}"
)


# ============================================================
# EQUAL BIAXIAL TEST
# ============================================================

section_stress = 100e6

Nx = section_stress * total_thickness
Ny = section_stress * total_thickness

N = np.array([
    Nx,
    Ny,
    0.0
])


result_biaxial = evaluate_laminate_case(
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


print()
print("-----------------------------------")
print("100 MPa EQUAL BIAXIAL TEST")
print("-----------------------------------")

print(
    f"sigma_x = sigma_y = "
    f"{section_stress / 1e6:.1f} MPa"
)

print(
    f"Nx = Ny = "
    f"{Nx:.3f} N/m"
)

print(
    f"Hashin FI = "
    f"{result_biaxial['hashin_fi']:.6f}"
)

print(
    f"Failure mode = "
    f"{result_biaxial['hashin_mode']}"
)

print(
    f"Failure = "
    f"{result_biaxial['failed']}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("-----------------------------------")
print("WWFE CASE 3 INPUT VALIDATION COMPLETE")
print("-----------------------------------")