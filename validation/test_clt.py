import numpy as np
from src.micromechanics import calculate_lamina_properties
from src.lamina import calculate_Q
from src.clt import calculate_ABD, solve_laminate_response
from src.clt import (
    calculate_ABD,
    solve_laminate_response,
    calculate_ply_strains,
    global_to_local_strain,
    local_stress_from_strain
)


# Fibre properties
Ef = 230e9
Gf = 30e9
nu_f = 0.20

# Matrix properties
Em = 3.5e9
Gm = 1.3e9
nu_m = 0.35

# Fibre volume fraction
Vf = 0.60


# Calculate lamina properties
E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef, Em, Gf, Gm, nu_f, nu_m, Vf
)

# Calculate Q
Q = calculate_Q(E1, E2, G12, nu12)


# Define laminate
ply_angles = [0, 45, -45, 90, 90, -45, 45, 0]

# Ply thickness = 0.125 mm
ply_thickness = 0.125e-3


# Calculate ABD
A, B, D, z = calculate_ABD(
    Q,
    ply_angles,
    ply_thickness
)

print("Laminate")
print("--------")
print("Ply angles:", ply_angles)
print(f"Ply thickness: {ply_thickness * 1000:.3f} mm")
print(f"Total thickness: {len(ply_angles) * ply_thickness * 1000:.3f} mm")


print("\nA Matrix")
print("--------")
print(A)


print("\nB Matrix")
print("--------")
print(B)


print("\nD Matrix")
print("--------")
print(D)


print("\nZ Coordinates")
print("------------")
print(z)

print("\nValidation")
print("----------")

if np.allclose(B, 0, atol=1e-10):
    print("PASS: B matrix is approximately zero for symmetric laminate.")
else:
    print("FAIL: B matrix is not zero.")

# Applied in-plane loads
# Units: N/m
N = np.array([
    10000.0,   # Nx
    0.0,       # Ny
    0.0        # Nxy
])

# Applied moments
# Units: N
M = np.array([
    0.0,       # Mx
    0.0,       # My
    0.0        # Mxy
])


# Solve laminate response
mid_plane_strain, curvature = solve_laminate_response(
    A, B, D, N, M
)


print("\nApplied Loads")
print("-------------")
print("N =", N)
print("M =", M)


print("\nMid-Plane Strains")
print("-----------------")
print(mid_plane_strain)


print("\nCurvatures")
print("----------")
print(curvature)

# Calculate global strains at each ply surface
ply_strains = calculate_ply_strains(
    mid_plane_strain,
    curvature,
    z
)


print("\nPly Strains and Local Stresses")
print("------------------------------")

for ply in ply_strains:

    ply_number = ply["ply"]
    angle = ply_angles[ply_number - 1]

    print(f"\nPly {ply_number} ({angle}°)")

    for location in ["bottom", "top"]:

        strain_global = ply[location]

        strain_local = global_to_local_strain(
            strain_global,
            angle
        )

        stress_local = local_stress_from_strain(
            Q,
            strain_local
        )

        print(f"  {location}:")
        print(
            f"    Local strain = "
            f"{strain_local}"
        )

        print(
            f"    Local stress = "
            f"{stress_local / 1e6} MPa"
        )