import numpy as np
from src.micromechanics import calculate_lamina_properties
from src.lamina import calculate_Q
from src.clt import calculate_ABD


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