from src.micromechanics import calculate_lamina_properties
from src.lamina import calculate_Q


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


# Calculate Q matrix
Q = calculate_Q(E1, E2, G12, nu12)


print("Lamina Properties")
print("-----------------")
print(f"E1   = {E1 / 1e9:.3f} GPa")
print(f"E2   = {E2 / 1e9:.3f} GPa")
print(f"G12  = {G12 / 1e9:.3f} GPa")
print(f"nu12 = {nu12:.4f}")

print("\nReduced Stiffness Matrix [Q]")
print("-----------------------------")
print(Q / 1e9)