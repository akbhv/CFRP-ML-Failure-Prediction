from src.micromechanics import calculate_lamina_properties
from src.lamina import calculate_Q
from src.transformation import transform_Q


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


angles = [0, 45, -45, 90]

for angle in angles:

    Qbar = transform_Q(Q, angle)

    print(f"\nQbar for {angle}° ply")
    print("---------------------")
    print(Qbar / 1e9)