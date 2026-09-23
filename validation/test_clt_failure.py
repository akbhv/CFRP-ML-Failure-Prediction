import numpy as np

from src.micromechanics import calculate_lamina_properties
from src.lamina import calculate_Q
from src.clt import (
    calculate_ABD,
    solve_laminate_response,
    calculate_ply_strains,
    global_to_local_strain,
    local_stress_from_strain
)
from src.failure_criteria import (
    maximum_stress_failure,
    tsai_hill_failure
)


# --------------------------------------------------
# 1. Material properties
# --------------------------------------------------

Ef = 230e9
Gf = 30e9
nu_f = 0.20

Em = 3.5e9
Gm = 1.3e9
nu_m = 0.35

Vf = 0.60


# --------------------------------------------------
# 2. Calculate lamina properties
# --------------------------------------------------

E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef, Em, Gf, Gm, nu_f, nu_m, Vf
)


# --------------------------------------------------
# 3. Calculate reduced stiffness matrix Q
# --------------------------------------------------

Q = calculate_Q(E1, E2, G12, nu12)


# --------------------------------------------------
# 4. Define laminate
# --------------------------------------------------

ply_angles = [0, 45, -45, 90, 90, -45, 45, 0]

ply_thickness = 0.125e-3


# --------------------------------------------------
# 5. Calculate ABD matrices
# --------------------------------------------------

A, B, D, z = calculate_ABD(
    Q,
    ply_angles,
    ply_thickness
)


# --------------------------------------------------
# 6. Apply laminate loading
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
# 7. Solve laminate response
# --------------------------------------------------

mid_plane_strain, curvature = solve_laminate_response(
    A,
    B,
    D,
    N,
    M
)


# --------------------------------------------------
# 8. Calculate ply strains
# --------------------------------------------------

ply_strains = calculate_ply_strains(
    mid_plane_strain,
    curvature,
    z
)


# --------------------------------------------------
# 9. Material strengths
# --------------------------------------------------

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# --------------------------------------------------
# 10. Evaluate every ply
# --------------------------------------------------

maximum_laminate_fi = 0.0
governing_ply = None

for ply_data, angle in zip(ply_strains, ply_angles):

    ply_number = ply_data["ply"]

    # Check bottom and top surfaces
    for surface in ["bottom", "top"]:

        strain_global = ply_data[surface]

        # Global → local strain
        strain_local = global_to_local_strain(
            strain_global,
            angle
        )

        # Local stress
        stress_local = local_stress_from_strain(
            Q,
            strain_local
        )

        # Maximum Stress criterion
        failure_indices, max_fi, failed = maximum_stress_failure(
            stress_local,
            strengths
        )

        tsai_hill_fi, tsai_hill_failed = tsai_hill_failure(
            stress_local,
            strengths
        )

        print(
            f"Ply {ply_number} | "
            f"{surface} | "
            f"angle = {angle}° | "
            f"Max Stress FI = {max_fi:.6f} | "
            f"Tsai-Hill FI = {tsai_hill_fi:.6f}"
        )

        # Track governing point
        if max_fi > maximum_laminate_fi:
            maximum_laminate_fi = max_fi
            governing_ply = (
                ply_number,
                surface
            )


# --------------------------------------------------
# 11. Final laminate result
# --------------------------------------------------

laminate_failed = maximum_laminate_fi >= 1.0


print("\n-----------------------------------")
print("LAMINATE FAILURE SUMMARY")
print("-----------------------------------")

print(
    f"Maximum laminate FI = "
    f"{maximum_laminate_fi:.6f}"
)

print(
    f"Governing ply/surface = "
    f"Ply {governing_ply[0]} ({governing_ply[1]})"
)

print(
    f"Laminate failed = "
    f"{laminate_failed}"
)