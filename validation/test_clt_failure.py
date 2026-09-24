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
    tsai_hill_failure,
    tsai_wu_failure,
    hashin_failure
)

max_hashin_fi = 0.0
max_hashin_mode = None
max_hashin_ply = None
max_hashin_surface = None

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

        tsai_wu_fi, tsai_wu_failed = tsai_wu_failure(
            stress_local,
            strengths
        )

        hashin_indices, hashin_max, hashin_mode, hashin_failed = hashin_failure(
            stress_local,
            strengths
        )
        if hashin_max > max_hashin_fi:
            max_hashin_fi = hashin_max
            max_hashin_mode = hashin_mode
            max_hashin_ply = ply_number
            max_hashin_surface = surface

        print(
            f"Ply {ply_number} | "
            f"{surface} | "
            f"angle = {angle}° | "
            f"Max Stress FI = {max_fi:.6f} | "
            f"Tsai-Hill FI = {tsai_hill_fi:.6f} | "
            f"Tsai-Wu FI = {tsai_wu_fi:.6f}"
        )
        print(
            f"Hashin: "
            f"FI={hashin_max:.6f}, "
            f"Mode={hashin_mode}, "
            f"Failed={hashin_failed}"
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

print(
    f"\nMaximum Hashin FI = "
    f"{max_hashin_fi:.6f}"
)

print(
    f"Hashin governing mode = "
    f"{max_hashin_mode}"
)

print(
    f"Hashin governing ply/surface = "
    f"Ply {max_hashin_ply} ({max_hashin_surface})"
)

print(
    f"Hashin laminate failed = "
    f"{max_hashin_fi >= 1.0}"
)