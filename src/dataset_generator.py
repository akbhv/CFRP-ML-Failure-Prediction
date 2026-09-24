import numpy as np

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


def evaluate_laminate_case(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    N,
    M
):
    pass

def find_hashin_failure_scale(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    load_direction,
    initial_scale=1.0,
    growth_factor=2.0,
    max_scale=1e6,
    tolerance=1e-5,
    max_iterations=100
):
    """
    Find the load scaling factor at which the laminate
    reaches Hashin failure.

    The loading vector is defined as:

        load = scale * load_direction

    Parameters
    ----------
    load_direction : array-like
        Six-component normalized loading direction:
        [Nx, Ny, Nxy, Mx, My, Mxy]

    Returns
    -------
    failure_scale : float
        Scale at which Hashin FI ≈ 1.

    failure_result : dict
        Evaluation result at the failure boundary.
    """

    load_direction = np.asarray(
        load_direction,
        dtype=float
    )

    if load_direction.shape != (6,):
        raise ValueError(
            "load_direction must contain exactly 6 values."
        )

    if np.linalg.norm(load_direction) == 0:
        raise ValueError(
            "load_direction cannot be zero."
        )

    # Normalize loading direction
    load_direction = (
        load_direction
        / np.linalg.norm(load_direction)
    )

    def evaluate_at_scale(scale):

        load_vector = (
            scale * load_direction
        )

        N = load_vector[:3]
        M = load_vector[3:]

        return evaluate_laminate_case(
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
    # Find an upper bound where failure occurs
    # --------------------------------------------------

    lower_scale = 0.0
    upper_scale = initial_scale

    result = evaluate_at_scale(
        upper_scale
    )

    while (
        result["hashin_fi"] < 1.0
        and upper_scale < max_scale
    ):
        lower_scale = upper_scale

        upper_scale *= growth_factor

        result = evaluate_at_scale(
            upper_scale
        )

    # No failure found within allowed range
    if result["hashin_fi"] < 1.0:
        return None, result

    # --------------------------------------------------
    # Binary search
    # --------------------------------------------------

    for _ in range(max_iterations):

        middle_scale = (
            lower_scale + upper_scale
        ) / 2.0

        middle_result = evaluate_at_scale(
            middle_scale
        )

        if middle_result["hashin_fi"] < 1.0:
            lower_scale = middle_scale
        else:
            upper_scale = middle_scale

        if (
            upper_scale - lower_scale
            < tolerance * upper_scale
        ):
            break

    # Evaluate final boundary
    failure_scale = upper_scale

    failure_result = evaluate_at_scale(
        failure_scale
    )

    return failure_scale, failure_result

def evaluate_laminate_case(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    N,
    M
):
    """
    Evaluate one laminate loading case using CLT
    and multiple failure criteria.

    Returns a dictionary containing laminate-level
    failure information.
    """

    # --------------------------------------------------
    # 1. Calculate reduced stiffness matrix
    # --------------------------------------------------

    Q = calculate_Q(
        E1,
        E2,
        G12,
        nu12
    )

    # --------------------------------------------------
    # 2. Calculate laminate ABD matrices
    # --------------------------------------------------

    A, B, D, z = calculate_ABD(
        Q,
        ply_angles,
        ply_thickness
    )

    # --------------------------------------------------
    # 3. Solve laminate response
    # --------------------------------------------------

    mid_plane_strain, curvature = solve_laminate_response(
        A,
        B,
        D,
        N,
        M
    )

    # --------------------------------------------------
    # 4. Calculate ply strains
    # --------------------------------------------------

    ply_strains = calculate_ply_strains(
        mid_plane_strain,
        curvature,
        z
    )

    # --------------------------------------------------
    # 5. Initialize governing values
    # --------------------------------------------------

    max_stress_fi = 0.0
    tsai_hill_max_fi = 0.0
    tsai_wu_max_fi = -np.inf
    hashin_max_fi = 0.0

    max_stress_location = None
    tsai_hill_location = None
    tsai_wu_location = None
    hashin_location = None

    hashin_mode = None

    # --------------------------------------------------
    # 6. Evaluate every ply and surface
    # --------------------------------------------------

    for ply_data, angle in zip(ply_strains, ply_angles):

        ply_number = ply_data["ply"]

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

            # ------------------------------------------
            # Maximum Stress
            # ------------------------------------------

            _, max_fi, _ = maximum_stress_failure(
                stress_local,
                strengths
            )

            if max_fi > max_stress_fi:
                max_stress_fi = max_fi
                max_stress_location = (
                    ply_number,
                    surface
                )

            # ------------------------------------------
            # Tsai-Hill
            # ------------------------------------------

            tsai_hill_fi, _ = tsai_hill_failure(
                stress_local,
                strengths
            )

            if tsai_hill_fi > tsai_hill_max_fi:
                tsai_hill_max_fi = tsai_hill_fi
                tsai_hill_location = (
                    ply_number,
                    surface
                )

            # ------------------------------------------
            # Tsai-Wu
            # ------------------------------------------

            tsai_wu_fi, _ = tsai_wu_failure(
                stress_local,
                strengths
            )

            if tsai_wu_fi > tsai_wu_max_fi:
                tsai_wu_max_fi = tsai_wu_fi
                tsai_wu_location = (
                    ply_number,
                    surface
                )

            # ------------------------------------------
            # Hashin
            # ------------------------------------------

            _, hashin_fi, mode, _ = hashin_failure(
                stress_local,
                strengths
            )

            if hashin_fi > hashin_max_fi:
                hashin_max_fi = hashin_fi
                hashin_mode = mode
                hashin_location = (
                    ply_number,
                    surface
                )

    # --------------------------------------------------
    # 7. Final result
    # --------------------------------------------------

    return {
        "max_stress_fi": max_stress_fi,
        "tsai_hill_fi": tsai_hill_max_fi,
        "tsai_wu_fi": tsai_wu_max_fi,
        "hashin_fi": hashin_max_fi,

        "max_stress_ply": max_stress_location[0],
        "max_stress_surface": max_stress_location[1],

        "tsai_hill_ply": tsai_hill_location[0],
        "tsai_hill_surface": tsai_hill_location[1],

        "tsai_wu_ply": tsai_wu_location[0],
        "tsai_wu_surface": tsai_wu_location[1],

        "hashin_ply": hashin_location[0],
        "hashin_surface": hashin_location[1],
        "hashin_mode": hashin_mode,

        "failed": hashin_max_fi >= 1.0
    }

def inspect_ply_stresses(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    N,
    M
):
    """
    Return local stresses and Hashin results at every
    ply surface for a given laminate loading case.
    """

    Q = calculate_Q(E1, E2, G12, nu12)

    A, B, D, z = calculate_ABD(
        Q,
        ply_angles,
        ply_thickness
    )

    mid_plane_strain, curvature = solve_laminate_response(
        A,
        B,
        D,
        N,
        M
    )

    ply_strains = calculate_ply_strains(
        mid_plane_strain,
        curvature,
        z
    )

    results = []

    for k, angle in enumerate(ply_angles):

        # Bottom surface
        strain_global = ply_strains[k]["bottom"]

        strain_local = global_to_local_strain(
            strain_global,
            angle
        )

        stress_local = local_stress_from_strain(
            Q,
            strain_local
        )

        hashin = hashin_failure(
            stress_local,
            strengths
        )

        results.append({
            "ply": k + 1,
            "surface": "bottom",
            "angle": angle,
            "sigma_1": stress_local[0],
            "sigma_2": stress_local[1],
            "tau_12": stress_local[2],
            "hashin_fi": hashin[1],
            "hashin_mode": hashin[2]
        })

        # Top surface
        strain_global = ply_strains[k]["top"]

        strain_local = global_to_local_strain(
            strain_global,
            angle
        )

        stress_local = local_stress_from_strain(
            Q,
            strain_local
        )

        hashin = hashin_failure(
            stress_local,
            strengths
        )

        results.append({
            "ply": k + 1,
            "surface": "top",
            "angle": angle,
            "sigma_1": stress_local[0],
            "sigma_2": stress_local[1],
            "tau_12": stress_local[2],
            "hashin_fi": hashin[1],
            "hashin_mode": hashin[2]
        })

    return results