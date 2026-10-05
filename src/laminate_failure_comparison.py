"""
Laminate-level comparison of validated failure criteria.

Pipeline:
    Laminate loads
        -> CLT response
        -> ply global strains
        -> local strains
        -> local stresses
        -> failure criteria

The module evaluates both surfaces of every ply.
"""

import numpy as np

from src.clt import (
    calculate_ABD,
    solve_laminate_response,
    calculate_ply_strains,
    global_to_local_strain,
    local_stress_from_strain,
)

from src.failure_comparison import evaluate_all_criteria
from src.lamina import calculate_Q


def evaluate_laminate_failure(
    E1,
    E2,
    G12,
    nu12,
    ply_angles,
    ply_thickness,
    N,
    M,
    strengths,
    puck_parameters=None,
):
    """
    Evaluate all validated failure criteria at both surfaces
    of every ply in a laminate.

    Parameters
    ----------
    E1, E2, G12 : float
        Lamina elastic properties in Pa.

    nu12 : float
        Major Poisson's ratio.

    ply_angles : list or array
        Ply orientations in degrees.

    ply_thickness : float
        Thickness of each ply in metres.

    N : array-like
        In-plane laminate force resultants [Nx, Ny, Nxy].

    M : array-like
        Laminate moment resultants [Mx, My, Mxy].

    strengths : dict
        Lamina strengths in Pa.

    puck_parameters : dict, optional
        Parameters required by Puck.

    Returns
    -------
    dict
        Laminate-level failure results for every ply surface.
    """

    ply_angles = np.asarray(ply_angles, dtype=float)
    N = np.asarray(N, dtype=float)
    M = np.asarray(M, dtype=float)

    # ========================================================
    # Lamina stiffness
    # ========================================================

    Q = calculate_Q(
        E1,
        E2,
        G12,
        nu12,
    )

    # ========================================================
    # Classical Lamination Theory
    # ========================================================

    A, B, D, z = calculate_ABD(
        Q,
        ply_angles,
        ply_thickness,
    )

    mid_plane_strain, curvature = solve_laminate_response(
        A,
        B,
        D,
        N,
        M,
    )

    # ========================================================
    # Ply global strains
    # ========================================================

    ply_strains = calculate_ply_strains(
        mid_plane_strain,
        curvature,
        z,
    )

    # ========================================================
    # Evaluate every ply surface
    # ========================================================

    results = []

    for ply_index, theta in enumerate(ply_angles):

        ply_number = ply_index + 1

        # calculate_ply_strains() returns:
        #{
        #     "ply": ply_number,
        #     "bottom": strain_vector,
        #     "top": strain_vector
        # }

        ply_strain_data = ply_strains[ply_index]

        bottom_strain_global = ply_strain_data["bottom"]
        top_strain_global = ply_strain_data["top"]

        for surface, strain_global in [
            ("bottom", bottom_strain_global),
            ("top", top_strain_global),
        ]:

            strain_local = global_to_local_strain(
                strain_global,
                theta,
            )

            stress_local = local_stress_from_strain(
                Q,
                strain_local,
            )

            criterion_results = evaluate_all_criteria(
                stress_local=stress_local,
                strengths=strengths,
                E1=E1,
                E2=E2,
                G12=G12,
                nu12=nu12,
                puck_parameters=puck_parameters,
            )

            results.append({
                "ply": ply_number,
                "angle": float(theta),
                "surface": surface,

                "strain_global": np.asarray(
                    strain_global,
                    dtype=float,
                ),

                "strain_local": np.asarray(
                    strain_local,
                    dtype=float,
                ),

                "stress_local": np.asarray(
                    stress_local,
                    dtype=float,
                ),

                "criteria": criterion_results,
            })

    return {
        "A": A,
        "B": B,
        "D": D,
        "z": z,
        "mid_plane_strain": mid_plane_strain,
        "curvature": curvature,
        "ply_results": results,
    }


def find_governing_failure(
    laminate_results,
    criterion,
):
    """
    Find the highest failure index for a selected criterion.

    Parameters
    ----------
    laminate_results : dict
        Output from evaluate_laminate_failure().

    criterion : str
        One of the validated criterion names.

    Returns
    -------
    dict
        Governing ply/surface result.
    """

    ply_results = laminate_results["ply_results"]

    if not ply_results:
        raise ValueError("No ply results were found.")

    governing_result = max(
        ply_results,
        key=lambda result: result["criteria"][criterion]["failure_index"],
    )

    return governing_result