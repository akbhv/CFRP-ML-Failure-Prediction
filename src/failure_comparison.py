"""
Unified comparison interface for validated composite failure criteria.

This module evaluates the same local stress state using all currently
validated failure criteria.

LaRC03 and LaRC04 are intentionally excluded until their implementations
are complete and properly parameterized.
"""

import numpy as np

from src.failure_criteria import (
    maximum_stress_failure,
    maximum_strain_failure,
    tsai_hill_failure,
    tsai_wu_failure,
    hoffman_failure,
    hashin_failure,
    puck_failure,
)


def evaluate_all_criteria(
    stress_local,
    strengths,
    E1,
    E2,
    G12,
    nu12,
    puck_parameters=None,
):
    """
    Evaluate all currently validated failure criteria for one
    local material stress state.

    Parameters
    ----------
    stress_local : array-like
        Local material stresses [sigma_1, sigma_2, tau_12] in Pa.

    strengths : dict
        Material strengths in Pa:
        Xt, Xc, Yt, Yc, S.

    E1, E2, G12 : float
        Lamina elastic properties in Pa.

    nu12 : float
        Major Poisson's ratio.

    puck_parameters : dict, optional
        Parameters required by the Puck criterion.

    Returns
    -------
    dict
        Results from every validated criterion.
    """

    stress_local = np.asarray(stress_local, dtype=float)

    if stress_local.shape != (3,):
        raise ValueError(
            "stress_local must contain exactly "
            "[sigma_1, sigma_2, tau_12]."
        )

    # ========================================================
    # Convert local stress to local engineering strain
    # ========================================================

    sigma_1, sigma_2, tau_12 = stress_local

    strain_local = np.array([
        (sigma_1 - nu12 * sigma_2) / E1,
        (sigma_2 - nu12 * sigma_1) / E2,
        tau_12 / G12,
    ])

    # ========================================================
    # Maximum Stress
    # ========================================================

    (
        max_stress_indices,
        max_stress_fi,
        max_stress_failed,
    ) = maximum_stress_failure(
        stress_local,
        strengths,
    )

    # ========================================================
    # Maximum Strain
    # ========================================================

    (
        max_strain_indices,
        max_strain_fi,
        max_strain_failed,
    ) = maximum_strain_failure(
        strain_local,
        strengths,
        E1,
        E2,
        G12,
    )

    # ========================================================
    # Tsai-Hill
    # ========================================================

    tsai_hill_fi, tsai_hill_failed = tsai_hill_failure(
        stress_local,
        strengths,
    )

    # ========================================================
    # Tsai-Wu
    # ========================================================

    tsai_wu_fi, tsai_wu_failed = tsai_wu_failure(
        stress_local,
        strengths,
    )

    # ========================================================
    # Hoffman
    # ========================================================

    hoffman_fi, hoffman_failed = hoffman_failure(
        stress_local,
        strengths,
    )

    # ========================================================
    # Hashin
    # ========================================================

    (
        hashin_indices,
        hashin_fi,
        hashin_mode,
        hashin_failed,
    ) = hashin_failure(
        stress_local,
        strengths,
    )

    # ========================================================
    # Puck
    # ========================================================

    if puck_parameters is None:
        puck_parameters = {
            "p_perp_parallel_t": 0.30,
            "p_perp_parallel_c": 0.25,
            "p_perp_perp_c": 0.25,
            "m_sigma_f": 1.10,
            "E1": E1,
            "Ef": 230e9,
            "nu12": nu12,
            "nuf": 0.20,
        }

    (
        puck_indices,
        puck_fi,
        puck_mode,
        puck_angle,
        puck_failed,
    ) = puck_failure(
        stress_local,
        strengths,
        puck_parameters,
    )

    # ========================================================
    # Unified result dictionary
    # ========================================================

    return {
        "Maximum Stress": {
            "failure_indices": max_stress_indices,
            "failure_index": float(max_stress_fi),
            "failed": bool(max_stress_failed),
        },

        "Maximum Strain": {
            "failure_indices": max_strain_indices,
            "failure_index": float(max_strain_fi),
            "failed": bool(max_strain_failed),
        },

        "Tsai-Hill": {
            "failure_index": float(tsai_hill_fi),
            "failed": bool(tsai_hill_failed),
        },

        "Tsai-Wu": {
            "failure_index": float(tsai_wu_fi),
            "failed": bool(tsai_wu_failed),
        },

        "Hoffman": {
            "failure_index": float(hoffman_fi),
            "failed": bool(hoffman_failed),
        },

        "Hashin": {
            "failure_indices": hashin_indices,
            "failure_index": float(hashin_fi),
            "failure_mode": hashin_mode,
            "failed": bool(hashin_failed),
        },

        "Puck": {
            "failure_indices": puck_indices,
            "failure_index": float(puck_fi),
            "failure_mode": puck_mode,
            "fracture_angle": float(puck_angle),
            "failed": bool(puck_failed),
        },
    }