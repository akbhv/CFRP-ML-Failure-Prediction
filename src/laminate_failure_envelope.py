"""
First-ply-failure load-factor analysis for the laminate.

The applied laminate load vector is scaled by a load factor lambda:

    N(lambda) = lambda * N_base
    M(lambda) = lambda * M_base

For each failure criterion, the code determines the load factor
at which the maximum failure index reaches 1.0.
"""

import numpy as np

from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)


CRITERIA = [
    "Maximum Stress",
    "Maximum Strain",
    "Tsai-Hill",
    "Tsai-Wu",
    "Hoffman",
    "Hashin",
    "Puck",
]


def _evaluate_at_load_factor(
    load_factor,
    E1,
    E2,
    G12,
    nu12,
    ply_angles,
    ply_thickness,
    N_base,
    M_base,
    strengths,
    puck_parameters,
):
    """
    Evaluate laminate failure criteria at a particular load factor.
    """

    N = load_factor * np.asarray(N_base, dtype=float)
    M = load_factor * np.asarray(M_base, dtype=float)

    return evaluate_laminate_failure(
        E1=E1,
        E2=E2,
        G12=G12,
        nu12=nu12,
        ply_angles=ply_angles,
        ply_thickness=ply_thickness,
        N=N,
        M=M,
        strengths=strengths,
        puck_parameters=puck_parameters,
    )


def _maximum_failure_index(
    laminate_results,
    criterion,
):
    """
    Return the maximum failure index and its governing ply result.
    """

    governing = find_governing_failure(
        laminate_results,
        criterion,
    )

    failure_index = governing["criteria"][criterion]["failure_index"]

    return float(failure_index), governing


def find_first_ply_failure(
    E1,
    E2,
    G12,
    nu12,
    ply_angles,
    ply_thickness,
    N_base,
    M_base,
    strengths,
    puck_parameters=None,
    initial_upper_bound=1.0,
    growth_factor=2.0,
    tolerance=1e-6,
    max_iterations=100,
):
    """
    Determine first-ply-failure load factors for all criteria.

    The load factor lambda is defined relative to N_base and M_base.

    The algorithm first brackets failure by increasing lambda, then
    uses bisection to find the point where FI = 1.

    Returns
    -------
    dict
        One result dictionary per failure criterion.
    """

    N_base = np.asarray(N_base, dtype=float)
    M_base = np.asarray(M_base, dtype=float)

    results = {}

    for criterion in CRITERIA:

        lower = 0.0
        upper = float(initial_upper_bound)

        # ----------------------------------------------------
        # Find an upper bound where FI >= 1
        # ----------------------------------------------------

        for _ in range(max_iterations):

            laminate_results = _evaluate_at_load_factor(
                upper,
                E1,
                E2,
                G12,
                nu12,
                ply_angles,
                ply_thickness,
                N_base,
                M_base,
                strengths,
                puck_parameters,
            )

            failure_index, governing = _maximum_failure_index(
                laminate_results,
                criterion,
            )

            if failure_index >= 1.0:
                break

            upper *= growth_factor

        else:
            raise RuntimeError(
                f"Could not bracket first-ply failure for "
                f"{criterion}."
            )

        # ----------------------------------------------------
        # Bisection
        # ----------------------------------------------------

        for _ in range(max_iterations):

            midpoint = 0.5 * (lower + upper)

            laminate_results = _evaluate_at_load_factor(
                midpoint,
                E1,
                E2,
                G12,
                nu12,
                ply_angles,
                ply_thickness,
                N_base,
                M_base,
                strengths,
                puck_parameters,
            )

            failure_index, _ = _maximum_failure_index(
                laminate_results,
                criterion,
            )

            if failure_index >= 1.0:
                upper = midpoint
            else:
                lower = midpoint

            if upper - lower <= tolerance:
                break

        critical_load_factor = 0.5 * (lower + upper)

        # ----------------------------------------------------
        # Evaluate final critical state
        # ----------------------------------------------------

        final_results = _evaluate_at_load_factor(
            critical_load_factor,
            E1,
            E2,
            G12,
            nu12,
            ply_angles,
            ply_thickness,
            N_base,
            M_base,
            strengths,
            puck_parameters,
        )

        final_fi, final_governing = _maximum_failure_index(
            final_results,
            criterion,
        )

        criterion_result = final_governing["criteria"][criterion]

        results[criterion] = {
            "critical_load_factor": critical_load_factor,
            "failure_index": final_fi,
            "ply": final_governing["ply"],
            "angle": final_governing["angle"],
            "surface": final_governing["surface"],
            "failed": criterion_result["failed"],
        }

        if "failure_mode" in criterion_result:
            results[criterion]["failure_mode"] = (
                criterion_result["failure_mode"]
            )

    return results