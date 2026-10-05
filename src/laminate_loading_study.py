import numpy as np
import pandas as pd

from src.laminate_failure_envelope import find_first_ply_failure


DEFAULT_LOADING_CASES = {
    "Longitudinal tension": np.array([100e3, 0.0, 0.0]),
    "Transverse tension": np.array([0.0, 100e3, 0.0]),
    "In-plane shear": np.array([0.0, 0.0, 100e3]),
    "Biaxial tension": np.array([100e3, 100e3, 0.0]),
    "Tension-compression": np.array([100e3, -100e3, 0.0]),
    "Combined loading": np.array([100e3, 50e3, 20e3]),
}


def run_loading_study(
    ply_angles,
    ply_thickness,
    strengths,
    E1,
    E2,
    G12,
    nu12,
    loading_cases=None,
    M_base=None,
):
    """
    Run first-ply-failure analysis for multiple membrane
    loading paths.

    Parameters
    ----------

    ply_angles : list
        Ply orientation angles in degrees.

    ply_thickness : float
        Thickness of each ply (m).

    strengths : dict
        Lamina strength values.

    E1, E2, G12, nu12 : float
        Lamina engineering properties.

    loading_cases : dict, optional
        Dictionary mapping case names to membrane load vectors
        [Nx, Ny, Nxy] in N/m.

    M_base : array-like, optional
        Moment resultants [Mx, My, Mxy] in N.

    Returns
    -------
    results : pandas.DataFrame
        First-ply-failure results for every loading case
        and failure criterion.
    """

    if loading_cases is None:
        loading_cases = DEFAULT_LOADING_CASES

    if M_base is None:
        M_base = np.zeros(3)

    M_base = np.asarray(M_base, dtype=float)

    records = []

    for case_name, N_base in loading_cases.items():

        N_base = np.asarray(N_base, dtype=float)

        failure_results = find_first_ply_failure(

         E1=E1,
         E2=E2,
         G12=G12,
         nu12=nu12,
         ply_angles=ply_angles,
         ply_thickness=ply_thickness,
         N_base=N_base,
         M_base=M_base,
         strengths=strengths,
        )

        for criterion, result in failure_results.items():

            critical_lambda = result["critical_load_factor"]

            critical_load = critical_lambda * N_base

            record = {
                "loading_case": case_name,
                "criterion": criterion,
                "base_Nx_N_per_m": N_base[0],
                "base_Ny_N_per_m": N_base[1],
                "base_Nxy_N_per_m": N_base[2],
                "critical_load_factor": critical_lambda,
                "critical_Nx_N_per_m": critical_load[0],
                "critical_Ny_N_per_m": critical_load[1],
                "critical_Nxy_N_per_m": critical_load[2],
                "failure_index": result["failure_index"],
                "ply": result["ply"],
                "angle_deg": result["angle"],
                "surface": result["surface"],
                "failure_mode": result.get("failure_mode"),
            }

            records.append(record)

    return pd.DataFrame(records)