def maximum_stress_failure(stress_local, strengths):
    """
    Maximum Stress Failure Criterion.

    Parameters
    ----------
    stress_local : array-like
        Local lamina stresses [sigma_1, sigma_2, tau_12] in Pa.

    strengths : dict
        Strength values in Pa:
        Xt, Xc, Yt, Yc, S

    Returns
    -------
    failure_indices : dict
        Failure index for each stress component.

    max_failure_index : float
        Maximum failure index.

    failed : bool
        True if any failure index >= 1.
    """

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    if sigma_1 >= 0:
        fi_1 = sigma_1 / Xt
    else:
        fi_1 = abs(sigma_1) / Xc

    if sigma_2 >= 0:
        fi_2 = sigma_2 / Yt
    else:
        fi_2 = abs(sigma_2) / Yc

    fi_6 = abs(tau_12) / S

    failure_indices = {
        "FI_1": fi_1,
        "FI_2": fi_2,
        "FI_6": fi_6
    }

    max_failure_index = max(failure_indices.values())
    failed = max_failure_index >= 1.0

    return failure_indices, max_failure_index, failed