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

def tsai_hill_failure(stress_local, strengths):
    """
    Tsai-Hill Failure Criterion.

    Parameters
    ----------
    stress_local : array-like
        Local lamina stresses [sigma_1, sigma_2, tau_12] in Pa.

    strengths : dict
        Strength values in Pa:
        Xt, Xc, Yt, Yc, S

    Returns
    -------
    failure_index : float
        Tsai-Hill failure index.

    failed : bool
        True if failure index >= 1.
    """

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    # Select tensile or compressive strength
    # according to the sign of the corresponding stress.
    if sigma_1 >= 0:
        X = Xt
    else:
        X = Xc

    if sigma_2 >= 0:
        Y = Yt
    else:
        Y = Yc

    failure_index = (
        (sigma_1 / X) ** 2
        - (sigma_1 * sigma_2) / (X ** 2)
        + (sigma_2 / Y) ** 2
        + (tau_12 / S) ** 2
    )

    failed = failure_index >= 1.0

    return failure_index, failed

def tsai_wu_failure(stress_local, strengths, F12=0.0):
    """
    Tsai-Wu Failure Criterion for a plane-stress lamina.

    Parameters
    ----------
    stress_local : array-like
        Local lamina stresses [sigma_1, sigma_2, tau_12] in Pa.

    strengths : dict
        Strength values in Pa:
        Xt, Xc, Yt, Yc, S

    Returns
    -------
    failure_index : float
        Tsai-Wu failure index.

    failed : bool
        True if failure index >= 1.
    """

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    # First-order strength coefficients
    F1 = (1.0 / Xt) - (1.0 / Xc)
    F2 = (1.0 / Yt) - (1.0 / Yc)

    # Second-order coefficients
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S ** 2)

    # Interaction coefficient.
    # For now, use the commonly adopted assumption F12 = 0.

    failure_index = (
        F1 * sigma_1
        + F2 * sigma_2
        + F11 * sigma_1**2
        + F22 * sigma_2**2
        + F66 * tau_12**2
        + 2.0 * F12 * sigma_1 * sigma_2
    )

    failed = failure_index >= 1.0

    return failure_index, failed