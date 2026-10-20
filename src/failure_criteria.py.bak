import numpy as np

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

def maximum_strain_failure(
    strain_local,
    strengths,
    E1,
    E2,
    G12
):
    """
    Maximum Strain failure criterion.

    Parameters
    ----------
    strain_local : array-like
        Local lamina strains [epsilon_1, epsilon_2, gamma_12].
    strengths : dict
        Lamina strengths in Pa.
    E1, E2, G12 : float
        Lamina elastic properties in Pa.

    Returns
    -------
    failure_indices : dict
        Failure indices for each strain component.
    max_failure_index : float
        Governing failure index.
    failed : bool
        True if any failure index >= 1.
    """

    epsilon_1, epsilon_2, gamma_12 = strain_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    epsilon_1_t = Xt / E1
    epsilon_1_c = Xc / E1
    epsilon_2_t = Yt / E2
    epsilon_2_c = Yc / E2
    gamma_12_limit = S / G12

    if epsilon_1 >= 0:
        fi_1 = epsilon_1 / epsilon_1_t
    else:
        fi_1 = abs(epsilon_1) / epsilon_1_c

    if epsilon_2 >= 0:
        fi_2 = epsilon_2 / epsilon_2_t
    else:
        fi_2 = abs(epsilon_2) / epsilon_2_c

    fi_12 = abs(gamma_12) / gamma_12_limit

    failure_indices = {
        "epsilon_1": fi_1,
        "epsilon_2": fi_2,
        "gamma_12": fi_12,
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

def hoffman_failure(stress_local, strengths):
    """
    Hoffman failure criterion for an orthotropic lamina under plane stress.

    Parameters
    ----------
    stress_local : array-like
        Local lamina stresses [sigma_1, sigma_2, tau_12] in Pa.
    strengths : dict
        Lamina strengths in Pa:
        Xt, Xc, Yt, Yc, S.

    Returns
    -------
    failure_index : float
        Hoffman failure index.
    failed : bool
        True if failure_index >= 1.
    """

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    failure_index = (
        (sigma_1 ** 2) / (Xt * Xc)
        + (sigma_2 ** 2) / (Yt * Yc)
        + (sigma_1 * sigma_2) / (Xt * Xc)
        + (1.0 / Xt - 1.0 / Xc) * sigma_1
        + (1.0 / Yt - 1.0 / Yc) * sigma_2
        + (tau_12 ** 2) / (S ** 2)
    )

    failed = np.isclose(failure_index, 1.0, atol=1e-12) or failure_index > 1.0

    return failure_index, failed

def hashin_failure(stress_local, strengths):
    """
    Hashin Failure Criterion for a plane-stress UD lamina.

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
        Failure index for each Hashin failure mode.

    max_failure_index : float
        Maximum Hashin failure index.

    failure_mode : str
        Governing failure mode.

    failed : bool
        True if any failure index >= 1.
    """

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    # Initialize all failure indices
    FI_ft = 0.0
    FI_fc = 0.0
    FI_mt = 0.0
    FI_mc = 0.0

    # Fiber failure
    if sigma_1 >= 0:
        FI_ft = (
            (sigma_1 / Xt) ** 2
            + (tau_12 / S) ** 2
        )
    else:
        FI_fc = (
            (sigma_1 / Xc) ** 2
        )

    # Matrix failure
    if sigma_2 >= 0:
        FI_mt = (
            (sigma_2 / Yt) ** 2
            + (tau_12 / S) ** 2
        )
    else:
        FI_mc = (
            (sigma_2 / (2.0 * S)) ** 2
            + (
                (Yc / (2.0 * S)) ** 2 - 1.0
            ) * (sigma_2 / Yc)
            + (tau_12 / S) ** 2
        )

    failure_indices = {
        "fiber_tension": FI_ft,
        "fiber_compression": FI_fc,
        "matrix_tension": FI_mt,
        "matrix_compression": FI_mc
    }

    failure_mode = max(
        failure_indices,
        key=failure_indices.get
    )

    max_failure_index = failure_indices[failure_mode]

    failed = max_failure_index >= 1.0

    return (
        failure_indices,
        max_failure_index,
        failure_mode,
        failed
    )

def _puck_fracture_plane_stress(sigma_2, tau_12, angle):
    """
    Transform transverse/shear stresses to a candidate
    Puck fracture plane.

    Parameters
    ----------
    sigma_2 : float
        Transverse normal stress [Pa].
    tau_12 : float
        In-plane shear stress [Pa].
    angle : float
        Fracture-plane angle [rad].

    Returns
    -------
    sigma_n : float
        Normal stress on the candidate fracture plane [Pa].
    tau_nt : float
        Shear stress acting on the candidate plane [Pa].
    """



    c = np.cos(angle)
    s = np.sin(angle)

    sigma_n = (
        sigma_2 * c**2
        + 2.0 * tau_12 * s * c
    )

    tau_nt = (
        -sigma_2 * s * c
        + tau_12 * (c**2 - s**2)
    )

    return sigma_n, tau_nt

# ============================================================
# Puck Failure Criterion
# ============================================================

PUCK_CFRP_PARAMETERS = {
    # Inclination parameters
    "p_perp_parallel_t": 0.30,
    "p_perp_parallel_c": 0.25,
    "p_perp_perp_c": 0.25,

    # Fiber-failure parameters
    "m_sigma_f": 1.10,

    # Fiber properties required by Puck FF
    "E1": 139.4e9,
    "Ef": 230e9,
    "nu12": 0.26,
    "nuf": 0.20,
}


def puck_failure(stress_local, strengths, parameters=None):
    """
    Puck failure criterion for plane-stress UD composites.

    Returns:
        failure_indices : dict
        max_failure_index : float
        failure_mode : str
        fracture_angle : float
        failed : bool

    Modes:
        fiber_tension
        fiber_compression
        IFF_A_tension
        IFF_B_compression_shear
        IFF_C_compression
    """

    if parameters is None:
        parameters = PUCK_CFRP_PARAMETERS

    sigma_1, sigma_2, tau_12 = stress_local

    Xt = strengths["Xt"]
    Xc = strengths["Xc"]
    Yt = strengths["Yt"]
    Yc = strengths["Yc"]
    S = strengths["S"]

    p_plus = parameters["p_perp_parallel_t"]
    p_minus = parameters["p_perp_parallel_c"]
    p_perp_perp_c = parameters["p_perp_perp_c"]

    # --------------------------------------------------------
    # 1. Fiber failure
    # --------------------------------------------------------

    E1 = parameters["E1"]
    Ef = parameters["Ef"]
    nu12 = parameters["nu12"]
    nuf = parameters["nuf"]
    m_sigma_f = parameters["m_sigma_f"]

    # Effective longitudinal stress according to Puck
    if sigma_1 >= 0.0:

        effective_sigma_1 = (
            sigma_1
            - (nu12 - (E1 / Ef) * nuf) * sigma_2
        )

        fi_fiber = effective_sigma_1 / Xt
        fiber_mode = "fiber_tension"

    else:

        effective_sigma_1 = (
            sigma_1
            - (
                nu12
                - (E1 / Ef) * nuf * m_sigma_f
            ) * sigma_2
        )

        fi_fiber = abs(effective_sigma_1) / Xc
        fiber_mode = "fiber_compression"

    # --------------------------------------------------------
    # 2. Inter-fiber failure
    # --------------------------------------------------------

    # Mode A: transverse tension
    #
    # sqrt[
    #   (tau12/S)^2
    #   +
    #   (1 - p+*Yt/S)^2 * (sigma2/Yt)^2
    # ]
    # + p+*sigma2/S

    sigma_2_tolerance = 1e-10 * max(Yc, S, 1.0)

    if sigma_2 >= -sigma_2_tolerance:

        fi_iff = np.sqrt(
            (tau_12 / S) ** 2
            +
            (
                1.0
                - p_plus * Yt / S
            ) ** 2
            *
            (sigma_2 / Yt) ** 2
        ) + p_plus * sigma_2 / S

        iff_mode = "IFF_A_tension"
        fracture_angle = 0.0

    else:
        sigma_2_compression = min(sigma_2, -sigma_2_tolerance)

        # ----------------------------------------------------
        # Mode B
        #
        # [ sqrt(tau^2 + (p- sigma2)^2)
        #   + p- sigma2 ] / S
        # ----------------------------------------------------

        fi_mode_b = (
            np.sqrt(
                tau_12 ** 2
                +
                (p_minus * sigma_2) ** 2
            )
            + p_minus * sigma_2
        ) / S

        # ----------------------------------------------------
        # Mode C
        #
        # [
        #   (tau12 /
        #    [2(1+p--)*S])^2
        #   +
        #   (sigma2/Yc)^2
        # ]
        # * (-Yc/sigma2)
        # ----------------------------------------------------

        fi_mode_c = (
            (
                (
                    tau_12
                    /
                    (
                        2.0
                        * (1.0 + p_perp_perp_c)
                        * S
                    )
                ) ** 2
                +
                (sigma_2_compression / Yc) ** 2
            )
            *
            (-Yc / sigma_2_compression)
        )

        # Puck distinguishes the compression/shear
        # regimes using the governing failure mode.
        if fi_mode_b >= fi_mode_c:

            fi_iff = fi_mode_b
            iff_mode = "IFF_B_compression_shear"
            fracture_angle = 0.0

        else:

            fi_iff = fi_mode_c
            iff_mode = "IFF_C_compression"
            fracture_angle = np.degrees(
                np.arctan2(
                    abs(tau_12),
                    abs(sigma_2)
                )
            )

    failure_indices = {
        fiber_mode: float(fi_fiber),
        iff_mode: float(fi_iff),
    }

    if fi_fiber >= fi_iff:
        max_failure_index = float(fi_fiber)
        failure_mode = fiber_mode
    else:
        max_failure_index = float(fi_iff)
        failure_mode = iff_mode

    failed = max_failure_index >= 1.0

    return (
        failure_indices,
        max_failure_index,
        failure_mode,
        fracture_angle,
        failed,
    )