import numpy as np

from src.transformation import transform_Q


def calculate_ABD(Q, ply_angles, ply_thickness):
    """
    Calculate the A, B and D stiffness matrices
    for a laminate using Classical Lamination Theory.

    Parameters
    ----------
    Q : numpy.ndarray
        3x3 reduced stiffness matrix of the lamina (Pa).

    ply_angles : list
        Ply orientation angles in degrees.

    ply_thickness : float
        Thickness of each ply (m).

    Returns
    -------
    A : numpy.ndarray
        Extensional stiffness matrix.

    B : numpy.ndarray
        Extension-bending coupling matrix.

    D : numpy.ndarray
        Bending stiffness matrix.

    z : numpy.ndarray
        Through-thickness coordinates of ply interfaces.
    """

    number_of_plies = len(ply_angles)

    # Total laminate thickness
    total_thickness = number_of_plies * ply_thickness

    # z-coordinates of ply interfaces
    z = np.linspace(
        -total_thickness / 2,
        total_thickness / 2,
        number_of_plies + 1
    )

    # Initialize A, B and D
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    # Loop through each ply
    for k, angle in enumerate(ply_angles):

        # Transformed stiffness matrix
        Qbar = transform_Q(Q, angle)

        z_bottom = z[k]
        z_top = z[k + 1]

        # A matrix
        A += Qbar * (z_top - z_bottom)

        # B matrix
        B += 0.5 * Qbar * (z_top**2 - z_bottom**2)

        # D matrix
        D += (1.0 / 3.0) * Qbar * (z_top**3 - z_bottom**3)

    return A, B, D, z

def solve_laminate_response(A, B, D, N, M):
    """
    Solve the Classical Lamination Theory equations
    for mid-plane strains and curvatures.

    Parameters
    ----------
    A : numpy.ndarray
        3x3 extensional stiffness matrix (N/m).

    B : numpy.ndarray
        3x3 extension-bending coupling matrix (N).

    D : numpy.ndarray
        3x3 bending stiffness matrix (N*m).

    N : array-like
        In-plane force resultants [Nx, Ny, Nxy] (N/m).

    M : array-like
        Moment resultants [Mx, My, Mxy] (N).

    Returns
    -------
    mid_plane_strain : numpy.ndarray
        [epsilon_x, epsilon_y, gamma_xy]

    curvature : numpy.ndarray
        [kappa_x, kappa_y, kappa_xy]
    """

    # Assemble the 6x6 ABD matrix
    ABD = np.block([
        [A, B],
        [B, D]
    ])

    # Assemble the loading vector
    load_vector = np.concatenate([
        np.asarray(N, dtype=float),
        np.asarray(M, dtype=float)
    ])

    # Solve ABD * response = load
    response = np.linalg.solve(ABD, load_vector)

    # First three values = mid-plane strains
    mid_plane_strain = response[:3]

    # Last three values = curvatures
    curvature = response[3:]

    return mid_plane_strain, curvature

def calculate_ply_strains(mid_plane_strain, curvature, z):
    """
    Calculate global strains at the top and bottom
    surfaces of every ply.

    Parameters
    ----------
    mid_plane_strain : numpy.ndarray
        [epsilon_x, epsilon_y, gamma_xy]

    curvature : numpy.ndarray
        [kappa_x, kappa_y, kappa_xy]

    z : numpy.ndarray
        Through-thickness coordinates of ply interfaces (m).

    Returns
    -------
    ply_strains : list
        Global strain values at bottom and top of each ply.
    """

    ply_strains = []

    for k in range(len(z) - 1):

        z_bottom = z[k]
        z_top = z[k + 1]

        strain_bottom = (
            mid_plane_strain + z_bottom * curvature
        )

        strain_top = (
            mid_plane_strain + z_top * curvature
        )

        ply_strains.append({
            "ply": k + 1,
            "bottom": strain_bottom,
            "top": strain_top
        })

    return ply_strains

def global_to_local_strain(strain_global, theta):
    """
    Transform engineering strains from global x-y
    coordinates to local 1-2 material coordinates.

    Parameters
    ----------
    strain_global : array-like
        [epsilon_x, epsilon_y, gamma_xy]

    theta : float
        Ply angle in degrees.

    Returns
    -------
    strain_local : numpy.ndarray
        [epsilon_1, epsilon_2, gamma_12]
    """

    theta_rad = np.radians(theta)

    m = np.cos(theta_rad)
    n = np.sin(theta_rad)

    epsilon_x = strain_global[0]
    epsilon_y = strain_global[1]
    gamma_xy = strain_global[2]

    epsilon_1 = (
        m**2 * epsilon_x
        + n**2 * epsilon_y
        + m * n * gamma_xy
    )

    epsilon_2 = (
        n**2 * epsilon_x
        + m**2 * epsilon_y
        - m * n * gamma_xy
    )

    gamma_12 = (
        -2 * m * n * epsilon_x
        + 2 * m * n * epsilon_y
        + (m**2 - n**2) * gamma_xy
    )

    return np.array([
        epsilon_1,
        epsilon_2,
        gamma_12
    ])

def local_stress_from_strain(Q, strain_local):
    """
    Calculate local lamina stresses from local strains.

    Parameters
    ----------
    Q : numpy.ndarray
        Reduced stiffness matrix (Pa).

    strain_local : numpy.ndarray
        [epsilon_1, epsilon_2, gamma_12]

    Returns
    -------
    stress_local : numpy.ndarray
        [sigma_1, sigma_2, tau_12] (Pa)
    """

    return Q @ strain_local