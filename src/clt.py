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