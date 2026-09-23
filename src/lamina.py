import numpy as np


def calculate_Q(E1, E2, G12, nu12):
    """
    Calculate the reduced stiffness matrix [Q]
    for an orthotropic lamina under plane stress.

    Inputs:
        E1   : Longitudinal Young's modulus (Pa)
        E2   : Transverse Young's modulus (Pa)
        G12  : In-plane shear modulus (Pa)
        nu12 : Major Poisson's ratio

    Returns:
        Q : 3x3 reduced stiffness matrix (Pa)
    """

    # Minor Poisson's ratio
    nu21 = nu12 * E2 / E1

    # Denominator
    delta = 1.0 - nu12 * nu21

    # Reduced stiffness coefficients
    Q11 = E1 / delta
    Q22 = E2 / delta
    Q12 = nu12 * E2 / delta
    Q66 = G12

    Q = np.array([
        [Q11, Q12, 0.0],
        [Q12, Q22, 0.0],
        [0.0, 0.0, Q66]
    ])

    return Q