import numpy as np


def transform_Q(Q, theta):
    """
    Transform the reduced stiffness matrix [Q]
    from material coordinates to global coordinates.

    Parameters
    ----------
    Q : numpy.ndarray
        3x3 reduced stiffness matrix in Pa.

    theta : float
        Ply orientation angle in degrees.

    Returns
    -------
    Qbar : numpy.ndarray
        3x3 transformed reduced stiffness matrix in Pa.
    """

    # Convert angle from degrees to radians
    theta_rad = np.radians(theta)

    m = np.cos(theta_rad)
    n = np.sin(theta_rad)

    # Extract Q components
    Q11 = Q[0, 0]
    Q12 = Q[0, 1]
    Q22 = Q[1, 1]
    Q66 = Q[2, 2]

    # Transformed stiffness terms
    Qbar11 = (
        Q11 * m**4
        + 2 * (Q12 + 2 * Q66) * m**2 * n**2
        + Q22 * n**4
    )

    Qbar22 = (
        Q11 * n**4
        + 2 * (Q12 + 2 * Q66) * m**2 * n**2
        + Q22 * m**4
    )

    Qbar12 = (
        (Q11 + Q22 - 4 * Q66) * m**2 * n**2
        + Q12 * (m**4 + n**4)
    )

    Qbar16 = (
        (Q11 - Q12 - 2 * Q66) * m**3 * n
        - (Q22 - Q12 - 2 * Q66) * m * n**3
    )

    Qbar26 = (
        (Q11 - Q12 - 2 * Q66) * m * n**3
        - (Q22 - Q12 - 2 * Q66) * m**3 * n
    )

    Qbar66 = (
        (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m**2 * n**2
        + Q66 * (m**4 + n**4)
    )

    Qbar = np.array([
        [Qbar11, Qbar12, Qbar16],
        [Qbar12, Qbar22, Qbar26],
        [Qbar16, Qbar26, Qbar66]
    ])

    return Qbar