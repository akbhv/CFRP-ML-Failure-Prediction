def calculate_lamina_properties(Ef, Em, Gf, Gm, nu_f, nu_m, Vf):
    """
    Calculate the engineering constants of a unidirectional composite lamina.

    Inputs:
        Ef   : Fibre Young's modulus (Pa)
        Em   : Matrix Young's modulus (Pa)
        Gf   : Fibre shear modulus (Pa)
        Gm   : Matrix shear modulus (Pa)
        nu_f : Fibre Poisson's ratio
        nu_m : Matrix Poisson's ratio
        Vf   : Fibre volume fraction

    Returns:
        E1
        E2
        G12
        nu12
    """

    Vm = 1.0 - Vf

    # Longitudinal Young's modulus
    E1 = Vf * Ef + Vm * Em

    # Transverse Young's modulus
    E2 = 1.0 / (Vf / Ef + Vm / Em)

    # Poisson's ratio
    nu12 = Vf * nu_f + Vm * nu_m

    # Shear modulus
    G12 = 1.0 / (Vf / Gf + Vm / Gm)

    return E1, E2, G12, nu12
