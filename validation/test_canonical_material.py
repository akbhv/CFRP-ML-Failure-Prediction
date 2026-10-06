from src.materials import get_fiber, get_matrix
from src.micromechanics import calculate_lamina_properties
from src.strengths import find_lamina_strengths


def test_canonical_t300_914c():
    # ---------------------------------------------------------
    # 1. Load canonical constituent materials
    # ---------------------------------------------------------
    fiber = get_fiber("T300")
    matrix = get_matrix("Standard Epoxy")

    assert fiber["E"] == 230e9
    assert fiber["G"] == 15e9
    assert fiber["nu"] == 0.20

    assert matrix["E"] == 4.0e9
    assert matrix["G"] == 1.481e9
    assert matrix["nu"] == 0.35

    # ---------------------------------------------------------
    # 2. Calculate canonical lamina properties
    # ---------------------------------------------------------
    E1, E2, G12, nu12 = calculate_lamina_properties(
        Ef=fiber["E"],
        Gf=fiber["G"],
        nu_f=fiber["nu"],
        Em=matrix["E"],
        Gm=matrix["G"],
        nu_m=matrix["nu"],
        Vf=0.60,
    )

    assert abs(E1 - 139.6e9) < 1e6
    assert abs(E2 - 9.745762711864407e9) < 1e6
    assert abs(G12 - 3.224893301977179e9) < 1e6
    assert abs(nu12 - 0.26) < 1e-9

    # ---------------------------------------------------------
    # 3. Resolve canonical lamina strength dataset
    # ---------------------------------------------------------
    strengths = find_lamina_strengths(
        "T300",
        "Standard Epoxy",
    )

    assert strengths is not None
    assert strengths["system_id"] == "T300_BSL914C"

    assert strengths["Xt"] == 1500e6
    assert strengths["Xc"] == 900e6
    assert strengths["Yt"] == 27e6
    assert strengths["Yc"] == 200e6
    assert strengths["S"] == 80e6


if __name__ == "__main__":
    test_canonical_t300_914c()
    print("CANONICAL T300/914C MATERIAL VALIDATION PASSED")
