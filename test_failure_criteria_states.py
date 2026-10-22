
import numpy as np

from src.generate_dataset_final import strengths
from src.failure_criteria import (
    maximum_stress_failure,
    maximum_strain_failure,
    tsai_hill_failure,
    tsai_wu_failure,
    hoffman_failure,
    hashin_failure,
    puck_failure,
)
from src.micromechanics import calculate_lamina_properties
from src.materials import get_fiber, get_matrix

fiber = get_fiber("T300")
matrix = get_matrix("Standard Epoxy")

E1, E2, G12, nu12 = calculate_lamina_properties(
    fiber["E"], matrix["E"],
    fiber["G"], matrix["G"],
    fiber["nu"], matrix["nu"], 0.60
)

puck_parameters = {
    "p_perp_parallel_t": 0.30,
    "p_perp_parallel_c": 0.25,
    "p_perp_perp_c": 0.25,
    "m_sigma_f": 1.10,
    "E1": E1,
    "Ef": 230e9,
    "nu12": nu12,
    "nuf": 0.20,
}

stress_states = {
    "longitudinal tension": [0.8 * strengths["Xt"], 0, 0],
    "longitudinal compression": [-0.8 * strengths["Xc"], 0, 0],
    "transverse tension": [0, 0.8 * strengths["Yt"], 0],
    "transverse compression": [0, -0.8 * strengths["Yc"], 0],
    "pure shear": [0, 0, 0.8 * strengths["S"]],
    "compression + shear": [
        0, -0.5 * strengths["Yc"], 0.5 * strengths["S"]
    ],
}

print("Material strengths (Pa):", strengths)
print("\nFailure-criterion smoke test")

for name, state in stress_states.items():
    s = np.asarray(state, dtype=float)

    ms = maximum_stress_failure(s, strengths)[1]
    eps = np.array([
        (s[0] - nu12 * s[1]) / E1,
        (s[1] - (nu12 * E2 / E1) * s[0]) / E2,
        s[2] / G12,
    ])
    me = maximum_strain_failure(eps, strengths, E1, E2, G12)[1]
    th = tsai_hill_failure(s, strengths)[0]
    tw = tsai_wu_failure(s, strengths)[0]
    ho = hoffman_failure(s, strengths)[0]
    ha = hashin_failure(s, strengths)
    pu = puck_failure(s, strengths, puck_parameters)

    values = [ms, me, th, tw, ho, ha[1], pu[1]]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite result for {name}: {values}")

    print(f"\n{name}")
    print("  FI [MaxStress, MaxStrain, TsaiHill, TsaiWu, Hoffman, Hashin, Puck]")
    print(" ", [round(float(x), 5) for x in values])
    print(f"  Hashin mode: {ha[2]}")
    print(f"  Puck mode:   {pu[2]}")

print("\nPASS: all six stress states produced finite results.")
