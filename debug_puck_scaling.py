import numpy as np

from src.laminate_failure_comparison import (
    evaluate_laminate_failure,
    find_governing_failure,
)

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}

angles = [0, 45, -45, 90, 90, -45, 45, 0]

N_base = np.array([0.0, 0.0, 100e3])
M_base = np.zeros(3)

print("lambda | Puck FI | mode | ply")
print("-" * 60)

for lam in [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0]:

    results = evaluate_laminate_failure(
        E1=139.4e9,
        E2=8.554729e9,
        G12=3.051643e9,
        nu12=0.26,
        ply_angles=angles,
        ply_thickness=0.125e-3,
        N=lam * N_base,
        M=lam * M_base,
        strengths=strengths,
    )

    governing = find_governing_failure(
        results,
        "Puck",
    )

    puck = governing["criteria"]["Puck"]

    print(
        f"{lam:.1e} | "
        f"{puck['failure_index']:.12g} | "
        f"{puck.get('failure_mode')} | "
        f"{governing['ply']}"
    )
