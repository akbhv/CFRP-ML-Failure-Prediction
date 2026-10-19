
import numpy as np

from src.failure_criteria import puck_failure, PUCK_CFRP_PARAMETERS

strengths = {
    "Xt": 1.5e9,
    "Xc": 9.0e8,
    "Yt": 27e6,
    "Yc": 200e6,
    "S": 80e6,
}

stress_cases = [
    (
        "Original extreme sample",
        [-18.8813165e6, -1.80965577, 12.5944446e6],
    ),
    (
        "Near-zero transverse compression",
        [-18.8813165e6, -1.0, 12.5944446e6],
    ),
    (
        "Moderate transverse compression",
        [-18.8813165e6, -20e6, 12.5944446e6],
    ),
    (
        "High transverse compression",
        [-18.8813165e6, -160e6, 12.5944446e6],
    ),
    (
        "Zero transverse stress",
        [-18.8813165e6, 0.0, 12.5944446e6],
    ),
    (
        "Zero shear and compression",
        [-18.8813165e6, -20e6, 0.0],
    ),
]

for label, stress in stress_cases:
    (
        failure_indices,
        max_failure_index,
        failure_mode,
        fracture_angle,
        failed,
    ) = puck_failure(
        np.array(stress, dtype=float),
        strengths,
        PUCK_CFRP_PARAMETERS,
    )

    print(f"\n{label}")
    print("Stress [MPa]:", np.array(stress) / 1e6)
    print("Failure indices:", failure_indices)
    print("Maximum failure index:", max_failure_index)
    print("Failure mode:", failure_mode)
    print("Fracture angle:", fracture_angle)
    print("Failed:", failed)
    print("Finite FI:", np.isfinite(max_failure_index))
