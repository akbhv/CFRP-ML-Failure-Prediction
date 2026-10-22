
import numpy as np

from src.failure_criteria import (
    hoffman_failure,
    puck_failure,
    PUCK_CFRP_PARAMETERS,
)

strengths = {
    "Xt": 1.5e9,
    "Xc": 0.9e9,
    "Yt": 27e6,
    "Yc": 200e6,
    "S": 80e6,
}

print("=== Hoffman threshold test ===")

for fi in [0.999, 0.99999, 0.999999, 1.0, 1.000001]:
    # Pure longitudinal tension gives a controllable Hoffman FI.
    stress = [fi * strengths["Xt"], 0.0, 0.0]
    actual_fi, failed = hoffman_failure(stress, strengths)
    print(
        f"Requested FI={fi:.6f}, "
        f"actual FI={actual_fi:.9f}, failed={failed}"
    )

print("\n=== Puck fiber-tension edge case ===")

params = dict(PUCK_CFRP_PARAMETERS)

for sigma_2 in [0.0, 10e6, 27e6, 50e6]:
    stress = [1e6, sigma_2, 0.0]
    indices, max_fi, mode, angle, failed = puck_failure(
        stress, strengths, params
    )
    fiber_key = "fiber_tension"
    print(
        f"sigma1=1 MPa, sigma2={sigma_2/1e6:.1f} MPa: "
        f"fiber FI={indices.get(fiber_key)}, "
        f"governing FI={max_fi:.6f}, mode={mode}"
    )
