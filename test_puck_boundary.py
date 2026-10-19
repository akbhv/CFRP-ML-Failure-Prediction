
import numpy as np

from src.failure_criteria import (
    puck_failure,
    PUCK_CFRP_PARAMETERS,
)

strengths = {
    "Xt": 1.5e9,
    "Xc": 9.0e8,
    "Yt": 27e6,
    "Yc": 200e6,
    "S": 80e6,
}

p = PUCK_CFRP_PARAMETERS
S = strengths["S"]

ratio_boundary = (
    S * p["p_perp_perp_c"] / p["p_perp_parallel_c"]
) / (
    S * np.sqrt(1 + 2 * p["p_perp_perp_c"])
)

tau = 10e6

print(f"Theoretical transition ratio: {ratio_boundary:.8f}")

for multiplier in [0.999, 1.0, 1.001]:
    sigma_2 = -ratio_boundary * multiplier * tau
    stress = np.array([-20e6, sigma_2, tau])

    indices, maximum, mode, angle, failed = puck_failure(
        stress, strengths, p
    )

    print(
        f"multiplier={multiplier:.3f} | "
        f"ratio={abs(sigma_2 / tau):.8f} | "
        f"mode={mode} | "
        f"FI={maximum:.8f} | "
        f"finite={np.isfinite(maximum)}"
    )
    print("  Mode-specific indices:", indices)
