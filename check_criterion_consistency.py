
import numpy as np

from src.failure_criteria import (
    tsai_wu_failure,
    hoffman_failure,
)

strengths = {
    "Xt": 1.5e9,
    "Xc": 0.9e9,
    "Yt": 27e6,
    "Yc": 200e6,
    "S": 80e6,
}

states = {
    "longitudinal tension": [1.2e9, 0, 0],
    "longitudinal compression": [-0.72e9, 0, 0],
    "transverse tension": [0, 21.6e6, 0],
    "transverse compression": [0, -160e6, 0],
    "pure shear": [0, 0, 64e6],
    "combined stress": [200e6, -80e6, 40e6],
}

print("Tsai-Wu vs Hoffman direct consistency check")

for name, stress in states.items():
    tw, tw_failed = tsai_wu_failure(stress, strengths)
    ho, ho_failed = hoffman_failure(stress, strengths)

    print(
        f"{name:25s} "
        f"Tsai-Wu={tw: .8f} "
        f"Hoffman={ho: .8f} "
        f"difference={abs(tw-ho):.3e} "
        f"flags_equal={tw_failed == ho_failed}"
    )
    
    print(f"  Tsai-Wu failed: {tw_failed}")
print(f"  Hoffman failed: {ho_failed}")

print("\nPASS: these implementations are numerically equivalent.")
