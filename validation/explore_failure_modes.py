import numpy as np
import pandas as pd

from src.dataset_generator import (
    evaluate_laminate_case,
    find_hashin_failure_scale,
)


# ==================================================
# MATERIAL
# ==================================================

E1 = 139.4e9
E2 = 8.554729011689692e9
G12 = 3.051643192488263e9
nu12 = 0.26

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# ==================================================
# LAMINATE
# ==================================================

ply_angles = [0, 45, -45, 90, 90, -45, 45, 0]
ply_thickness = 0.125e-3


# ==================================================
# SETTINGS
# ==================================================

NUMBER_OF_DIRECTIONS = 5000
RANDOM_SEED = 123


# ==================================================
# MAIN
# ==================================================

rng = np.random.default_rng(RANDOM_SEED)

mode_counts = {
    "fiber_tension": 0,
    "fiber_compression": 0,
    "matrix_tension": 0,
    "matrix_compression": 0
}

failure_scales = []


print("-----------------------------------")
print("FAILURE MODE EXPLORATION")
print("-----------------------------------")


for i in range(NUMBER_OF_DIRECTIONS):

    # Random direction in 6D load space
    direction = rng.normal(size=6)
    direction /= np.linalg.norm(direction)

    # Find failure boundary
    failure_scale, result = find_hashin_failure_scale(
        E1,
        E2,
        G12,
        nu12,
        strengths,
        ply_angles,
        ply_thickness,
        direction
    )

    if failure_scale is None:
        continue

    mode = result["hashin_mode"]

    mode_counts[mode] += 1
    failure_scales.append(failure_scale)


# ==================================================
# RESULTS
# ==================================================

total = sum(mode_counts.values())

print()
print("-----------------------------------")
print("FAILURE MODE COUNTS")
print("-----------------------------------")

for mode, count in mode_counts.items():

    percentage = 100.0 * count / total

    print(
        f"{mode:20s} "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


print()
print("-----------------------------------")
print("TOTAL")
print("-----------------------------------")

print(f"Valid boundaries = {total}")
print(f"Failed searches = {NUMBER_OF_DIRECTIONS - total}")


print()
print("-----------------------------------")
print("FAILURE SCALE")
print("-----------------------------------")

failure_scales = np.array(failure_scales)

print(f"Minimum = {failure_scales.min():.6f}")
print(f"Maximum = {failure_scales.max():.6f}")
print(f"Mean    = {failure_scales.mean():.6f}")
print(f"Median  = {np.median(failure_scales):.6f}")