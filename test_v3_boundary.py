
import numpy as np

from src.generate_dataset_final import (
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    find_hashin_failure_scale_v3,
)
from src.laminate_failure_comparison import find_governing_failure

# Test a pure Nx load direction in the scaled load space.
direction = np.array([
    100_000.0,
     50_000.0,
     50_000.0,
        100.0,
        100.0,
        100.0,
])

scale, result = find_hashin_failure_scale_v3(
    E1,
    E2,
    G12,
    nu12,
    strengths,
    ply_angles,
    ply_thickness,
    direction,
)

if scale is None:
    raise RuntimeError("No Hashin failure boundary found.")

governing = find_governing_failure(result, "Hashin")
fi = float(governing["criteria"]["Hashin"]["failure_index"])

print(f"Boundary scale: {scale:.8g}")
print(f"Hashin FI: {fi:.8f}")
print(f"Governing ply: {governing['ply']}")
print(f"Governing surface: {governing['surface']}")

assert abs(fi - 1.0) < 1e-3, f"Boundary FI is not close to 1: {fi}"
print("PASS: Hashin boundary test")
