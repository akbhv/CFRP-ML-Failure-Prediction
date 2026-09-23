import numpy as np

from src.failure_criteria import tsai_hill_failure


strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


stress = np.array([
    500e6,
    20e6,
    30e6
])


failure_index, failed = tsai_hill_failure(
    stress,
    strengths
)


print("Tsai-Hill failure index:", failure_index)
print("Failed:", failed)