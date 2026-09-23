import numpy as np

from src.failure_criteria import maximum_stress_failure


strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


stress = np.array([
    500e6,   # sigma_1
    20e6,    # sigma_2
    30e6     # tau_12
])


failure_indices, max_fi, failed = maximum_stress_failure(
    stress,
    strengths
)


print("Failure indices:")
print(failure_indices)

print("Maximum failure index:", max_fi)
print("Failed:", failed)