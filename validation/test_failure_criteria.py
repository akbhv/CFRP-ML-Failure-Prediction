import numpy as np

from src.failure_criteria import (
    maximum_stress_failure,
    tsai_hill_failure,
    tsai_wu_failure,
    hashin_failure,
)


strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# Safe test stress state
stress = np.array([
    500e6,   # sigma_1
    20e6,    # sigma_2
    30e6     # tau_12
])


print("=" * 60)
print("FAILURE CRITERIA VALIDATION")
print("=" * 60)


# ------------------------------------------------------------
# Maximum Stress
# ------------------------------------------------------------

fi_ms, max_fi_ms, failed_ms = maximum_stress_failure(
    stress,
    strengths
)

print("\nMaximum Stress")
print("-" * 60)
print("Failure indices:", fi_ms)
print("Maximum FI:", max_fi_ms)
print("Failed:", failed_ms)

assert max_fi_ms < 1.0
assert not failed_ms


# ------------------------------------------------------------
# Tsai-Hill
# ------------------------------------------------------------

fi_th, failed_th = tsai_hill_failure(
    stress,
    strengths
)

print("\nTsai-Hill")
print("-" * 60)
print("Failure index:", fi_th)
print("Failed:", failed_th)

assert fi_th < 1.0
assert not failed_th


# ------------------------------------------------------------
# Tsai-Wu
# ------------------------------------------------------------

fi_tw, failed_tw = tsai_wu_failure(
    stress,
    strengths
)

print("\nTsai-Wu")
print("-" * 60)
print("Failure index:", fi_tw)
print("Failed:", failed_tw)

assert fi_tw < 1.0
assert not failed_tw


# ------------------------------------------------------------
# Hashin
# ------------------------------------------------------------

fi_hashin, max_fi_hashin, failure_mode_hashin, failed_hashin = hashin_failure(
    stress,
    strengths
)

print("\nHashin")
print("-" * 60)
print("Failure indices:", fi_hashin)
print("Maximum FI:", max_fi_hashin)
print("Failure mode:", failure_mode_hashin)
print("Failed:", failed_hashin)

assert max_fi_hashin < 1.0
assert not failed_hashin


print("\n" + "=" * 60)
print("ALL FAILURE CRITERIA TESTS PASSED")
print("=" * 60)