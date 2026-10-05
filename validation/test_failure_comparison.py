"""
Validation test for the unified failure-criteria comparison.

All criteria are evaluated for the same local stress state.
"""

import numpy as np

from src.failure_comparison import evaluate_all_criteria


# ============================================================
# Material properties
# ============================================================

E1 = 139.4e9
E2 = 8.554729e9
G12 = 3.051643e9
nu12 = 0.26

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}


# ============================================================
# Puck parameters
# ============================================================

puck_parameters = {
    "p_perp_parallel_t": 0.30,
    "p_perp_parallel_c": 0.25,
    "p_perp_perp_c": 0.25,
    "m_sigma_f": 1.10,
    "E1": E1,
    "Ef": 230e9,
    "nu12": nu12,
    "nuf": 0.20,
}


# ============================================================
# Test stress state
# ============================================================

stress_local = np.array([
    500e6,
    20e6,
    30e6,
])


# ============================================================
# Evaluate all criteria
# ============================================================

results = evaluate_all_criteria(
    stress_local=stress_local,
    strengths=strengths,
    E1=E1,
    E2=E2,
    G12=G12,
    nu12=nu12,
    puck_parameters=puck_parameters,
)


# ============================================================
# Display results
# ============================================================

print("=" * 70)
print("UNIFIED FAILURE CRITERIA COMPARISON")
print("=" * 70)

print("\nLocal stress state")
print("-" * 70)
print(f"sigma_1  = {stress_local[0] / 1e6:.3f} MPa")
print(f"sigma_2  = {stress_local[1] / 1e6:.3f} MPa")
print(f"tau_12   = {stress_local[2] / 1e6:.3f} MPa")

print("\nFailure Criteria")
print("-" * 70)

for criterion, result in results.items():

    print(
        f"{criterion:<20} "
        f"FI = {result['failure_index']:.6f}   "
        f"Failed = {result['failed']}"
    )

    if "failure_mode" in result:
        print(
            f"{'':<20}"
            f"Mode = {result['failure_mode']}"
        )


# ============================================================
# Basic validation checks
# ============================================================

assert len(results) == 7

for criterion, result in results.items():

    assert "failure_index" in result
    assert "failed" in result

    assert np.isfinite(result["failure_index"])
    assert result["failure_index"] >= 0.0

    assert isinstance(result["failed"], bool)


assert "failure_mode" in results["Hashin"]
assert "failure_mode" in results["Puck"]

print("\n" + "=" * 70)
print("UNIFIED COMPARISON TEST PASSED")
print("=" * 70)