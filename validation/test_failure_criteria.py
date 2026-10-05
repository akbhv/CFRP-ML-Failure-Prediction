import numpy as np

from src.failure_criteria import (
    maximum_stress_failure,
    maximum_strain_failure,
    tsai_hill_failure,
    tsai_wu_failure,
    hoffman_failure,
    hashin_failure,
    _puck_fracture_plane_stress,
    puck_failure,
)


strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}

# Lamina elastic properties
E1 = 139.4e9
E2 = 8.554729e9
G12 = 3.051643e9

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


# ============================================================
# Maximum Strain
# ============================================================

strain_local = np.array([
    0.002,
    0.003,
    0.004
])

strain_failure_indices, strain_max_fi, strain_failed = maximum_strain_failure(
    strain_local,
    strengths,
    E1,
    E2,
    G12
)

print("\nMaximum Strain")
print("-" * 60)
print("Failure indices:", strain_failure_indices)
print("Maximum FI:", strain_max_fi)
print("Failed:", strain_failed)

assert strain_max_fi < 1.0
assert not strain_failed

# ============================================================
# Maximum Strain Boundary Tests
# ============================================================

epsilon_1_t = strengths["Xt"] / E1
epsilon_1_c = strengths["Xc"] / E1
epsilon_2_t = strengths["Yt"] / E2
epsilon_2_c = strengths["Yc"] / E2
gamma_12_limit = strengths["S"] / G12


# Longitudinal tension boundary
_, fi, failed = maximum_strain_failure(
    np.array([epsilon_1_t, 0.0, 0.0]),
    strengths,
    E1,
    E2,
    G12
)

assert np.isclose(fi, 1.0)
assert failed


# Longitudinal compression boundary
_, fi, failed = maximum_strain_failure(
    np.array([-epsilon_1_c, 0.0, 0.0]),
    strengths,
    E1,
    E2,
    G12
)

assert np.isclose(fi, 1.0)
assert failed


# Transverse tension boundary
_, fi, failed = maximum_strain_failure(
    np.array([0.0, epsilon_2_t, 0.0]),
    strengths,
    E1,
    E2,
    G12
)

assert np.isclose(fi, 1.0)
assert failed


# Transverse compression boundary
_, fi, failed = maximum_strain_failure(
    np.array([0.0, -epsilon_2_c, 0.0]),
    strengths,
    E1,
    E2,
    G12
)

assert np.isclose(fi, 1.0)
assert failed


# Shear boundary
_, fi, failed = maximum_strain_failure(
    np.array([0.0, 0.0, gamma_12_limit]),
    strengths,
    E1,
    E2,
    G12
)

assert np.isclose(fi, 1.0)
assert failed


print("\nMaximum Strain Boundary Tests")
print("-" * 60)
print("Tension, compression, transverse and shear boundaries: PASS")

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
# Hoffman
# ------------------------------------------------------------

fi_hoffman, failed_hoffman = hoffman_failure(
    stress,
    strengths
)

print("\nHoffman")
print("-" * 60)
print("Failure index:", fi_hoffman)
print("Failed:", failed_hoffman)

assert fi_hoffman < 1.0
assert not failed_hoffman

# ============================================================
# Hoffman Boundary Tests
# ============================================================

hoffman_cases = {
    "longitudinal tension": np.array([strengths["Xt"], 0.0, 0.0]),
    "longitudinal compression": np.array([-strengths["Xc"], 0.0, 0.0]),
    "transverse tension": np.array([0.0, strengths["Yt"], 0.0]),
    "transverse compression": np.array([0.0, -strengths["Yc"], 0.0]),
    "shear": np.array([0.0, 0.0, strengths["S"]]),
}

print("\nHoffman Boundary Tests")
print("-" * 60)

for name, stress_case in hoffman_cases.items():
    fi, failed = hoffman_failure(stress_case, strengths)

    print(f"{name}: FI = {fi:.10f}, Failed = {failed}")

    assert np.isclose(fi, 1.0, atol=1e-12)
    assert failed

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


# ============================================================
# Puck Fracture Plane Stress Transformation
# ============================================================

sigma_2 = 20e6
tau_12 = 30e6

sigma_n, tau_nt = _puck_fracture_plane_stress(
    sigma_2,
    tau_12,
    0.0
)

assert np.isclose(sigma_n, sigma_2)
assert np.isclose(tau_nt, tau_12)

print("\nPuck Fracture Plane Transformation")
print("-" * 60)
print("Zero-angle transformation: PASS")

# ============================================================
# Puck Parameters
# ============================================================

puck_parameters = {
    "p_perp_parallel_t": 0.30,
    "p_perp_parallel_c": 0.25,
    "p_perp_perp_c": 0.25,

    "m_sigma_f": 1.10,

    "E1": 139.4e9,
    "Ef": 230e9,
    "nu12": 0.26,
    "nuf": 0.20,
}

# ============================================================
# Puck
# ============================================================

fi_puck, max_fi_puck, mode_puck, theta_puck, failed_puck = puck_failure(
    stress,
    strengths,
    puck_parameters,
)

print("\nPuck")
print("-" * 60)
print("Failure indices:", fi_puck)
print("Maximum FI:", max_fi_puck)
print("Failure mode:", mode_puck)
print("Puck fracture angle [deg]:", np.degrees(theta_puck))
print("Failed:", failed_puck)

assert max_fi_puck < 1.0
assert not failed_puck

# ============================================================
# Puck Boundary Tests
# ============================================================

print("\nPuck Boundary Tests")
print("-" * 60)

puck_boundary_cases = {
    "longitudinal tension": np.array([
        strengths["Xt"], 0.0, 0.0
    ]),

    "longitudinal compression": np.array([
        -strengths["Xc"], 0.0, 0.0
    ]),

    "transverse tension": np.array([
        0.0, strengths["Yt"], 0.0
    ]),

    "transverse compression": np.array([
        0.0, -strengths["Yc"], 0.0
    ]),

    "pure shear": np.array([
        0.0, 0.0, strengths["S"]
    ]),
}

puck_boundary_results = {}

for name, stress_case in puck_boundary_cases.items():

    (
        failure_indices,
        max_fi,
        failure_mode,
        fracture_angle,
        failed,
    ) = puck_failure(
        stress_case,
        strengths,
        puck_parameters,
    )

    puck_boundary_results[name] = {
        "fi": max_fi,
        "mode": failure_mode,
        "angle": fracture_angle,
        "failed": failed,
    }

    print(
        f"{name}: "
        f"FI = {max_fi:.10f}, "
        f"Mode = {failure_mode}, "
        f"Failed = {failed}"
    )


# ============================================================
# Puck Boundary Assertions
# ============================================================

print("\nPuck Boundary Assertions")
print("-" * 60)

for name, result in puck_boundary_results.items():

    assert np.isclose(
        result["fi"],
        1.0,
        atol=1e-10,
    ), (
        f"Puck boundary failed for {name}: "
        f"FI = {result['fi']}"
    )

    assert result["failed"] is True, (
        f"Puck boundary should fail for {name}"
    )

print("All Puck boundary tests: PASS")

print("\n" + "=" * 60)
print("ALL FAILURE CRITERIA TESTS PASSED")
print("=" * 60)