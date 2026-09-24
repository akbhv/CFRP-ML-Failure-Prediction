from src.failure_criteria import hashin_failure


strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6
}


# Pure fiber tension
stress_local = [1950e6, 0.0, 0.0]

failure_indices, max_failure_index, failure_mode, failed = hashin_failure(
    stress_local,
    strengths
)

print("Failure indices:")
for mode, index in failure_indices.items():
    print(f"{mode}: {index:.6f}")

print(f"\nMaximum failure index: {max_failure_index:.6f}")
print(f"Governing failure mode: {failure_mode}")
print(f"Failed: {failed}")