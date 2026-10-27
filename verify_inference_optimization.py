
import numpy as np

from api.main import ML_TARGETS, ML_FAILURE_THRESHOLD, ml_models, ml_scaler

test_cases = {
    "zero_load": [0, 0, 0, 0, 0, 0],
    "low_load": [1000, 200, 100, 0, 0, 0],
    "normal_load": [100000, 20000, 10000, 0, 0, 0],
    "higher_load": [300000, 60000, 30000, 0, 0, 0],
}

all_passed = True

for case_name, loads in test_cases.items():
    features = np.asarray([loads], dtype=np.float32)
    scaled = ml_scaler.transform(features).astype(np.float32)

    print(f"\nCase: {case_name}")

    for target, display_name in ML_TARGETS.items():
        model = ml_models[target]

        # Original inference method
        old_value = float(
            model.predict(scaled, verbose=0).reshape(-1)[0]
        )

        # Optimized inference method
        new_value = float(
            model(scaled, training=False).numpy().reshape(-1)[0]
        )

        old_value = max(0.0, old_value)
        new_value = max(0.0, new_value)

        difference = abs(old_value - new_value)
        same_decision = (
            (old_value >= ML_FAILURE_THRESHOLD)
            == (new_value >= ML_FAILURE_THRESHOLD)
        )
        passed = np.isclose(
            old_value, new_value, rtol=1e-5, atol=1e-6
        ) and same_decision

        all_passed = all_passed and passed

        print(
            f"  {display_name:18s} "
            f"old={old_value:.8f} "
            f"new={new_value:.8f} "
            f"diff={difference:.2e} "
            f"{'PASS' if passed else 'FAIL'}"
        )

print("\nOVERALL:", "PASS" if all_passed else "FAIL")
