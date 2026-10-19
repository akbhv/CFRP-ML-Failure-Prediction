
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("data/processed/failure_dataset_v2.csv")
COMPARISON = Path("data/processed/puck_recalculation_comparison.csv")
OUTPUT = Path("data/processed/failure_dataset_v2_puck_corrected_verified.csv")

original = pd.read_csv(SOURCE)
report = pd.read_csv(COMPARISON)

if len(original) != 20000 or len(report) != len(original):
    raise ValueError("Unexpected row count.")

if not np.array_equal(report["row_index"].to_numpy(), np.arange(len(original))):
    raise ValueError("Comparison row indices do not align.")

if not np.isfinite(report["new_puck_fi"].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite recomputed Puck FI detected.")

# Start from the original in-memory DataFrame.
corrected = original.copy(deep=True)

puck_map = {
    "puck_fi": "new_puck_fi",
    "puck_mode": "new_puck_mode",
    "puck_failed": "new_puck_failed",
    "puck_ply": "new_puck_ply",
    "puck_angle": "new_puck_angle",
    "puck_surface": "new_puck_surface",
    "puck_fracture_angle": "new_fracture_angle",
}

missing = [c for c in puck_map if c not in corrected.columns]
missing_report = [c for c in puck_map.values() if c not in report.columns]

if missing or missing_report:
    raise ValueError(
        f"Missing dataset columns: {missing}; "
        f"missing report columns: {missing_report}"
    )

for target, source in puck_map.items():
    corrected[target] = report[source].to_numpy()

# Preserve the original integer representation of the failure label.
corrected["puck_failed"] = corrected["puck_failed"].astype(original["puck_failed"].dtype)

# Verify all non-Puck columns are unchanged before writing.
non_puck = [c for c in original.columns if not c.startswith("puck_")]

if not original[non_puck].equals(corrected[non_puck]):
    raise AssertionError("Non-Puck columns changed in memory.")

if not corrected["failed"].equals(corrected["hashin_failed"]):
    raise AssertionError("Hashin target mismatch.")

if OUTPUT.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT}")

corrected.to_csv(OUTPUT, index=False, float_format="%.17g")

# Verify persisted numerical equality with a tolerant comparison.
saved = pd.read_csv(OUTPUT)
for column in non_puck:
    if pd.api.types.is_numeric_dtype(original[column]):
        if not np.allclose(
            original[column].to_numpy(dtype=float),
            saved[column].to_numpy(dtype=float),
            rtol=1e-14,
            atol=1e-14,
            equal_nan=True,
        ):
            raise AssertionError(f"Persisted non-Puck values differ: {column}")
    elif not original[column].equals(saved[column]):
        raise AssertionError(f"Persisted non-Puck values differ: {column}")

print(f"Saved verified corrected dataset: {OUTPUT}")
print("Non-Puck columns unchanged in memory: True")
print("Persisted non-Puck columns numerically preserved: True")
print("Hashin target unchanged: True")
print("All Puck FI values finite:", bool(np.isfinite(saved["puck_fi"]).all()))
print("Rows:", len(saved))
