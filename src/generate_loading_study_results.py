import os

import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.laminate_loading_study import run_loading_study


# ============================================================
# MATERIAL PROPERTIES
# ============================================================

Ef = 230e9
Em = 3.5e9

Gf = 30e9
Gm = 1.3e9

nu_f = 0.20
nu_m = 0.35

Vf = 0.60

E1, E2, G12, nu12 = calculate_lamina_properties(
    Ef,
    Em,
    Gf,
    Gm,
    nu_f,
    nu_m,
    Vf,
)


# ============================================================
# LAMINA STRENGTHS
# ============================================================

strengths = {
    "Xt": 1950e6,
    "Xc": 1480e6,
    "Yt": 48e6,
    "Yc": 200e6,
    "S": 79e6,
}


# ============================================================
# LAMINATE
# ============================================================

ply_angles = [
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0,
]

ply_thickness = 0.125e-3


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = "results/loading_study"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# RUN STUDY
# ============================================================

results = run_loading_study(
    ply_angles=ply_angles,
    ply_thickness=ply_thickness,
    strengths=strengths,
    E1=E1,
    E2=E2,
    G12=G12,
    nu12=nu12,
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

results.to_csv(
    os.path.join(output_dir, "loading_study_complete.csv"),
    index=False,
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison_table = results.pivot(
    index="loading_case",
    columns="criterion",
    values="critical_load_factor",
)

criterion_order = [
    "Maximum Stress",
    "Maximum Strain",
    "Tsai-Hill",
    "Tsai-Wu",
    "Hoffman",
    "Hashin",
    "Puck",
]

comparison_table = comparison_table[criterion_order]

comparison_table.to_csv(
    os.path.join(output_dir, "failure_load_factor_comparison.csv")
)


# ============================================================
# CREATE GOVERNING-PLY TABLE
# ============================================================

governing_columns = [
    "loading_case",
    "criterion",
    "critical_load_factor",
    "ply",
    "angle_deg",
    "surface",
    "failure_mode",
]

governing_table = results[governing_columns].copy()

governing_table.to_csv(
    os.path.join(output_dir, "governing_failure_locations.csv"),
    index=False,
)


# ============================================================
# CREATE SUMMARY STATISTICS
# ============================================================

summary_records = []

for case_name, group in results.groupby("loading_case"):

    minimum = group.loc[group["critical_load_factor"].idxmin()]
    maximum = group.loc[group["critical_load_factor"].idxmax()]

    summary_records.append(
        {
            "loading_case": case_name,
            "most_conservative_criterion": minimum["criterion"],
            "minimum_failure_load_factor": minimum["critical_load_factor"],
            "least_conservative_criterion": maximum["criterion"],
            "maximum_failure_load_factor": maximum["critical_load_factor"],
        }
    )

summary_table = pd.DataFrame(summary_records)

summary_table.to_csv(
    os.path.join(output_dir, "loading_study_summary.csv"),
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 100)
print("LOADING STUDY RESULTS GENERATED")
print("=" * 100)

print()
print("Material properties")
print("-" * 100)
print(f"E1   = {E1 / 1e9:.6f} GPa")
print(f"E2   = {E2 / 1e9:.6f} GPa")
print(f"G12  = {G12 / 1e9:.6f} GPa")
print(f"nu12 = {nu12:.6f}")

print()
print("Failure load-factor comparison")
print("-" * 100)
print(comparison_table.to_string(float_format=lambda x: f"{x:.4f}"))

print()
print("Most conservative criterion for each loading case")
print("-" * 100)

for _, row in summary_table.iterrows():
    print(
        f"{row['loading_case']:<25}"
        f"{row['most_conservative_criterion']:<20}"
        f"lambda = {row['minimum_failure_load_factor']:.4f}"
    )

print()
print("Output files")
print("-" * 100)
print(os.path.join(output_dir, "loading_study_complete.csv"))
print(os.path.join(output_dir, "failure_load_factor_comparison.csv"))
print(os.path.join(output_dir, "governing_failure_locations.csv"))
print(os.path.join(output_dir, "loading_study_summary.csv"))

print()
print("=" * 100)
print("DONE")
print("=" * 100)