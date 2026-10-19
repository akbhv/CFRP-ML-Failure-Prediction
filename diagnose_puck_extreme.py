import numpy as np
import pandas as pd

from src.micromechanics import calculate_lamina_properties
from src.materials import get_fiber, get_matrix
from src.strengths import find_lamina_strengths
from src.laminate_failure_comparison import evaluate_laminate_failure
from src.failure_criteria import PUCK_CFRP_PARAMETERS

df = pd.read_csv("data/processed/failure_dataset_v2.csv")
idx = df["puck_fi"].idxmax()
row = df.loc[idx]

fiber = get_fiber("T300")
matrix = get_matrix("Standard Epoxy")

E1, E2, G12, nu12 = calculate_lamina_properties(
    fiber["E"], matrix["E"],
    fiber["G"], matrix["G"],
    fiber["nu"], matrix["nu"], 0.60
)

strength_data = find_lamina_strengths("T300", "Standard Epoxy")
strengths = {
    key: strength_data[key]
    for key in ["Xt", "Xc", "Yt", "Yc", "S"]
}

angles = [0, 45, -45, 90, 90, -45, 45, 0]
thickness = 0.125e-3

N = row[["Nx", "Ny", "Nxy"]].to_numpy(dtype=float)
M = row[["Mx", "My", "Mxy"]].to_numpy(dtype=float)

result = evaluate_laminate_failure(
    E1, E2, G12, nu12,
    angles, thickness, N, M, strengths,
    puck_parameters=PUCK_CFRP_PARAMETERS
)

governing = max(
    result["ply_results"],
    key=lambda p: p["criteria"]["Puck"]["failure_index"]
)

print("\n--- RECORDED WORST SAMPLE ---")
print("CSV row index:", idx)
print("Recorded Puck FI:", row["puck_fi"])
print("Recorded Puck mode:", row["puck_mode"])
print("Recorded fracture angle:", row["puck_fracture_angle"])
print("Sampling region:", row["sampling_region"])
print("Load vector N:", N)
print("Load vector M:", M)

print("\n--- RECOMPUTED GOVERNING PLY SURFACE ---")
print("Ply:", governing["ply"])
print("Angle:", governing["angle"])
print("Surface:", governing["surface"])
print("Local strain:", governing["strain_local"])
print("Local stress [Pa]:", governing["stress_local"])
print("Local stress [MPa]:", governing["stress_local"] / 1e6)
print("Puck result:", governing["criteria"]["Puck"])

print("\n--- MATERIAL STRENGTHS ---")
for key, value in strengths.items():
    print(f"{key}: {value:.8g} Pa ({value / 1e6:.6g} MPa)")

print("\n--- PUCK PARAMETERS ---")
print(PUCK_CFRP_PARAMETERS)
