import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

input_file = "results/loading_study/failure_load_factor_comparison.csv"
output_dir = "results/loading_study"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(input_file, index_col=0)

loading_order = [
    "Longitudinal tension",
    "Transverse tension",
    "In-plane shear",
    "Biaxial tension",
    "Tension-compression",
    "Combined loading",
]

criterion_order = [
    "Maximum Stress",
    "Maximum Strain",
    "Tsai-Hill",
    "Tsai-Wu",
    "Hoffman",
    "Hashin",
    "Puck",
]

df = df.loc[loading_order, criterion_order]


# ============================================================
# CREATE FINAL FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6.8))

values = df.values

heatmap = ax.imshow(
    values,
    aspect="auto",
)


# ============================================================
# AXES
# ============================================================

ax.set_xticks(range(len(criterion_order)))
ax.set_xticklabels(
    criterion_order,
    rotation=30,
    ha="right",
    fontsize=10,
)

ax.set_yticks(range(len(loading_order)))
ax.set_yticklabels(
    loading_order,
    fontsize=10,
)

ax.set_xlabel(
    "Failure Criterion",
    fontsize=11,
)

ax.set_ylabel(
    "Loading Case",
    fontsize=11,
)

ax.set_title(
    "First-Ply-Failure Load Factor Across Loading Cases and Failure Criteria",
    fontsize=13,
    pad=12,
)


# ============================================================
# ANNOTATIONS
# ============================================================

for i in range(values.shape[0]):

    row = values[i]

    minimum_index = np.argmin(row)

    for j in range(values.shape[1]):

        value = values[i, j]

        # Governing / most conservative prediction
        if j == minimum_index:
            fontweight = "bold"
            bbox = dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
            )
        else:
            fontweight = "normal"
            bbox = None

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight=fontweight,
            bbox=bbox,
        )


# ============================================================
# COLORBAR
# ============================================================

colorbar = fig.colorbar(
    heatmap,
    ax=ax,
    pad=0.02,
)

colorbar.set_label(
    "Critical Load Factor, λ",
    fontsize=11,
)


# ============================================================
# GRID
# ============================================================

ax.set_xticks(
    np.arange(-0.5, len(criterion_order), 1),
    minor=True,
)

ax.set_yticks(
    np.arange(-0.5, len(loading_order), 1),
    minor=True,
)

ax.grid(
    which="minor",
    linestyle="-",
    linewidth=0.8,
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.015,
    "Bold boxed values indicate the lowest predicted first-ply-failure load factor for each loading case.",
    ha="center",
    fontsize=9,
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    output_dir,
    "first_ply_failure_criterion_comparison_final.png",
)

plt.tight_layout(
    rect=[0, 0.04, 1, 1]
)

plt.savefig(
    output_file,
    dpi=500,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PRINT REPORT SUMMARY
# ============================================================

print("=" * 100)
print("FINAL REPORT FIGURE GENERATED")
print("=" * 100)

print()
print(f"Output: {output_file}")

print()
print("Governing criterion for each loading case")
print("-" * 100)

for case in loading_order:

    criterion = df.loc[case].idxmin()
    critical_lambda = df.loc[case].min()

    print(
        f"{case:<25}"
        f"{criterion:<20}"
        f"lambda = {critical_lambda:.4f}"
    )

print()
print("=" * 100)
print("DONE")
print("=" * 100)