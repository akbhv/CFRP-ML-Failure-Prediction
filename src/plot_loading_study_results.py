import os

import pandas as pd
import matplotlib.pyplot as plt


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
# FIGURE 1
# FAILURE LOAD FACTOR COMPARISON
# ============================================================

ax = df.plot(
    kind="bar",
    figsize=(13, 7),
)

ax.set_xlabel("Loading Case")
ax.set_ylabel("Critical Load Factor, λ")
ax.set_title(
    "First-Ply-Failure Load Factor for Different Failure Criteria"
)

ax.legend(
    title="Failure Criterion",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "failure_load_factor_comparison_final.png",
    ),
    dpi=400,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2
# FAILURE-CRITERION SPREAD
# ============================================================

spread = df.max(axis=1) - df.min(axis=1)

spread = spread.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 6.5))

spread.plot(
    kind="bar",
    ax=ax,
)

ax.set_xlabel("Loading Case")
ax.set_ylabel("Range of Critical Load Factor, Δλ")
ax.set_title(
    "Variation in First-Ply-Failure Prediction Among Failure Criteria"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "failure_criterion_spread_final.png",
    ),
    dpi=400,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3
# HEATMAP OF CRITICAL LOAD FACTOR
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6.5))

heatmap = ax.imshow(
    df.values,
    aspect="auto",
)

ax.set_xticks(range(len(criterion_order)))
ax.set_xticklabels(
    criterion_order,
    rotation=35,
    ha="right",
)

ax.set_yticks(range(len(loading_order)))
ax.set_yticklabels(loading_order)

ax.set_xlabel("Failure Criterion")
ax.set_ylabel("Loading Case")
ax.set_title(
    "Critical First-Ply-Failure Load Factor Across Loading Cases"
)

# Numerical annotations
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        ax.text(
            j,
            i,
            f"{df.iloc[i, j]:.3f}",
            ha="center",
            va="center",
        )

colorbar = fig.colorbar(
    heatmap,
    ax=ax,
)

colorbar.set_label(
    "Critical Load Factor, λ"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "failure_load_factor_heatmap_final.png",
    ),
    dpi=400,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SYMMETRY CHECK
# ============================================================

symmetry_pairs = [
    (
        "Longitudinal tension",
        "Transverse tension",
    ),
    (
        "In-plane shear",
        "Tension-compression",
    ),
]

print("=" * 100)
print("FINAL LOADING STUDY FIGURES")
print("=" * 100)

print()
print("Loading-case order:")
for i, case in enumerate(loading_order, start=1):
    print(f"{i}. {case}")

print()
print("Symmetry checks")
print("-" * 100)

for case_a, case_b in symmetry_pairs:

    difference = (
        df.loc[case_a] - df.loc[case_b]
    ).abs()

    max_difference = difference.max()

    print(
        f"{case_a} vs {case_b}: "
        f"maximum absolute difference = "
        f"{max_difference:.12e}"
    )

print()
print("Criterion spread")
print("-" * 100)

for case in spread.index:
    print(
        f"{case:<25}"
        f"Δλ = {spread[case]:.4f}"
    )

print()
print("Generated files")
print("-" * 100)

print(
    os.path.join(
        output_dir,
        "failure_load_factor_comparison_final.png",
    )
)

print(
    os.path.join(
        output_dir,
        "failure_criterion_spread_final.png",
    )
)

print(
    os.path.join(
        output_dir,
        "failure_load_factor_heatmap_final.png",
    )
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)