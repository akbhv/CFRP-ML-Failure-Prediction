
import pandas as pd
import numpy as np

path = "data/processed/puck_recalculation_comparison.csv"
df = pd.read_csv(path)

print("=== Puck FI distribution: old vs new ===")
for label, col in [
    ("Old", "old_puck_fi"),
    ("New", "new_puck_fi"),
]:
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n{label}:")
    print(s.describe(percentiles=[0.90, 0.95, 0.99, 0.999]))

print("\n=== Classification changes ===")
print(pd.crosstab(
    df["old_puck_failed"],
    df["new_puck_failed"],
    rownames=["Old failed"],
    colnames=["New failed"],
    margins=True,
))

print("\n=== Mode transitions ===")
changed = df[df["old_puck_mode"] != df["new_puck_mode"]]
print(changed.groupby(
    ["old_puck_mode", "new_puck_mode"]
).size().sort_values(ascending=False).head(15))

print("\n=== Remaining extreme new values ===")
print(
    df.nlargest(10, "new_puck_fi")[
        [
            "row_index",
            "old_puck_fi",
            "new_puck_fi",
            "old_puck_mode",
            "new_puck_mode",
            "new_puck_ply",
        ]
    ].to_string(index=False)
)
