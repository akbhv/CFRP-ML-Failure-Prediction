
import pandas as pd

original_path = "data/processed/failure_dataset_v2.csv"
corrected_path = (
    "data/processed/"
    "failure_dataset_v2_puck_corrected_precision.csv"
)

original = pd.read_csv(original_path)
corrected = pd.read_csv(corrected_path)

print("Original shape:", original.shape)
print("Corrected shape:", corrected.shape)

print("\nColumns missing from corrected:")
print(sorted(set(original.columns) - set(corrected.columns)))

print("\nExtra corrected columns:")
print(sorted(set(corrected.columns) - set(original.columns)))

print("\nDiffering columns:")

common = [c for c in original.columns if c in corrected.columns]
differences_found = False

for column in common:
    left = original[column]
    right = corrected[column]

    if not left.equals(right):
        differences_found = True
        unequal = left.ne(right) & ~(left.isna() & right.isna())

        print(
            f"\n{column}: "
            f"{int(unequal.sum())} differing rows; "
            f"dtypes {left.dtype} -> {right.dtype}"
        )
        print("Original examples:", left[unequal].head(3).tolist())
        print("Corrected examples:", right[unequal].head(3).tolist())

if not differences_found:
    print("None")

load_columns = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

print("\nLoad-column exact equality:")
for column in load_columns:
    print(f"{column}: {original[column].equals(corrected[column])}")

print(
    "\nHashin target unchanged:",
    original["failed"].equals(corrected["failed"]),
)
