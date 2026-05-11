import pandas as pd

df = pd.read_csv("data/processed/cleaned_pois.csv", index_col=0)

# Boolean flags
print("=== FLAG FIELDS ===")
for col in ["wheelchair_accessible", "has_takeaway", "is_24_7"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# Opening hours outliers
print("\n=== OPENING HOURS OUTLIERS ===")
days = ["mo", "tu", "we", "th", "fr", "sa", "su"]

for day in days:
    open_col = f"{day}_open"
    close_col = f"{day}_close"

    if open_col not in df.columns:
        continue

    subset = df[[open_col, close_col]].dropna()

    # open == close
    equal = subset[subset[open_col] == subset[close_col]]
    # close < open
    invalid = subset[subset[close_col] < subset[open_col]]

    print(f"\n{day}: open==close: {len(equal)}, close<open: {len(invalid)}")
    if len(invalid) > 0:
        print(invalid.head(3))