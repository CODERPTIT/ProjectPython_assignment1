import pandas as pd

# Load DataFrame
df = pd.read_csv("results.csv")

# Drop columns that are all NaN or have a single unique value
df = df.dropna(axis=1, how="all").loc[:, df.nunique() > 1]

# Ensure 'Player' column exists
if "Player" not in df.columns:
    raise ValueError("DataFrame must contain 'Player' column")

# Select numeric columns, excluding identifiers
exclude_cols = ["Player", "Nation", "Position", "Team"]
numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

# Initialize result string
result = []

# Find top 3 and bottom 3 for each numeric column
for col in numeric_cols:
    # Sort by column (stable sort by value, preserving original index order for ties)
    sorted_df = df[["Player", col]].dropna().sort_values(by=col, ascending=False)
    
    # Get top 3 and bottom 3
    top_3 = sorted_df.head(3)
    bottom_3 = sorted_df.tail(3)[::-1]  # Reverse to match ascending order
    
    # Format output
    result.append(f"\n=== {col} ===\nTop 3:")
    result.extend(f"{row['Player']}: {row[col]}" for _, row in top_3.iterrows())
    result.append("Bottom 3:")
    result.extend(f"{row['Player']}: {row[col]}" for _, row in bottom_3.iterrows())

# Save to file
output_path = "top_3.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(result))

print(f"Saved results to {output_path}")
