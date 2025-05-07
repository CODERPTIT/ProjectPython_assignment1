import pandas as pd

# Load DataFrame
try:
    df = pd.read_csv("results.csv")
except FileNotFoundError:
    raise FileNotFoundError("results.csv not found")

# Drop columns that are all NaN
df = df.dropna(axis=1, how="all")
print(f"Total columns in results.csv: {len(df.columns)}")

# Ensure 'Team' column exists
if "Team" not in df.columns:
    raise ValueError("DataFrame must contain 'Team' column")

# Identify non-numeric columns to exclude
exclude_columns = ["Player", "Nation", "Team", "Position"]

# Preprocess columns to handle "N/a" and convert to numeric
for col in df.columns:
    if col not in exclude_columns:
        df[col] = pd.to_numeric(df[col].replace("N/a", pd.NA), errors="coerce")

# Get numeric columns
numeric_columns = [
    col for col in df.columns
    if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col])
]
print(f"Number of numeric columns: {len(numeric_columns)}")

# Check if numeric columns exist
if not numeric_columns:
    raise ValueError("No numeric columns found in the dataset")
# Initialize list to store results
rows = []

# Row 0: Statistics for all players
all_stats = {"Player/Team": "all"}
for col in numeric_columns:
    valid_data = df[col].dropna()
    all_stats[f"Median of {col}"] = valid_data.median() if not valid_data.empty else pd.NA
    all_stats[f"Mean of {col}"] = valid_data.mean() if not valid_data.empty else pd.NA
    all_stats[f"Std of {col}"] = valid_data.std() if not valid_data.empty else pd.NA
rows.append(all_stats)

# Rows 1 to n: Statistics for each team
teams = sorted(df["Team"].unique())
for team in teams:
    team_df = df[df["Team"] == team]
    team_stats = {"Player/Team": team}
    for col in numeric_columns:
        valid_data = team_df[col].dropna()
        team_stats[f"Median of {col}"] = valid_data.median() if not valid_data.empty else pd.NA
        team_stats[f"Mean of {col}"] = valid_data.mean() if not valid_data.empty else pd.NA
        team_stats[f"Std of {col}"] = valid_data.std() if not valid_data.empty else pd.NA
    rows.append(team_stats)

# Create DataFrame from results
results_df = pd.DataFrame(rows)

# Round numeric values to 2 decimal places (except Player/Team)
for col in results_df.columns:
    if col != "Player/Team":
        results_df[col] = results_df[col].round(2)

# Save to CSV
results_df.to_csv("results2.csv", index=False, encoding="utf-8-sig")
print(f"✅ Successfully saved statistics to results2.csv with {results_df.shape[0]} rows and {results_df.shape[1]} columns.")

                                                                
