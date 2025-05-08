import pandas as pd

# Metrics to analyze
metrics = ["Gls/90", "Ast/90", "xG/90", "Tkl", "Int", "Blocks"]

# Load data with only Team and selected metrics
df = pd.read_csv("results.csv", usecols=["Team"] + metrics)

# Clean metrics: convert to numbers, replace invalids, clip negatives
for metric in metrics:
    df[metric] = pd.to_numeric(
        df[metric].replace(["N/a", "NA", "", "nan"], pd.NA),
        errors="coerce"
    ).clip(lower=0)

# Clean Team column: remove blanks, strip spaces, drop missing
df["Team"] = df["Team"].replace(["", "NA"], pd.NA).str.strip()
df = df.dropna(subset=["Team"])

# Raise error if no valid teams remain
if df["Team"].empty:
    raise ValueError("No valid teams")

# Group by team: use mean for per-90 stats, sum for defensive totals
team_stats = df.groupby("Team")[metrics].agg({
    "Gls/90": "mean",
    "Ast/90": "mean",
    "xG/90": "mean",
    "Tkl": "sum",
    "Int": "sum",
    "Blocks": "sum"
}).reset_index()

# Store top team for each metric
top_team_data = []

for metric in metrics:
    top_team = team_stats.loc[team_stats[metric].idxmax()]
    row = {"Top Metric": metric, "Team": top_team["Team"]}

    # Add all metric values, mark the top one with '*'
    for m in metrics:
        value = top_team[m]
        row[m] = f"{value:.2f}*" if m == metric else f"{value:.2f}"
    
    top_team_data.append(row)

# Convert results to DataFrame and save to CSV
top_team_df = pd.DataFrame(top_team_data)
top_team_df.to_csv("top_teams_metrics.csv", index=False)

print("\nResults saved to 'top_teams_metrics.csv'")
