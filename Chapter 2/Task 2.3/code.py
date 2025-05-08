import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.ticker import ScalarFormatter

# Set seaborn style
sns.set(style="whitegrid", rc={"grid.alpha": 0.3, "axes.grid": True})

# Create output folder
os.makedirs("histograms", exist_ok=True)

# Load and clean DataFrame
df = pd.read_csv("results.csv")
required_cols = ["Team", "Gls/90", "Ast/90", "xG/90", "Tkl", "Int", "Blocks"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("Missing required columns")

metrics = ["Gls/90", "Ast/90", "xG/90", "Tkl", "Int", "Blocks"]
for metric in metrics:
    df[metric] = pd.to_numeric(df[metric].replace(["N/a", "NA", "", "nan"], pd.NA), errors="coerce").clip(lower=0)

df["Team"] = df["Team"].replace(["", "NA"], pd.NA).str.strip().dropna()
teams = sorted(df["Team"].unique())
if not teams:
    raise ValueError("No valid teams")

# Plot histogram
def plot_histogram(metric, df, teams):
    total_plots = len(teams) + 1
    cols = 2
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.5))
    axes = axes.flatten()
    fig.suptitle(f"Distribution of {metric}", fontsize=14, y=1.02)
    
    # Bin edges (30 bins)
    league_data = df[metric].dropna()
    bins = np.linspace(league_data.min(), league_data.max(), 31) if not league_data.empty else 10
    
    # X-axis formatter
    is_per_90 = metric in ["Gls/90", "Ast/90", "xG/90"]
    formatter = FormatStrFormatter('%.2f') if is_per_90 else ScalarFormatter()
    
    # Plot all histograms
    for idx, (data, title) in enumerate([(league_data, "All Players")] + [(df[df["Team"] == team][metric].dropna(), team) for team in teams]):
        ax = axes[idx]
        sns.histplot(data=data, ax=ax, kde=True, bins=bins, edgecolor="black", 
                     color="skyblue" if title == "All Players" else "lightgreen", 
                     line_kws={"linewidth": 2, "color": "navy" if title == "All Players" else "darkgreen"})
        ax.set_title(title)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel("Players", fontsize=12)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MaxNLocator(6))  # ~6 ticks
    
    # Remove unused subplots
    for i in range(total_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join("histograms", f"{metric.replace('/', '_')}_histogram.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")

# Generate histograms
for metric in metrics:
    print(f"Generating {metric}...")
    plot_histogram(metric, df, teams)

# Verify output
generated_files = [f for f in os.listdir("histograms") if f.endswith("_histogram.png")]
print(f"\n✅ Saved in 'histograms': {', '.join(generated_files)}" if generated_files else "\n⚠️ No histograms generated")
