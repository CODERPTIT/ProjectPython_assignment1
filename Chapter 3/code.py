import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("clustering_results", exist_ok=True)

# Set seaborn style
sns.set(style="whitegrid")

# Load results.csv
try:
    df = pd.read_csv("results.csv")
except FileNotFoundError:
    print("Error: 'results.csv' not found.")
    exit(1)

# Select all numeric columns for clustering
non_numeric_cols = ["Player", "Nation", "Team", "Position"]
features = [col for col in df.columns if col not in non_numeric_cols and df[col].dtype in [np.float64, np.int64]]

# Verify numeric columns
if not features:
    print("Error: No numeric columns found in results.csv.")
    exit(1)

print("Statistics used for clustering:", features)
print(f"Number of features: {len(features)}")

# Prepare data
X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method and silhouette scores
inertia = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.savefig("clustering_results/elbow_plot.png", bbox_inches="tight", dpi=300)
plt.close()

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker="o")
plt.title("Silhouette Scores for Different K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.savefig("clustering_results/silhouette_plot.png", bbox_inches="tight", dpi=300)
plt.close()

# Print silhouette scores
print("Silhouette Scores:")
for k, score in zip(K_range, silhouette_scores):
    print(f"K={k}: {score:.4f}")

# Choose optimal k
optimal_k = 3  # Adjust based on elbow_plot.png and silhouette_plot.png
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio (PCA): {explained_variance_ratio}")
print(f"Total Variance Explained: {sum(explained_variance_ratio):.2%}")

# Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Create 2D scatter plot with circular markers and centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Cluster"],
    palette="deep",
    marker="o",  # Circular markers
    s=100
)
# Plot centroids as red stars
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    marker="*",
    s=300,
    label="Centroids"
)
plt.title("Player Clusters with Centroids (PCA)")
plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.2%} variance)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.2%} variance)")

# Label top 2 players per cluster
for cluster in range(optimal_k):
    cluster_data = df[df["Cluster"] == cluster]
    top_players = cluster_data["Player"].head(2).tolist()
    for player in top_players:
        idx = df[df["Player"] == player].index[0]
        plt.text(X_pca[idx, 0] + 0.1, X_pca[idx, 1], player, fontsize=8)

plt.legend()
plt.tight_layout()
plt.savefig("clustering_results/cluster_scatter.png", bbox_inches="tight", dpi=300)
plt.show()
