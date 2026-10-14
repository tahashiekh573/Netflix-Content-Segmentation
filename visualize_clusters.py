
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# TASK 4 - CLUSTER VISUALIZATION
# ==========================================

print("=" * 55)
print("NETFLIX CONTENT SEGMENTATION")
print("CLUSTER VISUALIZATION")
print("=" * 55)


# Load clustered dataset
df = pd.read_csv("clustered_netflix.csv")


# ==========================================
# Chart 1 - Cluster Distribution
# ==========================================

cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Netflix Content Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Titles")
plt.tight_layout()

plt.savefig("cluster_distribution.png", dpi=300)
plt.close()


# ==========================================
# Chart 2 - Content Type by Cluster
# ==========================================

type_counts = pd.crosstab(
    df["cluster"],
    df["type"]
)

type_counts.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Movies and TV Shows by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("content_type_by_cluster.png", dpi=300)
plt.close()


# ==========================================
# Chart 3 - Average Release Year
# ==========================================

average_year = df.groupby("cluster")["release_year"].mean()

plt.figure(figsize=(9, 5))

plt.bar(
    average_year.index.astype(str),
    average_year.values
)

plt.title("Average Release Year by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Release Year")
plt.tight_layout()

plt.savefig("average_release_year.png", dpi=300)
plt.close()


# ==========================================
# Chart 4 - Rating Distribution
# ==========================================

rating_counts = pd.crosstab(
    df["cluster"],
    df["rating"]
)

rating_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Rating Distribution Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)
plt.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig("rating_distribution.png", dpi=300)
plt.close()


# ==========================================
# Completion
# ==========================================

print("\nVisualizations saved successfully:")
print("1. cluster_distribution.png")
print("2. content_type_by_cluster.png")
print("3. average_release_year.png")
print("4. rating_distribution.png")

print("\nVisualization completed successfully.")
