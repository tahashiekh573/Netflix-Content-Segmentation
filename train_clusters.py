
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# TASK 4 - K-MEANS CLUSTERING
# ==========================================

print("=" * 55)
print("NETFLIX CONTENT SEGMENTATION")
print("K-MEANS CLUSTERING")
print("=" * 55)


# Load dataset
df = pd.read_csv("Dataset.csv")


# Select features
numerical_features = ["release_year"]
categorical_features = ["type", "country", "rating"]

features = df[numerical_features + categorical_features].copy()

features[numerical_features] = features[numerical_features].fillna(
    features[numerical_features].median()
)

for column in categorical_features:
    features[column] = features[column].fillna("Unknown")


# Scale and encode features
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

processed_features = preprocessor.fit_transform(features)

if hasattr(processed_features, "toarray"):
    processed_features = processed_features.toarray()


# ==========================================
# Find Suitable Number of Clusters
# ==========================================

print("\nEvaluating Different Cluster Counts")
print("-" * 55)

inertia_values = []
cluster_range = range(2, 9)

for k in cluster_range:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(processed_features)
    inertia_values.append(model.inertia_)

    score = silhouette_score(
        processed_features,
        labels,
        sample_size=min(2000, len(features)),
        random_state=42
    )

    print(
        f"Clusters: {k} | "
        f"Inertia: {model.inertia_:.2f} | "
        f"Silhouette Score: {score:.4f}"
    )


# ==========================================
# Plot Elbow Curve
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(list(cluster_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("elbow_curve.png", dpi=300)
plt.close()


# ==========================================
# Train Final K-Means Model
# ==========================================

best_k = 4

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(processed_features)


# Save clustered dataset
df.to_csv("clustered_netflix.csv", index=False)


# ==========================================
# Cluster Analysis
# ==========================================

print("\nFinal Cluster Distribution")
print("-" * 55)

print(df["cluster"].value_counts().sort_index())

print("\nContent Type by Cluster")
print("-" * 55)

print(pd.crosstab(df["cluster"], df["type"]))

print("\nAverage Release Year by Cluster")
print("-" * 55)

print(df.groupby("cluster")["release_year"].mean().round(1))

print("\nSaved: elbow_curve.png")
print("Saved: clustered_netflix.csv")
print("\nClustering completed successfully.")
