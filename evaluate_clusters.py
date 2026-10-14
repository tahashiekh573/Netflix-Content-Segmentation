
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# TASK 4 - CLUSTER EVALUATION
# ==========================================

print("=" * 55)
print("NETFLIX CONTENT SEGMENTATION")
print("CLUSTER EVALUATION")
print("=" * 55)


# Load dataset
df = pd.read_csv("Dataset.csv")


# Select features
numerical_features = ["release_year"]
categorical_features = ["type", "country", "rating"]

features = df[numerical_features + categorical_features].copy()

features["release_year"] = features["release_year"].fillna(
    features["release_year"].median()
)

for column in categorical_features:
    features[column] = features[column].fillna("Unknown")


# Encode and scale features
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

X = preprocessor.fit_transform(features)

if hasattr(X, "toarray"):
    X = X.toarray()


# ==========================================
# Evaluate Different Cluster Counts
# ==========================================

results = []

print("\nCluster Evaluation Results")
print("-" * 55)

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        sample_size=min(2000, len(X)),
        random_state=42
    )

    results.append({
        "Clusters": k,
        "Inertia": model.inertia_,
        "Silhouette Score": score
    })

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette Score = {score:.4f}"
    )


# Save evaluation results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "cluster_evaluation.csv",
    index=False
)


# ==========================================
# Silhouette Score Comparison
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Clusters"],
    results_df["Silhouette Score"],
    marker="o"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(results_df["Clusters"])
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "silhouette_comparison.png",
    dpi=300
)

plt.close()


# ==========================================
# Best Score
# ==========================================

best_result = results_df.loc[
    results_df["Silhouette Score"].idxmax()
]

print("\nBest Silhouette Score:")
print("Number of Clusters:", int(best_result["Clusters"]))
print("Score:", round(best_result["Silhouette Score"], 4))

print("\nSaved: cluster_evaluation.csv")
print("Saved: silhouette_comparison.png")
print("\nEvaluation completed successfully.")
