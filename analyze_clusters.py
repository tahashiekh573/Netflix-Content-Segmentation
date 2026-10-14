
import pandas as pd

print("=" * 55)
print("NETFLIX CONTENT SEGMENTATION")
print("CLUSTER ANALYSIS")
print("=" * 55)

# Load clustered dataset
df = pd.read_csv("clustered_netflix.csv")

# Check required columns
required_columns = [
    "cluster",
    "type",
    "country",
    "rating",
    "release_year"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
    raise ValueError(
        "Please check clustered_netflix.csv"
    )

# Overall cluster distribution
print("\nCluster Distribution")
print("-" * 55)

print(df["cluster"].value_counts().sort_index())

# Content type analysis
print("\nContent Type by Cluster")
print("-" * 55)

type_analysis = pd.crosstab(
    df["cluster"],
    df["type"]
)

print(type_analysis)

# Average release year
print("\nAverage Release Year by Cluster")
print("-" * 55)

print(
    df.groupby("cluster")["release_year"]
    .mean()
    .round(2)
)

# Most common ratings
print("\nMost Common Rating by Cluster")
print("-" * 55)

for cluster_id, group in df.groupby("cluster"):
    most_common = group["rating"].mode()

    if not most_common.empty:
        print(
            f"Cluster {cluster_id}: "
            f"{most_common.iloc[0]}"
        )

# Most common countries
print("\nTop Countries by Cluster")
print("-" * 55)

for cluster_id, group in df.groupby("cluster"):
    countries = (
        group["country"]
        .dropna()
        .astype(str)
        .str.split(", ")
        .explode()
    )

    countries = countries[
        countries.str.strip() != ""
    ]

    print(f"\nCluster {cluster_id}:")

    if not countries.empty:
        print(countries.value_counts().head(5))
    else:
        print("No country information available.")

# Save summary
summary = df.groupby("cluster").agg(
    total_content=("title", "count"),
    average_release_year=("release_year", "mean")
).round(2)

summary.to_csv("cluster_summary.csv")

print("\nSaved: cluster_summary.csv")
print("\nCluster analysis completed successfully.")
