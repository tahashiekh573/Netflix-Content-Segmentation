
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Netflix Content Segmentation",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Segmentation Dashboard")
st.markdown(
    "Explore Netflix movies and TV shows using "
    "K-Means clustering and interactive visualizations."
)

# -------------------------------
# Load Dataset
# -------------------------------
DATA_FILE = Path("clustered_netflix.csv")

if not DATA_FILE.exists():
    st.error(
        "clustered_netflix.csv not found. "
        "Please run train_clusters.py first."
    )
    st.stop()

df = pd.read_csv(DATA_FILE)

# -------------------------------
# Check Required Columns
# -------------------------------
required_columns = [
    "type", "title", "country", "release_year", "rating", "cluster"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Required columns are missing: "
        + ", ".join(missing_columns)
        + ". Please check clustered_netflix.csv."
    )
    st.stop()

df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Dashboard Filters")

content_types = sorted(df["type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Content Type",
    options=content_types,
    default=content_types
)

countries = sorted(
    df["country"].fillna("Unknown").unique().tolist()
)
selected_countries = st.sidebar.multiselect(
    "Country",
    options=countries,
    default=countries
)

ratings = sorted(
    df["rating"].fillna("Unknown").unique().tolist()
)
selected_ratings = st.sidebar.multiselect(
    "Content Rating",
    options=ratings,
    default=ratings
)

clusters = sorted(
    df["cluster"].dropna().unique().tolist()
)
selected_clusters = st.sidebar.multiselect(
    "Cluster",
    options=clusters,
    default=clusters,
    format_func=lambda value: f"Cluster {int(value)}"
)

# -------------------------------
# Apply Filters
# -------------------------------
filtered_df = df[
    df["type"].isin(selected_types)
    & df["country"].fillna("Unknown").isin(selected_countries)
    & df["rating"].fillna("Unknown").isin(selected_ratings)
    & df["cluster"].isin(selected_clusters)
].copy()

# -------------------------------
# KPI Cards
# -------------------------------
st.subheader("📊 Dataset Overview")

total_titles = len(filtered_df)
movie_count = (filtered_df["type"] == "Movie").sum()
tv_count = (filtered_df["type"] == "TV Show").sum()
cluster_count = filtered_df["cluster"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", f"{total_titles:,}")
col2.metric("Movies", f"{movie_count:,}")
col3.metric("TV Shows", f"{tv_count:,}")
col4.metric("Clusters", cluster_count)

if filtered_df.empty:
    st.warning("No titles match the selected filters.")
    st.stop()

# -------------------------------
# Charts
# -------------------------------
st.divider()
st.subheader("📈 Content Visualizations")

left, right = st.columns(2)

with left:
    st.markdown("**Titles by Cluster**")
    cluster_counts = (
        filtered_df["cluster"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [f"Cluster {int(c)}" for c in cluster_counts.index],
        cluster_counts.values
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Titles")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Movies vs TV Shows**")
    type_counts = filtered_df["type"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.markdown("**Titles by Release Year**")
    year_counts = (
        filtered_df.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(year_counts.index, year_counts.values, marker=".")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Content Rating Distribution**")
    rating_counts = (
        filtered_df["rating"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(rating_counts.index[::-1], rating_counts.values[::-1])
    ax.set_xlabel("Number of Titles")
    ax.set_ylabel("Rating")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------
# Cluster Summary
# -------------------------------
st.divider()
st.subheader("🧩 Cluster Analysis")

summary = (
    filtered_df.groupby("cluster")
    .agg(
        Total_Titles=("title", "count"),
        Movies=("type", lambda x: (x == "Movie").sum()),
        TV_Shows=("type", lambda x: (x == "TV Show").sum()),
        Average_Release_Year=("release_year", "mean")
    )
    .reset_index()
)

summary["cluster"] = summary["cluster"].astype(int)
summary["Average_Release_Year"] = (
    summary["Average_Release_Year"].round(1)
)

st.dataframe(summary, use_container_width=True, hide_index=True)

# -------------------------------
# Dataset Explorer
# -------------------------------
st.divider()
st.subheader("🎞️ Netflix Dataset Explorer")

search_text = st.text_input(
    "Search titles",
    placeholder="Enter a movie or TV show title..."
)

search_df = filtered_df.copy()

if search_text.strip():
    search_df = search_df[
        search_df["title"]
        .fillna("")
        .str.contains(search_text.strip(), case=False, regex=False)
    ]

display_columns = [
    column for column in [
        "show_id", "title", "type", "country",
        "release_year", "rating", "duration",
        "listed_in", "cluster"
    ]
    if column in search_df.columns
]

st.caption(f"Showing {len(search_df):,} matching titles")
st.dataframe(
    search_df[display_columns],
    use_container_width=True,
    hide_index=True
)

# -------------------------------
# Download Filtered Data
# -------------------------------
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Dataset (CSV)",
    data=csv_data,
    file_name="filtered_netflix_content.csv",
    mime="text/csv"
)

st.divider()
st.caption("Netflix Content Segmentation | Built with Python, Pandas, Matplotlib and Streamlit")
