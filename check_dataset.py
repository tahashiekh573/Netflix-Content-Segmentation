
import pandas as pd


# ==============================
# Dataset Analysis
# ==============================

print("=" * 55)
print("TASK 4 - NETFLIX CONTENT SEGMENTATION")
print("DATASET ANALYSIS")
print("=" * 55)

# Load dataset
df = pd.read_csv("Dataset.csv")


# ==============================
# Basic Information
# ==============================

print("\nDataset Information")
print("-" * 55)

print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())


# ==============================
# Missing Values
# ==============================

print("\nMissing Values")
print("-" * 55)

print(df.isnull().sum())


# ==============================
# Duplicate Records
# ==============================

print("\nDuplicate Records")
print("-" * 55)

print("Total Duplicates:", df.duplicated().sum())


# ==============================
# Content Type Distribution
# ==============================

print("\nContent Type Distribution")
print("-" * 55)

print(df["type"].value_counts())


# ==============================
# Release Year Analysis
# ==============================

print("\nRelease Year Information")
print("-" * 55)

print("Earliest Release Year:", df["release_year"].min())
print("Latest Release Year:", df["release_year"].max())


# ==============================
# Dataset Preview
# ==============================

print("\nDataset Preview")
print("-" * 55)

print(
    df[
        [
            "title",
            "type",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in"
        ]
    ].head()
)

print("\nDataset analysis completed successfully.")
