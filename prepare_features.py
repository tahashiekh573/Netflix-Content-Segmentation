
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==========================================
# TASK 4 - FEATURE PREPARATION
# ==========================================

print("=" * 55)
print("NETFLIX CONTENT SEGMENTATION")
print("FEATURE PREPARATION")
print("=" * 55)


# Load dataset
df = pd.read_csv("Dataset.csv")


# Select useful features
numerical_features = [
    "release_year"
]

categorical_features = [
    "type",
    "country",
    "rating"
]


# Prepare working dataset
features = df[numerical_features + categorical_features].copy()

# Handle missing values safely
features[numerical_features] = features[numerical_features].fillna(
    features[numerical_features].median()
)

for column in categorical_features:
    features[column] = features[column].fillna("Unknown")

print("\nSelected Numerical Features:")
print(numerical_features)

print("\nSelected Categorical Features:")
print(categorical_features)


# Scale numerical features and encode categories
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Transform features
processed_features = preprocessor.fit_transform(features)

# Convert sparse output to a dense array if necessary
if hasattr(processed_features, "toarray"):
    processed_features = processed_features.toarray()

print("\nFeature Preparation Results")
print("-" * 55)
print("Original Records:", len(df))
print("Original Selected Features:", features.shape[1])
print("Encoded Feature Count:", processed_features.shape[1])
print("Processed Data Shape:", processed_features.shape)

print("\nFeature preparation completed successfully.")
