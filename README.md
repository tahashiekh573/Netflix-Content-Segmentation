# Netflix Content Segmentation

## Project Overview

This project analyzes Netflix movies and TV shows using machine learning clustering techniques. The goal is to identify groups of content based on content type, country, release year, and audience rating.

The project uses **K-Means Clustering** to segment Netflix titles and visualizes the results to understand patterns in the dataset.

## Objectives

- Explore and analyze the Netflix dataset.
- Clean and prepare features for machine learning.
- Apply K-Means clustering to group similar content.
- Evaluate clustering performance using the Silhouette Score and Inertia.
- Visualize cluster distributions and content characteristics.
- Summarize the main characteristics of each cluster.

## Dataset

**File:** `Dataset.csv`

The dataset contains 8,790 Netflix titles and 10 columns.

Important features include:

- `show_id` — Unique identifier for each title.
- `type` — Movie or TV Show.
- `title` — Title of the content.
- `director` — Director of the content.
- `country` — Country associated with the content.
- `date_added` — Date the title was added to Netflix.
- `release_year` — Original release year.
- `rating` — Content rating.
- `duration` — Movie duration or number of seasons.
- `listed_in` — Genre or category information.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- NumPy

## Machine Learning Workflow

1. **Data Inspection:** Examined the dataset's dimensions, columns, missing values, and duplicate records.
2. **Feature Preparation:** Selected `release_year`, `type`, `country`, and `rating` for clustering.
3. **Preprocessing:** Scaled the numerical feature and applied one-hot encoding to categorical features.
4. **Model Training:** Applied K-Means clustering with different values of K.
5. **Model Evaluation:** Compared Inertia and Silhouette Scores for K values from 2 to 8.
6. **Cluster Analysis:** Examined cluster sizes, content types, release years, ratings, and countries.
7. **Visualization:** Generated charts to communicate the clustering results.

## Cluster Evaluation Results

The model was evaluated using K values from 2 through 8.

| Number of Clusters | Silhouette Score |
|---:|---:|
| 2 | 0.4036 |
| 3 | 0.2191 |
| 4 | 0.1754 |
| 5 | 0.1785 |
| 6 | 0.2012 |
| 7 | 0.2157 |
| 8 | 0.2199 |

**Best evaluated result:** K = 2, with a Silhouette Score of 0.4036.

The two-cluster configuration achieved the highest Silhouette Score among the tested values. A separate four-cluster model was also explored to provide more detailed content segmentation.

## Project Visualizations

The project generates the following visualizations:

- `elbow_curve.png` — Inertia across different cluster counts.
- `silhouette_comparison.png` — Silhouette Score comparison for K = 2 to 8.
- `cluster_distribution.png` — Number of titles in each cluster.
- `content_type_by_cluster.png` — Movie and TV Show distribution across clusters.
- `average_release_year.png` — Average release year for each cluster.
- `rating_distribution.png` — Content rating distribution across clusters.

## Project Files

- `Dataset.csv` — Original dataset.
- `check_dataset.py` — Dataset inspection.
- `prepare_features.py` — Feature preparation and encoding.
- `train_clusters.py` — K-Means model training.
- `analyze_clusters.py` — Cluster analysis and summary generation.
- `visualize_clusters.py` — Cluster visualization.
- `evaluate_clusters.py` — Clustering evaluation.
- `clustered_netflix.csv` — Dataset containing cluster assignments.
- `cluster_summary.csv` — Summary statistics for the clusters.
- `cluster_evaluation.csv` — Inertia and Silhouette Scores.
- `elbow_curve.png` — Elbow chart.
- `silhouette_comparison.png` — Silhouette comparison chart.
- `cluster_distribution.png` — Cluster size chart.
- `content_type_by_cluster.png` — Content type chart.
- `average_release_year.png` — Average release year chart.
- `rating_distribution.png` — Rating distribution chart.

## How to Run the Project

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Place the dataset

Place `Dataset.csv` in the project directory.

### 3. Run the scripts

Execute the scripts from the project directory:

```bash
python check_dataset.py
python prepare_features.py
python train_clusters.py
python analyze_clusters.py
python visualize_clusters.py
python evaluate_clusters.py
```

## Conclusion

This project demonstrates how unsupervised machine learning can be used to explore and segment Netflix content. K-Means clustering identifies groups based on selected metadata, while evaluation metrics and visualizations help assess and interpret the results.

The highest Silhouette Score among the tested configurations was achieved with two clusters. The four-cluster analysis provides an additional, more detailed view of the dataset.

## Future Improvements

- Include genre and duration features in the clustering process.
- Experiment with alternative clustering algorithms.
- Investigate different feature combinations and scaling methods.
- Improve the treatment of missing or unspecified metadata.
- Compare cluster quality across multiple evaluation metrics.
