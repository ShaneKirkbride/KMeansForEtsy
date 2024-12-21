# KMeansForEtsy
This is a unsupervised learning algorithm to group different types of Etsy accounts.
This project is a Python-based tool designed for analyzing and clustering data using 

**KMeans Clustering**. The tool supports data loading, preprocessing, clustering, and visualization, providing an end-to-end solution for cluster analysis.

---

## Features

1. **Configurable File Paths**:
   - File paths are saved in a `config.json` file, enabling persistence across runs.
   - No need to input file paths repeatedly; they are loaded automatically if available.

2. **Data Preprocessing**:
   - Handles missing data, standardizes features, and encodes categorical data for clustering readiness.

3. **KMeans Clustering**:
   - Identifies groupings in data using the KMeans algorithm.
   - Evaluates clustering quality with inertia and silhouette scores.

4. **Visualizations**:
   - Price distribution by cluster.
   - Shop age vs. reviews.
   - Favorites vs. views.
   - Category distribution.
   - Clustered bar charts.
   - Histograms by cluster for numeric fields.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>

## Install Dependencies
pip install -r requirements.txt


# KMeans Clustering: Overview and Implementation
What is KMeans Clustering?
KMeans is an unsupervised machine learning algorithm used for partitioning a dataset into distinct groups (clusters). It is particularly useful for identifying patterns and structures in data without predefined labels.

## How Does KMeans Work?
Initialization:

Choose k (number of clusters).
Initialize k cluster centroids randomly.
Assignment:

Assign each data point to the nearest centroid based on a distance metric (e.g., Euclidean distance).
Update:

Recalculate the centroids by taking the mean of all points assigned to each cluster.
Iterate:

Repeat steps 2 and 3 until centroids stabilize or a maximum number of iterations is reached.
Evaluating Clusters
Inertia:

Measures how tightly data points are clustered around centroids.
Lower inertia indicates better clustering.
Silhouette Score:

Evaluates how similar a data point is to its own cluster (cohesion) compared to other clusters (separation).
Score ranges from -1 to 1, with higher values indicating better-defined clusters.

# Project Workflow
## Load Data:

The DataLoader module reads the input CSV file.
Preprocess Data:

The DataPreprocessor module handles missing values, standardization, and encoding.
Evaluate Clusters:

The ClusterEvaluator module calculates inertia and silhouette scores to determine the optimal number of clusters (k).
Perform Clustering:

The ClusterData module applies KMeans clustering to group the data.
Save Results:

Clustered data and original data with cluster labels are saved as Excel files.
Visualize Results:

The DataVisualizer module generates insightful plots for analysis.

# Configuration File: config.json
The config.json file stores file paths for persistence. Example structure:

`{`
    `"input_file_path": "path/to/your/input.csv",`
    `"output_clustered_file_path": "path/to/save/clustered_data.xlsx",`
    `"output_original_with_clusters_path": "path/to/save/original_data_with_clusters.xlsx"`
`}

# Example Visualizations
Price Distribution by Cluster:

Shows how prices are distributed across clusters.
Shop Age vs. Reviews:

Highlights relationships between shop age and customer reviews.
Favorites vs. Views:

Explores popularity metrics within clusters.

# Dependencies
numpy
pandas
matplotlib
scikit-learn
seaborn
Install these using:

`pip install -r requirements.txt`

# Future Enhancements
Add support for hierarchical clustering.
Enable dynamic determination of k using elbow or silhouette methods.
Enhance visualization capabilities.
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

# License
This project is licensed under the MIT License. See LICENSE for more details.