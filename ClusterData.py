from sklearn.cluster import KMeans
import numpy as np

class ClusterData:
    def __init__(self, data, num_clusters=5):
        self.data = data
        self.num_clusters = num_clusters
        self.model = KMeans(n_clusters=self.num_clusters, random_state=0)

    def perform_clustering(self):
        # Fit the model and predict clusters
        cluster_labels = self.model.fit_predict(self.data.select_dtypes(include=[np.number]))  # Only numeric data for KMeans
        self.data['Cluster'] = cluster_labels  # Add cluster labels to the original data
        return self.data
