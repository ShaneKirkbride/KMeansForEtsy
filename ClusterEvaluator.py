import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusterEvaluator:
    def __init__(self, data, max_k=10):
        self.data = data
        self.max_k = max_k
        self.inertias = []
        self.silhouette_scores = []

    def calculate_inertia(self):
        """Calculate and plot inertia for a range of k values to find the elbow."""
        self.inertias = []
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.data)
            self.inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_k + 1), self.inertias, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(range(1, self.max_k + 1))
        plt.grid(True)
        plt.show()

    def calculate_silhouette_scores(self):
        """Calculate and plot silhouette scores for a range of k values."""
        # Silhouette scores can't be calculated for k=1 or when k equals the number of data points
        if len(self.data) <= 1:
            print("Not enough data to calculate silhouette scores.")
            return

        self.silhouette_scores = []
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            self.silhouette_scores.append(score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, self.max_k + 1), self.silhouette_scores, marker='o')
        plt.title('Silhouette Score For Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.xticks(range(2, self.max_k + 1))
        plt.grid(True)
        plt.show()