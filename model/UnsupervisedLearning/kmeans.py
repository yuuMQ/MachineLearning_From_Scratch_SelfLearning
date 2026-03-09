import numpy as np
import pandas as pd
from utils.data_operation import euclid_distance, manhattan_distance



class KMeans():
    def __init__(self, k=2, max_iterations=1000, distance=euclid_distance):
        self.k = k # The number of clusters
        self.max_iterations = max_iterations
        self.distance = distance # Default euclid_distance | manhattan_distance or can define any other distance algorithms.

    def _init_random_centroids(self, X):
        # Create random centroids for k-means clustering
        n_samples, n_features = X.shape # Samples: row, Features: Column
        print(f'n_samples: {n_samples}, n_features: {n_features}')

        centroids = np.zeros((self.k, n_features)) # Array list of k random centroid
        print(f'shape of centroids: {centroids.shape}')

        for i in range(self.k):
            # each centroid has the same shape with each sample of X
            centroid = X[np.random.choice(range(n_samples))] # Random sample of X to create random centroid
            print(f'random centroid: {centroid}')
            centroids[i] = centroid

        return centroids

    def _closest_centroid(self, sample, centroids):
        # Return index of the closest centroid to the sample (centroid is in array list of centroids -> return index of centroids)
        closest_centroid_index = 0
        closest_distance = float('inf')
        for i, centroid in enumerate(centroids):
            dist = self.distance(sample, centroid) # Calculate distance
            if dist < closest_distance:
                closest_centroid_index = i
                closest_distance = dist

        return closest_centroid_index # centroids[closest_centroid_index]

    def _create_cluster(self, X, centroids):
        # Assign the samples to the closest centroids
        # Each cluster is a array list of sample
        n_samples = X.shape[0] # X (n_samples, n_features)
        clusters = [[] for _ in range(self.k)] # initialize k empty cluster | clusters is a array list of cluster

        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids) # index of closest centroid to each sample
            clusters[centroid_i].append(sample_i)
        print(f'clusters: {clusters}')
        return clusters

    def _calculate_centroids(self, X, clusters):
        # Calculate centroid as a mean of the samples in each cluster
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0) # 0: sample | 1: feature
            centroids[i] = centroid # update centroid
        print(f'update centroids: {centroids}')
        return centroids

    def _get_cluster_label(self, X, clusters):
        # Classify samples as the index of the cluster
        y_pred = np.zeros(X.shape[0]) # y_pred is the array list of prediction for each sample of samples
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i # each index of y_pred will be cluster index of each sample -> the label
        return y_pred

    def predict(self, X):
        # 1. Initialize random centroids
        centroids = self._init_random_centroids(X)

        # 2. Loop util convergence or for max iterations
        for _ in range(self.max_iterations):
            # create clusters with the centroids
            clusters = self._create_cluster(X, centroids)
            prev_centroids = centroids

            # Calculate centroid with mean of cluster's samples
            centroids = self._calculate_centroids(X, clusters)

            # If no centroids changed -> convergence (hội tụ)
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self._get_cluster_label(X, clusters)


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(k=2, max_iterations=1000, distance=euclid_distance)
    # print(X.shape)
    # print(X)
    # centroids = kmeans._init_random_centroids(X)
    # clusters = kmeans._create_cluster(X, centroids)
    # update_clusters = kmeans._calculate_centroids(X, clusters)
    kmeans.predict(X)