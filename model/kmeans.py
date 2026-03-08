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


if __name__ == '__main__':
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(k=2, max_iterations=1000, distance=euclid_distance)
    print(X.shape)
    print(X)
    kmeans._init_random_centroids(X)