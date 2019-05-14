import numpy as np


class Kmeans:

    def __init__(self, K, max_iter, x):
        self.K = K
        self.max_iter = max_iter
        self.x = x
        new_x = np.random.RandomState(seed=3).permutation(self.x)
        self.centroids = new_x[:K]


    def compute_centroids(self, labels):
        self.centroids = np.zeros((self.K, self.x.shape[1]))
        for i in range(self.K):
            self.centroids[i, :] = np.mean(self.x[labels == i, :], axis=0)


    def compute_distances(self, y=None):
        distances = np.zeros((self.x.shape[0], self.K))
        if y != None:
            distances = np.zeros((y.shape[0], self.K))
        for i in range(self.K):
            row_norm = np.linalg.norm(self.x - self.centroids[i, :], axis=1)
            if y != None:
                row_norm = np.linalg.norm(y - self.centroids[i, :], axis=1)
            distances[:, i] = np.square(row_norm)
        return distances


    def find_cluster(self, distances):
        return np.argmin(distances, axis=1)


    def sum_square_error(self, labels):
        distances = np.zeros(self.x.shape[0])
        for i in range(self.K):
            distances[labels == i] = np.linalg.norm(self.x[labels == i] - self.centroids[i], axis=1)
        return np.sum(np.square(distances))

    
    def fit(self):
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distances = self.compute_distances()
            print(distances)
            labels = self.find_cluster(distances)
            print(labels)
            self.compute_centroids(labels)
            if np.all(old_centroids == self.centroids):
                break

        self.error = self.sum_square_error(labels)

    
    def predict(self, y):
        distances = self.compute_distances(self.centroids, y)
        return self.find_cluster(distances)
