import sys
import math

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self):
        self.s_clusters = []
        self.cluster_sizes = []
        self.centroids = []
        self.max_s_sum = []
        self.min_centroids_dist = []
        self.diameter = 0

    def s(self, X, cluster_k_index, cluster_sizes, labels, centroids):
        sss = 0
        for i in range(0, len(labels)):
            if labels[i] == cluster_k_index:
                sss += utils.euclidian_dist(X[i], self.centroids[cluster_k_index])
        if self.cluster_sizes[cluster_k_index] == 0:
            return float('inf')
        return sss / self.cluster_sizes[cluster_k_index]


    # db_star, DB*-index, min is better
    def find(self, X, labels, n_clusters):
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.max_s_sum = [[sys.float_info.min for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.min_centroids_dist = [[sys.float_info.max for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.s_clusters = [0 for _ in range(n_clusters)]
        self.diameter = utils.find_diameter(X)
        for i in range(n_clusters):
            self.s_clusters[i] = self.s(X, i, self.cluster_sizes, labels, self.centroids)
        numerator = 0.0
        for k in range(0, n_clusters):
            for l in range(k + 1, n_clusters):
                self.max_s_sum[k][l] = self.s_clusters[k] + self.s_clusters[l]
                self.min_centroids_dist[k][l] = utils.euclidian_dist(self.centroids[k], self.centroids[l])

            numerator += np.max(self.max_s_sum[k]) / np.min(self.min_centroids_dist[k])
        return numerator / n_clusters


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        delta = 10**(-math.log(len(X), 10) - 1)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(self.centroids, self.cluster_sizes, point, k, l)
        if utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.s_clusters[k] = self.s(X, k, self.cluster_sizes, labels, self.centroids)
        if utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.s_clusters[l] = self.s(X, l, self.cluster_sizes, labels, self.centroids)
        for i in range(n_clusters):
            if i > k:
                self.max_s_sum[k][i] = self.s_clusters[i] + self.s_clusters[k]
                self.min_centroids_dist[k][i] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
            if i > l:
                self.max_s_sum[l][i] = self.s_clusters[i] + self.s_clusters[l]
                self.min_centroids_dist[l][i] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
        numerator = 0.0
        for i in range(n_clusters):
            numerator += np.max(self.max_s_sum[i]) / np.min(self.min_centroids_dist[i])
        return numerator / n_clusters
