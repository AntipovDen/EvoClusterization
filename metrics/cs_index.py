import sys

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, dists=None,
                 centroids_dist=None, diameter=0):
        if centroids_dist is None:
            centroids_dist = []
        if dists is None:
            dists = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.dists = dists
        self.centroids_dist = centroids_dist
        self.diameter = diameter


    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        elements, ignore_columns = X.shape
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

        self.dists = [[0 for _ in range(elements)] for _ in range(elements)]

        for i in range(0, elements - 1):  # for every element
            for j in range(i + 1, elements):  # for every other
                if labels[i] != labels[j]: continue  # if they are in the same cluster
                # update the distance to the farthest element in the same cluster
                self.dists[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists[j][i] = self.dists[i][j]

        # max_self.dists contain for each element the farthest the his cluster

        numerator = 0.0
        for i in range(0, elements):
            max_dist = np.amax(self.dists[i])
            numerator += max_dist / self.cluster_sizes[labels[i]]

        denominator = 0.0
        self.centroids_dist = [[sys.float_info.max for _ in range(n_clusters)] for _ in range(n_clusters)]
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.centroids_dist[i][j] = utils.euclidian_dist(self.centroids[i], self.centroids[j])
        for i in range(n_clusters):
            min_centroid_dist = np.amin(self.centroids_dist[i])
            denominator += min_centroid_dist

        return numerator / denominator

    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        elements, ignore_columns = X.shape
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids),
                                                           np.copy(self.cluster_sizes), point, k, l)
        for i in range(elements):
            if labels[i] == k:
                self.dists[i][id] = 0
                self.dists[id][i] = 0
            if labels[i] == l:
                self.dists[id][i] = utils.euclidian_dist(X[i], X[id])
                self.dists[i][id] = self.dists[id][i]

        numerator = 0.0
        for i in range(elements):
            max_dist = np.amax(self.dists[i])
            numerator += max_dist / self.cluster_sizes[labels[i]]

        denominator = 0.0
        for i in range(n_clusters):
            if i != k:
                self.centroids_dist[k][i] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                self.centroids_dist[i][k] = self.centroids_dist[k][i]
            if i != l:
                self.centroids_dist[l][i] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                self.centroids_dist[i][l] = self.centroids_dist[l][i]

        for i in range(n_clusters):
            min_centroid_dist = np.amin(self.centroids_dist[i])
            denominator += min_centroid_dist

        return numerator / denominator

