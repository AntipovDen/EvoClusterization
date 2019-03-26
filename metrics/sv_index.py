# sv, SV-Index, max is better, added -
import heapq
import sys
import math

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, clusters_sizes=None, centroid_dists=None,
                 dists=None, diameter=0):
        if dists is None:
            dists = []
        if centroid_dists is None:
            centroid_dists = []
        if clusters_sizes is None:
            clusters_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = clusters_sizes
        self.centroid_dists = centroid_dists
        self.dists = dists
        self.diameter = diameter

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

        self.centroid_dists = [[sys.float_info.max for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.dists = [[0 for _ in range(len(labels))] for _ in range(n_clusters)]
        numerator = 0.0
        for k in range(0, n_clusters - 1):
            for l in range(k + 1, n_clusters):
                self.centroid_dists[k][l] = utils.euclidian_dist(self.centroids[k], self.centroids[l])
                self.centroid_dists[l][k] = self.centroid_dists[k][l]
        for i in range(n_clusters):
            min_dist = np.amin(self.centroid_dists[i])
            numerator += min_dist
        denominator = 0.0

        for k in range(n_clusters):
            for i in range(len(labels)):
                if labels[i] != k:
                    continue
                self.dists[k][i] = utils.euclidian_dist(X[i], self.centroids[k])
        for k in range(n_clusters):
            # get sum of 0.1*|Ck| largest elements
            acc = 0.0
            max_n = heapq.nlargest(int(math.ceil(0.1 * self.cluster_sizes[k])), self.dists[k])
            for i in range(0, len(max_n)):
                acc += max_n[i]
            denominator += acc * 10.0 / self.cluster_sizes[k]
        return -(numerator / denominator)

    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        for i in range(n_clusters):
            if i > k:
                self.centroid_dists[k][i] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                self.centroid_dists[i][k] = self.centroid_dists[k][i]
            if i > l:
                self.centroid_dists[l][i] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                self.centroid_dists[i][l] = self.centroid_dists[l][i]
        numerator = 0.0
        for i in range(n_clusters):
            min_dist = np.amin(self.centroid_dists[i])
            numerator += min_dist
        denominator = 0.0
        self.dists[k][id] = 0.
        delta = 10**(-math.log(len(X), 10) - 1)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
                self.dists[labels[i]][i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
        for c in range(n_clusters):
            # get sum of 0.1*|Ck| largest elements
            acc = 0.0
            max_n = heapq.nlargest(int(math.ceil(0.1 * self.cluster_sizes[c])), self.dists[c])
            for i in range(0, len(max_n)):
                acc += max_n[i]
            denominator += acc * 10.0 / self.cluster_sizes[c]
        return -(numerator / denominator)


