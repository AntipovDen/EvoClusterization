import sys

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, numerators=None,
                 inner_max_dists=None, outer_min_dists=None, accumulator=None, diameter=0):
        if accumulator is None:
            accumulator = []
        if outer_min_dists is None:
            outer_min_dists = []
        if inner_max_dists is None:
            inner_max_dists = []
        if numerators is None:
            numerators = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = np.copy(centroids)
        self.cluster_sizes = np.copy(cluster_sizes)
        self.numerators = np.copy(numerators)
        self.inner_max_dists = np.copy(inner_max_dists)
        self.outer_min_dists = np.copy(outer_min_dists)
        self.accumulator = np.copy(accumulator)
        self.diameter = diameter

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.numerators = [0.0] * n_clusters
        for i in range(0, len(labels)):
            self.numerators[labels[i]] += utils.euclidian_dist(X[i], self.centroids[labels[i]])

        self.inner_max_dists = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        self.outer_min_dists = [[sys.float_info.max for _ in range(len(labels))] for _ in range(n_clusters)]
        self.accumulator = [0 for _ in range(n_clusters)]
        for k in range(0, n_clusters):
            for i in range(len(labels)):  # iterate elements outside cluster
                if labels[i] == k:
                    continue
                for j in range(len(labels)):  # iterate inside cluster
                    if labels[j] != k:
                        continue
                    self.inner_max_dists[i][j] = utils.euclidian_dist(X[i], X[j])
                    self.inner_max_dists[j][i] = self.inner_max_dists[i][j]

        for c in range(n_clusters):
            for i in range(len(labels)):
                if labels[i] == c:
                    continue
                inner_max_dist = 0
                for j in range(len(self.inner_max_dists[i])):
                    if labels[j] == c:
                        inner_max_dist = max(inner_max_dist, self.inner_max_dists[i][j])
                if inner_max_dist != 0:
                    self.outer_min_dists[c][i] = inner_max_dist
            outer_min_dist = np.amin(self.outer_min_dists[c])
            self.accumulator[c] = self.numerators[c] / outer_min_dist
        return -sum(self.accumulator) / len(labels)


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        self.numerators[k] = 0.0
        self.numerators[l] = 0.0
        for i in range(len(labels)):
            if labels[i] == k or labels[i] == l:
                self.numerators[labels[i]] += utils.euclidian_dist(X[i], self.centroids[labels[i]])
        for i in range(len(labels)):
            if labels[i] == k:
                self.inner_max_dists[i][id] = utils.euclidian_dist(X[i], X[id])
                self.inner_max_dists[id][i] = self.inner_max_dists[i][id]
            if labels[i] == l:
                self.inner_max_dists[i][id] = 0
                self.inner_max_dists[id][i] = 0
                self.outer_min_dists[l][id] = sys.float_info.max
        for c in [k, l]:
            for i in range(len(labels)):
                if labels[i] == c:
                    continue
                inner_max_dist = 0
                for j in range(len(self.inner_max_dists[i])):
                    if labels[j] == c:
                        inner_max_dist = max(inner_max_dist, self.inner_max_dists[i][j])

                if inner_max_dist != 0:
                    self.outer_min_dists[c][i] = inner_max_dist
            outer_min_dist = np.amin(self.outer_min_dists[c])
            self.accumulator[c] = self.numerators[c] / outer_min_dist
        return -sum(self.accumulator) / len(labels)
