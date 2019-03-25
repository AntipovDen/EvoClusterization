import sys

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import math
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, centers=None, sums=None,
                 diameter=0, centroid_dists=None):
        if centroid_dists is None:
            centroid_dists = []
        if sums is None:
            sums = []
        if centers is None:
            centers = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.centers = centers
        self.sums = sums
        self.diameter = diameter
        self.centroid_dists = centroid_dists

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        rows, colums = X.shape
        self.sums = [0 for _ in range(n_clusters)]
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        centres_l = [[sys.float_info.max] * n_clusters] * n_clusters
        self.centers = np.array(centres_l)
        self.centroid_dists = [0 for _ in range(len(labels))]
        # self.centroid_dists = [utils.euclidian_dist(X[i], self.centroids[labels[i]]) for i in range(len(X))]
        for i in range(len(labels)):
            self.centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
            self.sums[labels[i]] += self.centroid_dists[i]
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.centers[i][j] = utils.euclidian_dist(self.centroids[i], self.centroids[j])
        for i in range(rows):
            for j in range(rows):
                if labels[i] != labels[j]:
                    dist = self.centers[labels[i]][labels[j]]
                    minimum_dif_c = min(dist, minimum_dif_c)

        denominator = list(self.sums)
        for i in range(n_clusters):
            denominator[i] *= (2 / self.cluster_sizes[i])

        return minimum_dif_c / max(denominator)

    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        # prev_cluster_sizes = list(self.cluster_sizes)
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, np.copy(labels))
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        # update denominator

        new_centroid_dists = list(self.centroid_dists)
        dell = 10 ** (-math.log(len(X), 10) - 1)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > dell * self.diameter
                    or labels[i] == l and utils.euclidian_dist(prev_centroids[l],
                                                               self.centroids[l]) > dell * self.diameter):
                new_centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
        new_sums = [0 for _ in range(n_clusters)]
        for i in range(n_clusters):
            if i != k and i != l:
                new_sums[i] = self.sums[i]
        for i in range(len(labels)):
            if labels[i] == k or labels[i] == l:
                new_sums[labels[i]] += new_centroid_dists[i]
        denominator = list(new_sums)
        for i in range(n_clusters):
            if self.cluster_sizes[i] != 0:
                denominator[i] *= (2 / self.cluster_sizes[i])

        # update numerator

        for i in range(n_clusters):
            if i != k:
                self.centers[i][k] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                self.centers[k][i] = self.centers[i][k]
            if i != l:
                self.centers[i][l] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                self.centers[l][i] = self.centers[i][l]

        minimum_dif_c = np.amin(self.centers)
        return minimum_dif_c / max(denominator)
