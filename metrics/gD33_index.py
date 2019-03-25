import numpy as np
import sys
import math

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure


class Index(Measure):
    def __init__(self, centroids=None, dists=None, delta=None, point_in_c=None,
                 sums=None, diameter=0, centroid_dists=None):
        if centroid_dists is None:
            centroid_dists = []
        if sums is None:
            sums = []
        if point_in_c is None:
            point_in_c = []
        if delta is None:
            delta = []
        if dists is None:
            dists = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.dists = dists
        self.delta = delta
        self.point_in_c = point_in_c
        self.sums = sums
        self.diameter = diameter
        self.centroid_dists = centroid_dists

    def find(self, X, labels, n_clusters):
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.dists = [[0. for _ in range(len(labels))] for _ in range(len(labels))]
        self.sums = [0 for _ in range(n_clusters)]
        rows, colums = X.shape
        self.point_in_c = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.delta_l = [[0.0] * n_clusters] * n_clusters
        self.delta = np.array(self.delta_l)
        self.centroid_dists = [0 for _ in range(len(labels))]
        #self.centroid_dists = [utils.euclidian_dist(X[i], self.centroids[labels[i]]) for i in range(len(X))]
        minimum_dif_c = sys.float_info.max
        for i in range(len(labels)):
            self.centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
            self.sums[labels[i]] += self.centroid_dists[i]
        for i in range(rows - 1):
            for j in range(i + 1, rows):
                self.dists[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists[j][i] = self.dists[i][j]
                if labels[i] != labels[j]:
                    self.delta[labels[i]][labels[j]] += self.dists[i][j]
        for i in range(n_clusters):
            for j in range(n_clusters):
                self.delta[i][j] /= float(self.point_in_c[i] * self.point_in_c[j])
                if self.delta[i][j] != 0:
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
            self.sums[i] *= (2 / self.point_in_c[i])
        print(max(self.sums))
        return minimum_dif_c / max(self.sums)


    def update(self, X, n_clusters, labels, k, l, id):
        self.diameter = utils.find_diameter(X)
        prev_point_in_c = list(self.point_in_c)
        prev_centroids = np.copy(self.centroids)
        self.point_in_c = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.point_in_c), X[id], k, l)
        minimum_dif_c = sys.float_info.max  # min dist in different clusters

        #update numerator

        for i in range(n_clusters):
            for j in range(n_clusters):
                self.delta[i][j] *= (prev_point_in_c[i] * prev_point_in_c[j])

        for i in range(len(labels)):
            if labels[i] != k and id < i:
                self.delta[k][labels[i]] -= self.dists[id][i]
            if labels[i] != k and id > i:
                self.delta[labels[i]][k] -= self.dists[i][id]
            if labels[i] != l and id < i:
                self.delta[l][labels[i]] += self.dists[id][i]
            if labels[i] != l and id > i:
                self.delta[labels[i]][l] += self.dists[i][id]

        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                self.delta[i][j] /= float(self.point_in_c[i] * self.point_in_c[j])
                if self.delta[i][j] != 0:
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
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
            if self.point_in_c[i] != 0:
                denominator[i] *= (2 / self.point_in_c[i])
        return minimum_dif_c / max(denominator)
