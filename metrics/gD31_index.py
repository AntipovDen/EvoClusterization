import numpy as np
import sys

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self):
        self.dists = []
        self.dist_same_c = []
        self.delta = []
        self.diameter = 0
        self.centroids = []
        self.cluster_sizes = []

    def find(self, X, labels, n_clusters):
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.diameter = utils.find_diameter(X)
        self.dists = [[0. for _ in range(len(labels))] for _ in range(len(labels))]
        self.dist_same_c = []
        rows, colums = X.shape
        delta_l = [[0.0] * n_clusters] * n_clusters
        self.delta = np.array(delta_l)
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        for i in range(rows - 1):
            for j in range(i + 1, rows):
                self.dists[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists[j][i] = self.dists[i][j]
                if labels[i] != labels[j]:
                    self.delta[labels[i]][labels[j]] += self.dists[i][j]
                else:
                    self.dist_same_c.append([i, j])
                    maximum_same_c = max(self.dists[i][j], maximum_same_c)
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                self.delta[i][j] /= float(self.cluster_sizes[i] * self.cluster_sizes[j])
                if self.delta[i][j] != 0:
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        return - minimum_dif_c / maximum_same_c


    def update(self, X, n_clusters, labels, k, l, id):
        prev_cluster_sizes = list(self.cluster_sizes)
        self.centroids = cluster_centroid.update_centroids(list(self.centroids), list(self.cluster_sizes),
                                                                               X[id], k, l)

        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        delete_from_same = []
        for i in range(0, len(labels)):
            if labels[i] == k:
                delete_from_same.append([i, id])
                delete_from_same.append([id, i])
            if labels[i] == l and i != id:

                self.dist_same_c.append([i, id])
                self.dist_same_c.append([id, i])

        for pair in self.dist_same_c:
            cur = self.dists[pair[0]][pair[1]]
            if cur > maximum_same_c:
                if pair not in delete_from_same:
                    maximum_same_c = cur
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                self.delta[i][j] *= (prev_cluster_sizes[i] * prev_cluster_sizes[j])
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
                self.delta[i][j] /= float(self.cluster_sizes[i] * self.cluster_sizes[j])
                if self.delta[i][j] != 0:
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        return - minimum_dif_c / maximum_same_c
