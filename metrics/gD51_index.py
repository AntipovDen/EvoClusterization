import sys
import math

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import numpy as np
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, dists_same_c=None, delta=None, dists=None,
                 centroid_dists=None, sums=None, diameter=0):
        if sums is None:
            sums = []
        if centroid_dists is None:
            centroid_dists = []
        if dists is None:
            dists = []
        if delta is None:
            delta = []
        if dists_same_c is None:
            dists_same_c = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.dists_same_c = dists_same_c
        self.delta = delta
        self.dists = dists
        self.centroid_dists = centroid_dists
        self.sums = sums
        self.diameter = diameter

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.dists = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        self.centroid_dists = [0 for _ in range(len(labels))]
        self.delta = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        self.sums = [0 for _ in range(n_clusters)]
        for i in range(len(labels)):
            self.centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
            self.sums[labels[i]] += self.centroid_dists[i]
        for i in range(len(labels) - 1):
            for j in range(i + 1, len(labels)):
                self.dists[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists[j][i] = self.dists[i][j]
                self.dists_same_c.append([i, j])
                maximum_same_c = max(self.dists[i][j], maximum_same_c)
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.delta[i][j] = (self.sums[i] + self.sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        return -(minimum_dif_c / maximum_same_c)


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_cluster_sizes = list(self.cluster_sizes)
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        delete_from_same = []

        #update denominator

        for i in range(0, len(labels)):
            if labels[i] == k:
                delete_from_same.append([i, id])
                delete_from_same.append([id, i])
            if labels[i] == l and i != id:
                self.dists_same_c.append([i, id])
                self.dists_same_c.append([id, i])

        for pair in self.dists_same_c:
            cur = self.dists[pair[0]][pair[1]]
            if cur > maximum_same_c:
                if pair not in delete_from_same:
                    maximum_same_c = cur

        #update numerator

        new_centroid_dists = list(self.centroid_dists)
        dell = 10**(-math.log(len(X), 10) - 1)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > dell * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > dell * self.diameter):
                new_centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])

        for i in range(n_clusters):
            for j in range(n_clusters):
                self.delta[i][j] *= (prev_cluster_sizes[i] + prev_cluster_sizes[j])

        new_sums = [0 for _ in range(n_clusters)]
        for i in range(n_clusters):
            if i != k and i != l:
                new_sums[i] = self.sums[i]
        for i in range(len(labels)):
            if labels[i] == k or labels[i] == l:
                new_sums[labels[i]] += new_centroid_dists[i]

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.delta[i][j] = (new_sums[i] + new_sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        return -(minimum_dif_c / maximum_same_c)

