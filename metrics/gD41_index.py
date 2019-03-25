import sys

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, diameter=0, centroids=None, cluster_sizes=None,
                 dist_same_c=None, dists=None, centers=None):
        if centers is None:
            centers = []
        if dists is None:
            dists = []
        if dist_same_c is None:
            dist_same_c = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.diameter = diameter
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.dist_same_c = dist_same_c
        self.dists = dists
        self.centers = centers

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.dist_same_c = []
        rows, colums = X.shape
        self.dists = [[0. for _ in range(rows)] for _ in range(rows)]
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        centres_l = [[sys.float_info.max] * n_clusters] * n_clusters
        self.centers = np.array(centres_l)
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.centers[i][j] = utils.euclidian_dist(self.centroids[i], self.centroids[j])

        for i in range(rows - 1):
            for j in range(i + 1, rows):
                self.dists[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists[j][i] = self.dists[i][j]
                if labels[i] != labels[j]:
                    dist = self.centers[labels[i]][labels[j]]
                    minimum_dif_c = min(dist, minimum_dif_c)
                else:
                    self.dist_same_c.append([i, j])
                    maximum_same_c = max(self.dists[i][j], maximum_same_c)
        return minimum_dif_c / maximum_same_c


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        maximum_same_c = sys.float_info.min  # max dist in the same cluster
        delete_from_same = []

        #update denominator

        for i in range(len(labels)):
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

        #update numerator

        for i in range(n_clusters):
            if i != k:
                self.centers[i][k] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                self.centers[k][i] = self.centers[i][k]
            if i != l:
                self.centers[i][l] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                self.centers[l][i] = self.centers[i][l]
        minimum_dif_c = np.amin(self.centers)
        return minimum_dif_c / maximum_same_c

