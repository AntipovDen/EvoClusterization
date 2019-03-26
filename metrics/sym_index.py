# Sym Index:
import heapq
import sys
import math

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure


class Index(Measure):

    def __init__(self, dist_centroids=None, dist_ps=None, centroids=None,
                 cluster_sizes=None, diameter=0):
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        if dist_ps is None:
            dist_ps = []
        if dist_centroids is None:
            dist_centroids = []
        self.dist_centroids = dist_centroids
        self.dist_ps = dist_ps
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.diameter = diameter


    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.dist_centroids = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.dist_ps = [0 for _ in range(len(labels))]
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

        for k in range(0, n_clusters - 1):
            for l in range(k + 1, n_clusters):
                self.dist_centroids[k][l] = utils.euclidian_dist(self.centroids[k], self.centroids[l])
        numerator = np.amax(self.dist_centroids)
        for i in range(0, len(labels)):
            self.dist_ps[i] = utils.d_ps(X, labels, X[i], labels[i], self.centroids)
        denominator = sum(self.dist_ps)
        return -(numerator / (denominator * n_clusters))


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        for i in range(n_clusters):
            if i > k:
                self.dist_centroids[k][i] = utils.euclidian_dist(self.centroids[i], self.centroids[k])
            if i > l:
                self.dist_centroids[l][i] = utils.euclidian_dist(self.centroids[i], self.centroids[l])
        numerator = np.amax(self.dist_centroids)
        delta = 10**(-math.log(len(X), 10) - 1)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
                self.dist_ps[i] = utils.d_ps(X, labels, X[i], labels[i], self.centroids)
        denominator = sum(self.dist_ps)
        return -(numerator / (denominator * n_clusters))
