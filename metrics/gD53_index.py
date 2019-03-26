import sys
import math

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import numpy as np
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, delta=None, centroid_dists=None,
                 sums=None, diameter=0):
        if sums is None:
            sums = []
        if centroid_dists is None:
            centroid_dists = []
        if delta is None:
            delta = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.delta = delta
        self.centroid_dists = centroid_dists
        self.sums = sums
        self.diameter = diameter

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroid_dists = [0 for _ in range(len(labels))]
        self.delta = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        minimum_dif_c = sys.float_info.max  # min dist in different clusters
        self.sums = [0 for _ in range(n_clusters)]
        for i in range(len(labels)):
            self.centroid_dists[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
            self.sums[labels[i]] += self.centroid_dists[i]
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    self.delta[i][j] = (self.sums[i] + self.sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])
        denominator = list(self.sums)
        #print(denominator)
        for i in range(n_clusters):
            denominator[i] *= (2 / self.cluster_sizes[i])

        return -(minimum_dif_c / max(denominator))

    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_cluster_sizes = list(self.cluster_sizes)
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(np.copy(labels), n_clusters)
        self.centroids = cluster_centroid.update_centroids(self.centroids, self.cluster_sizes, point, k, l)
        minimum_dif_c = sys.float_info.max  # min dist in different clusters

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
                    if self.cluster_sizes[i] + self.cluster_sizes[j] == 0:
                        self.delta[i][j] = float('inf')
                    else:
                        self.delta[i][j] = (new_sums[i] + new_sums[j]) / float(self.cluster_sizes[i] + self.cluster_sizes[j])
                    minimum_dif_c = min(minimum_dif_c, self.delta[i][j])


        #update denominator
        denominator = list(new_sums)
        #print(denominator)
        for i in range(n_clusters):
            if self.cluster_sizes[i] == 0:
                denominator[i] = float('inf')
            else:
                denominator[i] *= (2 / self.cluster_sizes[i])

        return -(minimum_dif_c / max(denominator))

