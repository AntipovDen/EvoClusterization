import heapq

import numpy as np
import math
import metrics.utils as utils
import metrics.cluster_centroid as cluster_centroid
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, diameter=0, cluster_sizes=None,
                 distances=None, s_c=0, n_w=0, s_min=None, s_max=None):
        if s_max is None:
            s_max = []
        if s_min is None:
            s_min = []
        if distances is None:
            distances = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.diameter = diameter
        self.cluster_sizes = cluster_sizes
        self.distances = distances
        self.s_c = s_c
        self.n_w = n_w
        self.s_min = s_min
        self.s_max = s_max

    def find(self, X, labels, n_clusters):
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.diameter = utils.find_diameter(X)
        self.cluster_sizes = []
        self.distances = []
        self.s_c = 0
        self.n_w = 0
        rows, colums = X.shape
        for i in range(rows - 1):
            for j in range(i + 1, rows):
                if labels[i] == labels[j]:
                    self.s_c += utils.euclidian_dist(X[i], X[j])
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)

        for k in range(0, n_clusters):
            self.n_w += self.cluster_sizes[k] * (self.cluster_sizes[k] - 1) / 2

        for i in range(0, len(labels) - 1):
            for j in range(i + 1, len(labels)):
                self.distances.append(utils.euclidian_dist(X[i], X[j]))

        self.s_min = heapq.nsmallest(int(self.n_w), self.distances)
        self.s_max = heapq.nlargest(int(self.n_w), self.distances)

        #ones = [1] * int(self.n_w)
        #s_min_c = np.dot(self.s_min, np.transpose(ones))
        #s_max_c = np.dot(self.s_max, np.transpose(ones))
        s_min_c = sum(self.s_min)
        s_max_c = sum(self.s_max)
        return -(self.s_c - s_min_c) / (s_max_c - s_min_c)


    def update(self, X, n_clusters, labels, k, l, id):
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes),
                                                                                 X[id], k, l)
        #self.cluster_sizes[k] -= 1
        #self.cluster_sizes[l] += 1
        for i in range(len(labels)):
            if labels[i] == k:
                self.s_c -= utils.euclidian_dist(X[i], X[id])
            if labels[i] == l:
                self.s_c += utils.euclidian_dist(X[i], X[id])
        prev_n_w = self.n_w
        self.n_w = self.n_w - (self.cluster_sizes[k] + 1) * self.cluster_sizes[k] / 2 + self.cluster_sizes[k] * (self.cluster_sizes[k] - 1) / 2 \
                - (self.cluster_sizes[l] - 1) * (self.cluster_sizes[l] - 2) / 2 + self.cluster_sizes[l] * (self.cluster_sizes[l] - 1) / 2

        delta = 0.1
        print(prev_n_w)
        print(self.n_w)
        print(delta * len(labels))

        if abs(self.n_w - prev_n_w) > delta * len(labels):
            self.s_min = heapq.nsmallest(int(self.n_w), self.distances)
            self.s_max = heapq.nlargest(int(self.n_w), self.distances)

        #ones = [1] * int(self.n_w)
        #s_min_c = np.dot(self.s_min, np.transpose(ones))
        #s_max_c = np.dot(self.s_max, np.transpose(ones))
        s_min_c = sum(self.s_min)
        s_max_c = sum(self.s_max)
        return -(self.s_c - s_min_c) / (s_max_c - s_min_c)

