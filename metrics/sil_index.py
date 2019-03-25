import sys

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import numpy as np
from metrics.measure import Measure


class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, a_ss=None, b_ss=None,
                 dists_e=None, dists_for_b=None, diameter=0):
        if dists_for_b is None:
            dists_for_b = []
        if dists_e is None:
            dists_e = []
        if b_ss is None:
            b_ss = []
        if a_ss is None:
            a_ss = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.a_ss = a_ss
        self.b_ss = b_ss
        self.dists_e = dists_e
        self.dists_for_b = dists_for_b
        self.diameter = diameter

    def a(self, X, labels, i, cluster_k_index):
        acc = 0.0
        for j in range(0, len(labels)):
            if labels[j] != cluster_k_index: continue
            acc += self.dists_e[i][j]
        if self.cluster_sizes[cluster_k_index] == 0:
            return float('inf')
        return acc / self.cluster_sizes[cluster_k_index]

    def b(self, X, n_clusters, labels, i, cluster_k_index):
        self.dists_for_b[i] = [sys.float_info.max for _ in range(n_clusters)]
        for c in range(n_clusters):
            if c == cluster_k_index:
                continue
            ssum = 0
            for j in range(len(labels)):
                if labels[j] != c:
                    continue
                ssum += self.dists_e[i][j]
            if self.cluster_sizes[c] == 0:
                self.dists_for_b[i][c] = float('inf')
            else:
                self.dists_for_b[i][c] = ssum / self.cluster_sizes[c]

        return min(self.dists_for_b[i])


    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.a_ss = [0 for _ in range(len(labels))]
        self.b_ss = [0 for _ in range(len(labels))]
        self.dists_e = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        self.dists_for_b = [0 for _ in range(len(labels))]
        for i in range(len(labels)):
            for j in range(len(labels)):
                self.dists_e[i][j] = utils.euclidian_dist(X[i], X[j])
        for i in range(len(labels)):
            self.a_ss[i] = self.a(X, labels, i, labels[i])
            self.b_ss[i] = self.b(X, n_clusters, labels, i, labels[i])

        ch = 0
        for i in range(len(labels)):
            ch += (self.b_ss[i] - self.a_ss[i]) / max(self.b_ss[i], self.a_ss[i])
        return ch / float(len(labels))


    def update(self, X, n_clusters, labels, k, l, id):
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), X[id], k, l)
        self.a_ss[id] = self.a(X, labels, id, l)
        self.b_ss[id] = self.b(X, n_clusters, labels, id, l)
        for i in range(len(labels)):
            if i == id:
                continue
            if labels[i] == k:
                self.a_ss[i] *= (self.cluster_sizes[k] + 1)
                self.a_ss[i] -= self.dists_e[i][id]
                if self.cluster_sizes[k] == 0:
                    self.a_ss[i] = float('inf')
                else:
                    self.a_ss[i] /= self.cluster_sizes[k]
            if labels[i] == l:
                self.a_ss[i] *= (self.cluster_sizes[l] - 1)
                self.a_ss[i] += self.dists_e[i][id]
                if self.cluster_sizes[l] == 0:
                    self.a_ss[i] = float('inf')
                else:
                    self.a_ss[i] /= self.cluster_sizes[l]
            self.dists_for_b[i][l] *= (self.cluster_sizes[l] - 1)
            self.dists_for_b[i][l] += self.dists_e[i][id]
            if self.cluster_sizes[l] == 0:
                self.dists_for_b[i][l] = float('inf')
            else:
                self.dists_for_b[i][l] /= self.cluster_sizes[l]
            self.dists_for_b[i][k] *= (self.cluster_sizes[k] + 1)
            self.dists_for_b[i][k] -= self.dists_e[i][id]
            if self.cluster_sizes[k] == 0:
                self.dists_for_b[i][k] = float('inf')
            else:
                self.dists_for_b[i][k] /= self.cluster_sizes[k]
            self.b_ss[i] = min(self.dists_for_b[i])

        ch = 0
        for i in range(len(labels)):
            ch += (self.b_ss[i] - self.a_ss[i]) / max(self.b_ss[i], self.a_ss[i])
        return ch / float(len(labels))
