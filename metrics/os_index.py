import math
import heapq

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import numpy as np
from metrics.measure import Measure


class Index(Measure):

    def __init__(self):
        self.centroids = []
        self.cluster_sizes = []
        self.dists = []
        self.a_ss = []
        self.b_ss = []
        self.dists_e = []
        self.diameter = 0
        self.dists_for_b = []
        self.max_b_ss = []
        self.b_ss_size = []

    def a(self, X, labels, i, cluster_k_index):
        acc = 0.0
        for j in range(0, len(labels)):
            if labels[j] != cluster_k_index: continue
            acc += self.dists_e[i][j]
        return acc / self.cluster_sizes[cluster_k_index]


    def b(self, X, labels, i, cluster_k_index):
        self.dists_for_b[i] = [float('inf') for _ in range(len(labels))]
        for j in range(0, len(labels)):
            if labels[j] == cluster_k_index:
                continue
            self.dists_for_b[i][j] = self.dists_e[i][j]

        self.dists_for_b[i].sort()
        min_n = self.dists_for_b[i][:self.cluster_sizes[cluster_k_index]]
        min_n_filtered = [x for x in min_n if x != float('inf')]
        self.max_b_ss[i] = max(min_n_filtered)
        self.b_ss_size[i] = len(min_n_filtered)
        return sum(min_n_filtered) / self.cluster_sizes[cluster_k_index]


    def ov(self, i):
        a_s = self.a_ss[i]
        b_s = self.b_ss[i]
        if (b_s - a_s) / (b_s + a_s) < 0.4:
            return a_s / b_s
        else:
            return 0

    # os, OS-Index, max is better, added -
    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.dists = [[0 for _ in range(len(labels))] for _ in range(n_clusters)]
        self.dists_e = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        self.dists_for_b = [0 for _ in range(len(labels))]
        self.max_b_ss = [0 for _ in range(len(labels))]
        self.b_ss_size = [0 for _ in range(len(labels))]
        for i in range(len(labels)):
            for j in range(len(labels)):
                self.dists_e[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dists_e[j][i] = self.dists_e[i][j]
        self.a_ss = [0 for _ in range(len(labels))]
        self.b_ss = [0 for _ in range(len(labels))]
        for i in range(len(labels)):
            self.a_ss[i] = self.a(X, labels, i, labels[i])
            self.b_ss[i] = self.b(X, labels, i, labels[i])
        numerator = 0.0
        for k in range(n_clusters):
            for i in range(len(labels)):
                if labels[i] != k:
                    continue
                numerator += self.ov(i)
        denominator = 0.0

        for k in range(n_clusters):
            for i in range(len(labels)):
                if labels[i] != k:
                    continue
                self.dists[k][i] = utils.euclidian_dist(X[i], self.centroids[k])
        for k in range(n_clusters):
            # get sum of 0.1*|Ck| largest elements
            acc = 0.0
            max_n = heapq.nlargest(int(math.ceil(0.1 * self.cluster_sizes[k])), self.dists[k])
            for i in range(0, len(max_n)):
                acc += max_n[i]
            denominator += acc * 10.0 / self.cluster_sizes[k]
        return -(numerator / denominator)

    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        prev_cluster_sizes = list(self.cluster_sizes)
        self.centroids = cluster_centroid.update_centroids(self.centroids, self.cluster_sizes, point, k, l)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        prev_dists_for_b = list(self.dists_for_b)
        self.a_ss[id] = self.a(X, labels, id, l)
        self.b_ss[id] = self.b(X, labels, id, l)
        for i in range(len(labels)):
            if i == id:
                continue
            if labels[i] == k:
                self.a_ss[i] *= prev_cluster_sizes[k]
                self.a_ss[i] -= self.dists_e[i][id]
                self.a_ss[i] /= self.cluster_sizes[k]
                self.b_ss[i] *= prev_cluster_sizes[k]
                j = prev_cluster_sizes[k]
                #if prev_dists_for_b[i][j] != float('inf'):
                    #self.b_ss[i] -= prev_dists_for_b[i][j]
                self.dists_for_b[i][id] = self.dists_e[i][id]
                if self.max_b_ss[i] > self.dists_e[i][id]:
                    self.b_ss[i] -= self.max_b_ss[i]
                    self.b_ss[i] += self.dists_e[i][id]
                    filtered = [x for x in self.dists_for_b[i] if x != float('inf')]
                    self.max_b_ss[i] = max(filtered)
                elif self.b_ss_size[i] < self.cluster_sizes[k]:
                    self.b_ss[i] += self.dists_e[i][id]
                    self.b_ss_size[i] += 1
                self.b_ss[i] /= self.cluster_sizes[k]
            if labels[i] == l:
                self.a_ss[i] *= prev_cluster_sizes[l]
                self.a_ss[i] += self.dists_e[i][id]
                self.a_ss[i] /= self.cluster_sizes[l]
                self.b_ss[i] *= prev_cluster_sizes[l]
                j = prev_cluster_sizes[l]
                #self.b_ss[i] += self.dists_e[i][j]
                self.b_ss[i] -= self.dists_e[i][id]
                self.b_ss_size[i] -= 1
                self.dists_for_b[i][id] = float('inf')
                if self.b_ss_size[i] < self.cluster_sizes[l]:
                    filtered = [x for x in self.dists_for_b[i] if x != float('inf')]
                    new_max = max(filtered)
                    self.b_ss[i] += new_max
                    self.b_ss_size[i] += 1
                    self.max_b_ss[i] = new_max
                #if self.dists_for_b[i][j] < self.dists_e[i][id]:
                    #self.b_ss[i] -= self.dists_e[i][id]
                #    self.b_ss[i] += self.dists_for_b[i][j]
                #elif self.dists_for_b[i][j - 1] == float('inf'):

                self.b_ss[i] /= self.cluster_sizes[l]
        numerator = 0.0
        for c in range(n_clusters):
            for i in range(len(labels)):
                if labels[i] != c:
                    continue
                numerator += self.ov(i)
        denominator = 0.0
        self.dists[k][id] = 0.
        #delta = 10**(-math.log(len(X), 10) - 1)
        delta = 10**(-9)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
                self.dists[labels[i]][i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])
        for c in range(n_clusters):
            # get sum of 0.1*|Ck| largest elements
            max_n = heapq.nlargest(int(math.ceil(0.1 * self.cluster_sizes[c])), self.dists[c])
            denominator += sum(max_n) * 10.0 / self.cluster_sizes[c]
        return -(numerator / denominator)



