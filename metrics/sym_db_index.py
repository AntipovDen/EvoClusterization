# SymDB, Sym Davies-Bouldin index, min is better
import heapq
import sys
import math

import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, dist_ps=None, sym_s_clusters=None,
                 fractions=None, diameter=0):
        if fractions is None:
            fractions = []
        if sym_s_clusters is None:
            sym_s_clusters = []
        if dist_ps is None:
            dist_ps = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.dist_ps = dist_ps
        self.sym_s_clusters = sym_s_clusters
        self.fractions = fractions
        self.diameter = diameter

    def sym_s(self, X, labels, cluster_k_index, cluster_sizes, centroids):
        acc = 0.0
        for i in range(0, len(labels)):
            if labels[i] != cluster_k_index: continue
            acc += self.dist_ps[i]
        return acc / float(self.cluster_sizes[cluster_k_index])



    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.dist_ps = [0 for _ in range(len(labels))]
        self.sym_s_clusters = [0 for _ in range(n_clusters)]
        self.fractions = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        db = 0
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        for i in range(0, len(labels)):
            self.dist_ps[i] = utils.d_ps(X, labels, X[i], labels[i], self.centroids)
        for i in range(n_clusters):
            self.sym_s_clusters[i] = self.sym_s(X, labels, i, self.cluster_sizes, self.centroids)
        for k in range(0, n_clusters):
            for l in range(0, n_clusters):
                if k != l:
                    self.fractions[k][l] = ((self.sym_s_clusters[k] + self.sym_s_clusters[l]) /
                                      utils.euclidian_dist(self.centroids[k], self.centroids[l]))
        for k in range(n_clusters):
            max_fraction = np.amax(self.fractions[k])
            db += max_fraction
        db /= float(n_clusters)
        return -db


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        delta = 10**(-math.log(len(X), 10) - 1)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
                self.dist_ps[i] = utils.d_ps(X, labels, X[i], labels[i], self.centroids)
        if utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.sym_s_clusters[k] = self.sym_s(X, labels, k, self.cluster_sizes, self.centroids)
        if utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.sym_s_clusters[l] = self.sym_s(X, labels, l, self.cluster_sizes, self.centroids)
        db = 0
        for i in range(n_clusters):
            if i != k:
                tm = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                self.fractions[i][k] = (self.sym_s_clusters[i] + self.sym_s_clusters[k]) / tm
                self.fractions[k][i] = self.fractions[i][k]
            if i != l:
                tm = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                self.fractions[i][l] = (self.sym_s_clusters[i] + self.sym_s_clusters[l]) / tm
                self.fractions[l][i] = self.fractions[i][l]
        for i in range(n_clusters):
            tmp = np.amax(self.fractions[i])
            db += tmp
        db /= float(n_clusters)
        return -db
