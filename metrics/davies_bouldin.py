import numpy as np
import math
import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self):
        self.s_clusters = []
        self.points_in_clusters = []
        self.centroids = []
        self.sums = []
        self.diameter = 0

    def s(self, X, cluster_k_index, cluster_sizes, labels, centroids):
        sss = 0
        for i in range(0, len(labels)):
            if labels[i] == cluster_k_index:
                sss += utils.euclidian_dist(X[i], centroids[cluster_k_index])
        if cluster_sizes[cluster_k_index] == 0:
            return float('inf')
        return sss / cluster_sizes[cluster_k_index]


    def count_cluster_sizes(self, labels, n_clusters):
        point_in_c = [0] * n_clusters
        for i in range(0, len(labels)):
            point_in_c[labels[i]] += 1
        return point_in_c


    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.s_clusters = [0. for _ in range(n_clusters)]
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        db = 0
        self.points_in_clusters = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        for i in range(n_clusters):
            self.s_clusters[i] = self.s(X, i, self.points_in_clusters, labels, self.centroids)
        self.sums = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        for i in range(0, n_clusters):
            for j in range(0, n_clusters):
                if i != j:
                    tm = utils.euclidian_dist(self.centroids[i], self.centroids[j])
                    if tm != 0:
                        self.sums[i][j] = (self.s_clusters[i] + self.s_clusters[j]) / tm
                    else:
                        pass
                        #a = -Constants.bad_cluster
            tmp = np.amax(self.sums[i])
            db += tmp
        db /= float(n_clusters)
        return db


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        delta = 10 ** (-math.log(len(X), 10) - 1)
        prev_centroids = np.copy(self.centroids)
        self.centroids, self.points_in_clusters = cluster_centroid.update_centroids(self.centroids, self.points_in_clusters, point, k, l)
        if utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.s_clusters[k] = self.s(X, k, self.points_in_clusters, labels, self.centroids)
        if utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.s_clusters[l] = self.s(X, l, self.points_in_clusters, labels, self.centroids)
        db = 0
        for i in range(n_clusters):
            if i != k:
                tm = utils.euclidian_dist(self.centroids[i], self.centroids[k])
                if tm != 0:
                    self.sums[i][k] = (self.s_clusters[i] + self.s_clusters[k]) / tm
                    self.sums[k][i] = (self.s_clusters[i] + self.s_clusters[k]) / tm
            if i != l:
                tm = utils.euclidian_dist(self.centroids[i], self.centroids[l])
                if tm != 0:
                    self.sums[i][l] = (self.s_clusters[i] + self.s_clusters[l]) / tm
                    self.sums[l][i] = (self.s_clusters[i] + self.s_clusters[l]) / tm
        for i in range(n_clusters):
            tmp = np.amax(self.sums[i])
            db += tmp
        db /= float(n_clusters)
        return db

