import numpy as np

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
import math
from metrics.measure import Measure


class Index(Measure):

    def __init__(self):
        self.centroids = []
        self.cluster_sizes = []
        self.sigmas = []
        self.normed_sigma_x = 0
        self.diameter = 0
        self.dens = 0

    def f(self, x_i, centroid_k, std):
        if std < utils.euclidian_dist(x_i, centroid_k):
            return 0
        else:
            return 1


    def mean(self, x_i, x_j):
        return (x_i + x_j) / 2


    def den2(self, X, labels, centroids, k, l, std):
        acc = 0.0
        elements = len(X)
        for i in range(0, elements):
            if labels[i] == k or labels[i] == l:
                acc += self.f(X[i], self.mean(self.centroids[k], self.centroids[l]), std)
        return acc


    def den1(self, X, labels, centroids, k, std):
        acc = 0.0
        elements = len(X)
        for i in range(0, elements):
            if labels[i] == k:
                acc += self.f(X[i], self.centroids[k], std)
        return acc


    def normed_sigma(self, X):
        elements = len(X)
        sum = 0.0
        for i in range(0, elements):
            sum += X[i]
        avg = sum / elements
        sigma = 0.0

        for i in range(0, elements):
            sigma += (X[i] - avg) * (X[i] - avg)
        sigma /= elements
        return math.sqrt(np.dot(sigma, np.transpose(sigma)))


    def normed_cluster_sigma(self, X, labels, k):
        elements = len(X)
        sum = 0.0
        ck_size = 0
        for i in range(0, elements):
            if labels[i] == k:
                sum += X[i]
                ck_size += 1
        avg = sum / elements
        sigma = 0.0

        for i in range(0, elements):
            if labels[i] == k:
                sigma += (X[i] - avg) * (X[i] - avg)
        sigma /= elements
        return math.sqrt(np.dot(sigma, np.transpose(sigma)))


    def stdev(self, n_clusters):
        sum = 0.0
        for k in range(n_clusters):
            sum += self.sigmas[k]
        sum = math.sqrt(sum)
        sum /= n_clusters
        return sum


    # s_dbw, S_Dbw index, min is better
    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.sigmas = [0 for _ in range(n_clusters)]
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

        for k in range(0, n_clusters):
            self.sigmas[k] = self.normed_cluster_sigma(X, labels, k)
        self.normed_sigma_x = self.normed_sigma(X)
        term1 = sum(self.sigmas) / (n_clusters * self.normed_sigma_x)
        stdev_val = self.stdev(n_clusters)

        self.dens = 0.0
        for k in range(0, n_clusters):
            for l in range(0, n_clusters):
                self.dens += self.den2(X, labels, self.centroids, k, l, stdev_val) /\
                            max(self.den1(X, labels, self.centroids, k, stdev_val),
                                self.den1(X, labels, self.centroids, l, stdev_val))

        self.dens /= n_clusters * (n_clusters - 1)
        return (term1 + self.dens)


    def update(self, X, n_clusters, labels, k, l, id):
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        delta = 10**(-math.log(len(X), 10) - 1)
        if utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.sigmas[k] = self.normed_cluster_sigma(X, labels, k)
        if utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.sigmas[l] = self.normed_cluster_sigma(X, labels, l)
        term1 = sum(self.sigmas) / (n_clusters * self.normed_sigma_x)
        stdev_val = self.stdev(n_clusters)

        if (utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
           or utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
            self.dens = 0.0
            for k in range(0, n_clusters):
                for l in range(0, n_clusters):
                    self.dens += self.den2(X, labels, self.centroids, k, l, stdev_val) /\
                            max(self.den1(X, labels, self.centroids, k, stdev_val),
                                self.den1(X, labels, self.centroids, l, stdev_val))

        self.dens /= n_clusters * (n_clusters - 1)
        return (term1 + self.dens)
