import numpy as np
import math

import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, centroids=None, cluster_sizes=None, x_center=0, numerator=None,
                 denominator=None, diameter=0):
        if denominator is None:
            denominator = []
        if numerator is None:
            numerator = []
        if cluster_sizes is None:
            cluster_sizes = []
        if centroids is None:
            centroids = []
        self.centroids = centroids
        self.cluster_sizes = cluster_sizes
        self.x_center = x_center
        self.numerator = numerator
        self.denominator = denominator
        self.diameter = diameter


    def find(self, X, labels, n_clusters):
        self.x_center = np.mean(X, axis=0)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        self.diameter = utils.find_diameter(X)
        rows, colums = X.shape

        ch = float(rows - n_clusters) / float(n_clusters - 1)

        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.numerator = [0 for _ in range(n_clusters)]
        for i in range(0, n_clusters):
            self.numerator[i] = self.cluster_sizes[i] * utils.euclidian_dist(self.centroids[i], self.x_center)
        self.denominator = [0 for _ in range(len(labels))]
        for i in range(len(labels)):
            self.denominator[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])

        ch *= sum(self.numerator)
        ch /= sum(self.denominator)
        return ch


    def update(self, X, n_clusters, labels, k, l, id):
        delta = 10**(-math.log(len(X), 10) - 1)
        point = X[id]
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = cluster_centroid.count_cluster_sizes(n_clusters, labels)
        self.centroids = cluster_centroid.update_centroids(np.copy(self.centroids), np.copy(self.cluster_sizes), point, k, l)
        #if 0 in self.cluster_sizes:
        #    print("!!! ", self.cluster_sizes)
        #    print("I'll go find with ", labels, n_clusters - 1)
        #    return self.find(X, labels, n_clusters - 1)
        ch = float(len(labels) - n_clusters) / float(n_clusters - 1)
        self.numerator[k] = self.cluster_sizes[k] * utils.euclidian_dist(self.centroids[k], self.x_center)
        self.numerator[l] = self.cluster_sizes[l] * utils.euclidian_dist(self.centroids[l], self.x_center)
        for i in range(len(labels)):
            if (labels[i] == k and utils.euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter
               or labels[i] == l and utils.euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter):
                self.denominator[i] = utils.euclidian_dist(X[i], self.centroids[labels[i]])

        ch *= sum(self.numerator)
        ch /= sum(self.denominator)
        return ch
