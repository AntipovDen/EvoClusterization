from time import time
from numpy import partition, sum
from random import sample
from numpy.random import choice
from math import ceil
from metrics import davies_bouldin
from metrics import cluster_centroid
from metrics.utils import euclidian_dist
import sys

class clusterization:

    X = []
    labels = []
    n_clusters = 0
    #measure = davies_bouldin.Index

    def __init__(self, X, labels, n_clusters, measure=None):
        self.X = X
        self.labels = labels
        self.n_clusters = n_clusters
        if measure is None:
            self.measure = davies_bouldin.Index #let it be by default
        else:
            self.measure = measure

    def init_measure(self):
        return self.measure.find(self.X, self.labels, self.n_clusters)

    def get_nearest_centroids(self):
        row, column = self.X.shape
        centroids_numbers, centroid_distances = [], []

        for i in range(row):
            centroid_distances.append(sys.float_info.max)
            centroids_numbers.append(0)

        default_centroids = cluster_centroid.cluster_centroid(self.X, self.labels, self.n_clusters)

        for i in range(len(self.X)):
            for j in range(len(default_centroids)):
                distance = euclidian_dist(self.X[i], default_centroids[j])
                if (distance <= centroid_distances[j]):
                    centroid_distances[i] = distance
                    centroids_numbers[i] = j

        return centroids_numbers, centroid_distances

    def recalculated_measure(self, point_to_move, number_of_new_cluster):
        return self.measure.update(self.X, self.n_clusters, self.labels, self.labels[point_to_move], number_of_new_cluster, point_to_move)

    def copy(self):
        return clusterization(self.X.copy(), self.labels.copy(), self.n_clusters, self.measure)


