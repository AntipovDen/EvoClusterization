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

    # recalculate without moving points
    def recalculated_measure_C(self, points_to_move, clusters_to_move_to):
        labels_cp = self.labels[:]
        measure = 0

        for i in range(len(points_to_move)):
            point = points_to_move[i]
            cluster = clusters_to_move_to[i]
            measure = self.measure.update(self.X, self.n_clusters, labels_cp, labels_cp[point], cluster, point)

        return measure

    # move points
    def move_points(self, points_to_move, clusters_to_move_to):
        for i in range(len(points_to_move)):
            self.labels[points_to_move[i]] = clusters_to_move_to[i]

    def copy(self):
        return clusterization(self.X.copy(), self.labels.copy(), self.n_clusters, self.measure)


