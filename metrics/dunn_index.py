# Dunn index, max is better, add -
import sys
import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
from metrics.measure import Measure

class Index(Measure):

    def __init__(self, dist=None, dist_dif_c=None, dist_same_c=None,
                 centroids=None, diameter=0):
        if centroids is None:
            centroids = []
        if dist_same_c is None:
            dist_same_c = []
        if dist_dif_c is None:
            dist_dif_c = []
        if dist is None:
            dist = []
        self.dist = dist
        self.dist_dif_c = dist_dif_c
        self.dist_same_c = dist_same_c
        self.centroids = centroids
        self.diameter = diameter

    def find(self, X, labels, n_clusters):
        self.diameter = utils.find_diameter(X)
        self.centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
        rows, colums = X.shape
        self.dist = [[0. for _ in range(rows)] for _ in range(rows)]
        self.dist_dif_c = []
        self.dist_same_c = []
        minimum_dif_c = sys.float_info.max  # min self.dist in different clusters
        maximum_same_c = sys.float_info.min  # max self.dist in the same cluster
        for i in range(rows - 1):
            for j in range(i + 1, rows):
                self.dist[i][j] = utils.euclidian_dist(X[i], X[j])
                self.dist[j][i] = self.dist[i][j]
                if labels[i] != labels[j]:
                    self.dist_dif_c.append([i, j])
                    minimum_dif_c = min(self.dist[i][j], minimum_dif_c)
                else:
                    self.dist_same_c.append([i, j])
                    maximum_same_c = max(self.dist[i][j], maximum_same_c)
        return minimum_dif_c / maximum_same_c


    def update_dunn(self, X, n_clusters, labels, k, l, id):
        minimum_dif_c = sys.float_info.max  # min self.dist in different clusters
        maximum_same_c = sys.float_info.min  # max self.dist in the same cluster
        delete_from_dif = []
        delete_from_same = []
        for i in range(len(labels)):
            if i == id:
                continue
            if labels[i] == k:
                self.dist_dif_c.append([i, id])
                delete_from_same.append([i, id])
                self.dist_dif_c.append([id, i])
                delete_from_same.append([id, i])
            if labels[i] == l and i != id:
                delete_from_dif.append([i, id])
                self.dist_same_c.append([i, id])
                delete_from_dif.append([id, i])
                self.dist_same_c.append([id, i])
        for pair in self.dist_dif_c:
            cur = self.dist[pair[0]][pair[1]]
            if cur < minimum_dif_c:
                if pair not in delete_from_dif:
                    minimum_dif_c = cur

        for pair in self.dist_same_c:
            cur = self.dist[pair[0]][pair[1]]
            if cur > maximum_same_c:
                if pair not in delete_from_same:
                    maximum_same_c = cur
        return minimum_dif_c / maximum_same_c




