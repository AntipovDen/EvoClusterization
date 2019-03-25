import numpy as np


def cluster_centroid(X, labels, n_clusters):
    rows, colums = X.shape
    #print(n_clusters)
    center = [[0.0] * colums] * n_clusters
    centroid = np.array(center)
    num_points = [0] * n_clusters
    for i in range(0, rows):
        c = labels[i]
        for j in range(0, colums):
            #print(centroid[c])
            #print(c, j)
            centroid[c][j] += X[i][j]
        num_points[c] += 1
    for i in range(0, n_clusters):
        for j in range(0, colums):
            centroid[i][j] /= num_points[i]
    return centroid

def update_centroids(centroid, num_points, point, k, l):
    #for j in range(len(point)):
    #    centroid[k][j] -= (point[j] / num_points[k])
    #    centroid[l][j] += (point[j] / num_points[l])
    #num_points[k] -= 1
    #num_points[l] += 1
    for j in range(len(point)):
        centroid[k][j] *= (num_points[k] + 1)
        centroid[k][j] -= point[j]
        if num_points[k] != 0:
            centroid[k][j] /= num_points[k]
        centroid[l][j] *= (num_points[l] - 1)
        centroid[l][j] += point[j]
        centroid[l][j] /= num_points[l]
    return centroid


def count_cluster_sizes(labels, n_clusters):
    point_in_c = [0 for _ in range(n_clusters)]
    #print(len(point_in_c))
    for i in range(len(labels)):
        point_in_c[labels[i]] += 1
    return point_in_c

