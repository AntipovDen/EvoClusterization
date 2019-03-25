import math
import heapq

def euclidian_dist(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return math.sqrt(sum)

def d_ps(X, labels, x_i, cluster_k_index, centroids):
    centroid = centroids[cluster_k_index]
    dists = []
    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index:
            continue
        dists.append(euclidian_dist(centroid + centroid - x_i, X[j]))
    mins = heapq.nsmallest(2, dists)
    if len(mins) < 2:
        return float('inf')
    return (mins[0] + mins[1]) / 2.0


def find_diameter(X):
    ch = grahamscan(X)
    max_diam = 0
    for i in range(len(ch) - 1):
        for j in range(i + 1, len(ch)):
            max_diam = max(euclidian_dist(ch[i], ch[j]), max_diam)
    return max_diam



def rotate(A, B, C):
    return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])


def grahamscan(points):
    n = len(points)
    p = [i for i in range(n)]
    for i in range(1, n):
        if points[p[i]][0] < points[p[0]][0]:
            p[i], p[0] = p[0], p[i]
    for i in range(2, n):
        j = i
        while j > 1 and rotate(points[p[0]], points[p[j - 1]], points[p[j]]) < 0:
            p[j], p[j - 1] = p[j - 1], p[j]
            j -= 1
    s = [p[0], p[1]]
    for i in range(2, n):
        while rotate(points[s[-2]], points[s[-1]], points[p[i]]) < 0:
            s = s[:-1]
        s.append(p[i])
    return [points[i] for i in s]



