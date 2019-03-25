import heapq
import math
import sys
import numpy as np
from sklearn import metrics
import metrics.cluster_centroid as cluster_centroid
import metrics.utils as utils
'''
import Constants

def metric(X, n_clusters, labels, metric):
    if (n_clusters == 1):
        return Constants.bad_cluster
    if (Constants.dunn_metric in metric):
        dun = dunn(X, labels)
        return dun
    if (Constants.cal_har_metric in metric):
        ch = calinski_harabasz(X, n_clusters, labels)
        return ch
    if (Constants.silhouette_metric in metric):
        sc = silhoette(X, labels)  # [-1, 1]
        return sc
    if (Constants.davies_bouldin_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        db = davies_bouldin(X, n_clusters, labels, centroids)
        return db
    if (Constants.dunn31_metric in metric):
        gd31 = dunn31(X, labels, n_clusters)
        return gd31
    if (Constants.dunn41_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd41 = dunn41(X, labels, n_clusters, centroids)
        return gd41
    if (Constants.dunn51_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd51 = dunn51(X, labels, n_clusters, centroids)
        return gd51
    if (Constants.dunn33_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd33 = dunn33(X, labels, n_clusters, centroids)
        return gd33
    if (Constants.dunn43_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd43 = dunn43(X, labels, n_clusters, centroids)
        return gd43
    if (Constants.dunn53_metric in metric):
        centroids = cluster_centroid(X, labels, n_clusters)
        gd53 = dunn53(X, labels, n_clusters, centroids)
        return gd53
    if (Constants.gamma_metric in metric):
        g = gamma(X, labels, n_clusters)
        return g
    if (Constants.cs_metric in metric):
        cs = cs_index(X, labels, n_clusters)
        return cs
    if (Constants.db_star_metric in metric):
        dbs = db_star_index(X, labels, n_clusters)
        return dbs
    if (Constants.sf_metric in metric):
        sf_score = sf(X, labels, n_clusters)
        return sf_score
    if (Constants.sym_metric in metric):
        sym_score = sym(X, labels, n_clusters)
        return sym_score
    if (Constants.cop_metric in metric):
        cop_score = cop(X, labels, n_clusters)
        return cop_score
    if (Constants.sv_metric in metric):
        sv_score = sv(X, labels, n_clusters)
        return sv_score
    if (Constants.os_metric in metric):
        os_score = os(X, labels, n_clusters)
        return os_score
    if (Constants.sym_bd_metric in metric):
        sym_db_score = sym_db(X, labels, n_clusters)
        return sym_db_score
    if (Constants.s_dbw_metric in metric):
        s_dbw_score = s_dbw(X, labels, n_clusters)
        return s_dbw_score
    if (Constants.c_ind_metric in metric):
        c_ind_score = c_ind(X, labels, n_clusters)
        return c_ind_score
    return 100.0
'''

# Dunn index, max is better, add -
def dunn(X, labels):
    rows, colums = X.shape
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            dist = utils.euclidian_dist(X[i], X[j])
            if (labels[i] != labels[j]):
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                maximum_same_c = max(dist, maximum_same_c)
    return - minimum_dif_c / maximum_same_c


# Calinski-Harabasz index, max is better, add -
def calinski_harabasz(X, n_clusters, labels):
    '''
    xlabels = [0 for _ in range(len(X))]
    x_center = cluster_centroid(X, xlabels, 1)[0]
    centroids = cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape

    ch = float(rows - n_clusters) / float(n_clusters - 1)

    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
         point_in_c[labels[i]] += 1

    sum = 0
    for i in range(0, n_clusters):
        sum += point_in_c[i] * euclidian_dist(centroids[i], x_center)

    sum_div = 0
    for i in range(0, rows):
        sum_div += euclidian_dist(X.iloc[i], centroids[labels[i]])

    ch *= float(sum)
    ch /= float(sum_div)
    return ch
    '''
    return -metrics.calinski_harabaz_score(X, labels)


# Silhouette Coefficient, max is better, [-1, 1], add -
def silhoette(X, labels):
    return -metrics.silhouette_score(X, labels, metric='euclidean')


# Not used anywhere.
# Will be removed soon.
def binomial_coeff(x, y):
    if y == x:
        return 1
    elif y == 1:
        return 1
    elif y > x:
        return 1
    else:
        a = math.factorial(x)
        b = math.factorial(y)
        c = math.factorial(x - y)
        div = a // (b + c)
        return div


# C-index, min is better
def c_ind(X, labels, n_clusters):
    rows, colums = X.shape
    s_c = 0
    for i in range(0, rows):
        for j in range(0, int(math.ceil(float(rows) / 2.0))):
            s_c += utils.euclidian_dist(X[i], X[j])
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

    n_w = 0
    for k in range(0, n_clusters):
        n_w += cluster_sizes[k] * (cluster_sizes[k] - 1) / 2

    distances = []
    for i in range(0, len(labels)-1):
        for j in range(i+1, len(labels)):
            distances.append(utils.euclidian_dist(X[i], X[j]))

    s_min = heapq.nsmallest(int(n_w), distances)
    s_max = heapq.nlargest(int(n_w), distances)

    ones = [1] * int(n_w)
    s_min_c = np.dot(s_min, np.transpose(ones))
    s_max_c = np.dot(s_max, np.transpose(ones))
    # TODO check dot product correct
    return (s_c - s_min_c) / (s_max_c - s_min_c)


def s(X, cluster_k_index, cluster_sizes, labels, centroids):
    sss = 0
    for i in range(0, len(labels)):
        if (labels[i] == cluster_k_index):
            sss += utils.euclidian_dist(X[i], centroids[cluster_k_index])
    return sss / cluster_sizes[cluster_k_index]



# Davies-Bouldin index, min is better
def davies_bouldin(X, n_clusters, labels):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    db = 0
    point_in_c = cluster_centroid.count_cluster_sizes(labels, n_clusters)
    tmp = sys.float_info.min
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            if i != j:
                tm = utils.euclidian_dist(centroids[i], centroids[j])
                if tm != 0:
                    a = (s(X, i, point_in_c, labels, centroids)
                         + s(X, j, point_in_c, labels, centroids)) / tm
                else:
                    pass
                    #a = -Constants.bad_cluster
                tmp = max(tmp, a)
        db += tmp
    db /= float(n_clusters)
    return db


# gD31, Dunn index, max is better, add -
def dunn31(X, labels, n_clusters):
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, rows):
        for j in range(0, rows):
            dist = utils.euclidian_dist(X[i], X[j])
            if labels[i] != labels[j]:
                delta[labels[i]][labels[j]] += dist
            else:
                maximum_same_c = max(dist, maximum_same_c)
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            delta[i][j] /= float(point_in_c[i] * point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD41, Dunn index, max is better, add -
def dunn41(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    centres_l = [[0.0] * n_clusters] * n_clusters
    centers = np.array(centres_l)
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            centers[i][j] = utils.euclidian_dist(centroids[i], centroids[j])

    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                dist = centers[labels[i]][labels[j]]
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                dist = utils.euclidian_dist(X[i], X[j])
                maximum_same_c = max(dist, maximum_same_c)
    return - minimum_dif_c / maximum_same_c


# gD51, Dunn index, max is better, add -
def dunn51(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                delta[labels[i]][labels[j]] += utils.euclidian_dist(X[i], centroids[labels[i]]) +\
                                               utils.euclidian_dist(X[j], centroids[labels[j]])
            else:
                dist = utils.euclidian_dist(X[i], X[j])
                maximum_same_c = max(dist, maximum_same_c)
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            delta[i][j] /= float(point_in_c[i] + point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD33, Dunn index, max is better, add -
def dunn33(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    dl = [0.0] * n_clusters
    d = np.array(dl)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, rows):
        for j in range(0, rows):
            dist = utils.euclidian_dist(X[i], X[j])
            if labels[i] != labels[j]:
                delta[labels[i]][labels[j]] += dist
            else:
                d[labels[i]] += utils.euclidian_dist(X[i], centroids[labels[i]])
    for i in range(0, n_clusters):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
        for j in range(0, n_clusters):
            delta[i][j] /= float(point_in_c[i] * point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# gD43, Dunn index, max is better, add -
def dunn43(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    dl = [0.0] * n_clusters
    d = np.array(dl)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    centres_l = [[0.0] * n_clusters] * n_clusters
    centers = np.array(centres_l)
    for i in range(0, n_clusters):
        for j in range(0, n_clusters):
            centers[i][j] = utils.euclidian_dist(centroids[i], centroids[j])

    for i in range(0, rows):
        for j in range(0, rows):
            if labels[i] != labels[j]:
                dist = centers[labels[i]][labels[j]]
                minimum_dif_c = min(dist, minimum_dif_c)
            else:
                d[labels[i]] += utils.euclidian_dist(X[i], centroids[labels[i]])

    for i in range(0, n_clusters):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
    return - minimum_dif_c / maximum_same_c


# gD53, Dunn index, max is better, add -
def dunn53(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    rows, colums = X.shape
    dl = [0.0] * n_clusters
    d = np.array(dl)
    point_in_c = [0] * n_clusters
    for i in range(0, len(labels)):
        point_in_c[labels[i]] += 1
    delta_l = [[0.0] * n_clusters] * n_clusters
    delta = np.array(delta_l)
    minimum_dif_c = sys.float_info.max  # min dist in different clusters
    maximum_same_c = sys.float_info.min  # max dist in the same cluster
    for i in range(0, int(math.ceil(float(rows) / 2.0))):
        for j in range(0, rows):
            if (labels[i] != labels[j]):
                delta[labels[i]][labels[j]] += (utils.euclidian_dist(X[i], centroids[labels[i]]) +
                                               utils.euclidian_dist(X[j], centroids[labels[j]]))
            else:
                d[labels[i]] += utils.euclidian_dist(X[i], centroids[labels[i]])

    for i in range(0, n_clusters):
        d[i] /= point_in_c[i]
        d[i] += 2.0
        maximum_same_c = max(d[i], maximum_same_c)
        for j in range(0, n_clusters):
            delta[i][j] /= float(point_in_c[i] + point_in_c[j])
            minimum_dif_c = min(minimum_dif_c, delta[i][j])
    return - minimum_dif_c / maximum_same_c


# Gamma index:
# TODO: may require adding explicit casts from int to double
def nW(n_clusters, cluster_sizes):
    result = 0.0
    for i in range(0, n_clusters):
        num = cluster_sizes[i]
        if num > 2:
            result += num * (num - 1.0) / 2.0
    return result


def dl(X, labels, distance, n_clusters):
    result = 0

    for k in range(0, n_clusters - 1):
        for l in range(k + 1, n_clusters):
            if labels[k] == labels[l]: continue
            # x_k and x_l different clusters:
            if utils.euclidian_dist(X[k], X[l]) < distance:
                result += 1
    return result


# gamma, gamma index, mim is better
def gamma(X, labels, n_clusters):
    numerator = 0.0
    elements, ignore_columns = X.shape

    for c_k in range(0, n_clusters):
        for i in range(0, elements - 1):
            if labels[i] != c_k: continue
            for j in range(i + 1, elements):
                if labels[j] != c_k: continue
                # x_i and x_j in c_k:
                distance = utils.euclidian_dist(X[i], X[j])
                numerator += dl(X, labels, distance, n_clusters)

    N = elements
    c_n_2 = (N * (N - 1)) / 2.0

    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
    nw = nW(n_clusters, cluster_sizes)

    return numerator / nw * (c_n_2 - nw)


# cs, CS-index, min is better
def cs_index(X, labels, n_clusters):
    elements, ignore_columns = X.shape
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
    max_dists = [sys.float_info.min] * elements

    for i in range(0, elements):  # for every element
        for j in range(i, elements - 1):  # for every other
            if labels[i] != labels[j]: continue  # if they are in the same cluster
            # update the distance to the farthest element in the same cluster
            max_dists[i] = max(max_dists[i], utils.euclidian_dist(X[i], X[j]))

    # max_dists contain for each element the farthest the his cluster

    numerator = 0.0
    for i in range(0, elements):
        numerator += max_dists[i] / cluster_sizes[labels[i]]

    denominator = 0.0
    for i in range(0, n_clusters):
        min_centroids_dist = sys.float_info.max
        for j in range(i + 1, n_clusters):
            min_centroids_dist = min(utils.euclidian_dist(centroids[i], centroids[j]), min_centroids_dist)
        denominator += min_centroids_dist

    assert denominator != 0.0
    return numerator / denominator


# db_star, DB*-index, min is better
def db_star_index(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
    numerator = 0.0
    for k in range(0, n_clusters):
        max_s_sum = sys.float_info.min
        min_centroids_dist = sys.float_info.max
        for l in range(k + 1, n_clusters):
            max_s_sum = max(max_s_sum,
                            s(X, k, cluster_sizes, labels, centroids)
                            + s(X, l, cluster_sizes, labels, centroids))
            min_centroids_dist = min(min_centroids_dist, utils.euclidian_dist(centroids[k], centroids[l]))
        numerator += max_s_sum / min_centroids_dist
    return numerator / n_clusters


# Score Function:
def bcd_score(X, labels, n_clusters, centroids, cluster_sizes):
    mean_x = np.mean(X, axis=0)
    numerator = 0.0
    for k in range(0, n_clusters):
        numerator += cluster_sizes[k] * utils.euclidian_dist(centroids[k], mean_x)
    return numerator / len(labels) / n_clusters


def wcd_score(X, labels, n_clusters, centroids, cluster_sizes):
    numerator = 0.0
    for i in range(0, len(labels)):
        numerator += utils.euclidian_dist(X[i], centroids[labels[i]]) / cluster_sizes[labels[i]]
    return numerator


# sf, Score Function, max is better, added -
def sf(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

    bcd = bcd_score(X, labels, n_clusters, centroids, cluster_sizes)
    wcd = wcd_score(X, labels, n_clusters, centroids, cluster_sizes)
    p = math.exp(- bcd - wcd) #?????

    return - (1.0 - 1.0 / math.exp(p))


# Sym Index:
def d_ps(X, labels, x_i, cluster_k_index, centroids):
    min1 = sys.float_info.max
    min2 = sys.float_info.max
    centroid = centroids[cluster_k_index]

    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index:
            continue
        t = utils.euclidian_dist(centroid + centroid - x_i, X[j])  # TODO: debug if addition is per coordinate
        if t < min1:
            min2 = min1
            min1 = t

    return (min1 + min2) / 2.0


# sym, Sym Index, max is better, added -
def sym(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)

    numerator = sys.float_info.min
    for k in range(0, n_clusters - 1):
        for l in range(k, n_clusters):
            numerator = max(numerator, utils.euclidian_dist(centroids[k], centroids[l]))

    denominator = 0.0
    for i in range(0, len(labels)):
        denominator += d_ps(X, labels, X[i], labels[i], centroids)
    return -(numerator / denominator / n_clusters)


# cop, COP Index, min is better
def cop(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    numerators = [0.0] * n_clusters
    for i in range(0, len(labels)):
        numerators[labels[i]] += utils.euclidian_dist(X[i], centroids[labels[i]])

    accumulator = 0.0
    for k in range(0, n_clusters):
        outer_min_dist = sys.float_info.max
        for i in range(0, len(labels)):  # iterate elements outside cluster
            if labels[i] == k: continue
            inner_max_dist = sys.float_info.min
            for j in range(i, len(labels)):  # iterate inside cluster
                if labels[j] != k: continue
                inner_max_dist = max(inner_max_dist, utils.euclidian_dist(X[i], X[j]))
            if inner_max_dist != sys.float_info.min:
                # TODO: there are cases, when inner_max_dist is not updated in iner loop. why?
                outer_min_dist = min(outer_min_dist, inner_max_dist)
        accumulator += numerators[k] / outer_min_dist
    return accumulator / len(labels)


# sv, SV-Index, max is better, added -
def sv(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters - 1):
        min_dist = sys.float_info.max
        for l in range(k + 1, n_clusters):
            min_dist = min(min_dist, utils.euclidian_dist(centroids[k], centroids[l]))
        numerator += min_dist

    denominator = 0.0
    for k in range(0, n_clusters):
        list = []
        for i in range(0, len(labels)):
            if labels[i] != k:
                continue
            list.append(utils.euclidian_dist(X[i], centroids[k]))

        # get sum of 0.1*|Ck| largest elements
        acc = 0.0
        max_n = heapq.nlargest(int(math.ceil(0.1 * cluster_sizes[k])), list)
        for i in range(0, len(max_n)):
            acc += max_n[i]
        denominator += acc * 10.0 / cluster_sizes[k]
    return - numerator / denominator


# OS-Index:
def a(X, labels, x_i, cluster_k_index):
    acc = 0.0
    count = 0
    for j in range(0, len(labels)):
        if labels[j] != cluster_k_index: continue
        acc += utils.euclidian_dist(x_i, X[j])
        count += 1
    return acc / count


def b(X, labels, x_i, cluster_k_index):
    dists = []
    c_k_size = 0
    for j in range(0, len(labels)):
        if labels[j] == cluster_k_index:
            c_k_size += 1
            continue
        dists.append(utils.euclidian_dist(x_i, X[j]))

    # TODO: it can happen, that c_k_size if bigger than len(dists). Is it supposed to be so?

    acc = 0.0
    min_n = heapq.nsmallest(c_k_size, dists)
    for i in range(0, len(min_n)):
        acc += min_n[i]
    return acc / c_k_size


def ov(X, labels, x_i, cluster_k_index):
    a_s = a(X, labels, x_i, cluster_k_index)
    b_s = b(X, labels, x_i, cluster_k_index)

    if (b_s - a_s) / (b_s + a_s) < 0.4:
        return a_s / b_s
    else:
        return 0


# os, OS-Index, max is better, added -
def os(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)

    numerator = 0.0
    for k in range(0, n_clusters):
        for i in range(0, len(labels)):
            if labels[i] != k: continue
            numerator += ov(X, labels, X[i], k)

    denominator = 0.0
    for k in range(0, n_clusters):
        l = []
        for i in range(0, len(labels)):
            if labels[i] != k:
                continue
            l.append(utils.euclidian_dist(X[i], centroids[k]))

        # get sum of 0.1*|Ck| largest elements
        acc = 0.0
        max_n = heapq.nlargest(int(math.ceil(0.1 * cluster_sizes[k])), l)
        for i in range(0, len(max_n)):
            acc += max_n[i]

        denominator += acc * 10.0 / cluster_sizes[k]

    return - numerator / denominator


# SymDB:
def sym_s(X, labels, cluster_k_index, cluster_sizes, centroids):
    acc = 0.0
    for i in range(0, len(labels)):
        if labels[i] != cluster_k_index: continue
        acc += d_ps(X, labels, X[i], cluster_k_index, centroids)
    return acc / float(cluster_sizes[cluster_k_index])


# SymDB, Sym Davies-Bouldin index, min is better
def sym_db(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)
    db = 0
    cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
    max_fraction = sys.float_info.min
    for k in range(0, n_clusters):
        for l in range(0, n_clusters):
            if k != l:
                fraction = ((sym_s(X, labels, k, cluster_sizes, centroids) +
                            sym_s(X, labels, l, cluster_sizes, centroids))
                           / utils.euclidian_dist(centroids[k], centroids[l]))
                max_fraction = max(max_fraction, fraction)
        db += max_fraction
    db /= float(n_clusters)
    return db


# S_Dbw: (under construction)
# def euclidean_norm(x):
#     return np.linalg.norm(x)

def f(x_i, centroid_k, std):
    if std < utils.euclidian_dist(x_i, centroid_k):
        return 0
    else:
        return 1


def mean(x_i, x_j):
    return (x_i + x_j) / 2


def den2(X, labels, centroids, k, l, std):
    acc = 0.0
    elements = len(X)
    for i in range(0, elements):
        if labels[i] == k or labels[i] == l:
            acc += f(X[i], mean(centroids[k], centroids[l]), std)
    return acc


def den1(X, labels, centroids, k, std):
    acc = 0.0
    elements = len(X)
    for i in range(0, elements):
        if labels[i] == k:
            acc += f(X[i], centroids[k], std)
    return acc


def normed_sigma(X):
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


def normed_cluster_sigma(X, labels, k):
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


def stdev(X, labels, n_clusters):
    sum = 0.0
    for k in range(0, n_clusters):
        sum += math.sqrt(normed_cluster_sigma(X, labels, k))
    sum /= n_clusters
    return sum


# s_dbw, S_Dbw index, min is better
def s_dbw(X, labels, n_clusters):
    centroids = cluster_centroid.cluster_centroid(X, labels, n_clusters)

    sigmas = 0.0
    for k in range(0, n_clusters):
        sigmas += normed_cluster_sigma(X, labels, k)
    sigmas /= n_clusters
    sigmas /= normed_sigma(X)
    print(sigmas)

    stdev_val = stdev(X, labels, n_clusters)
    print(stdev_val)
    dens = 0.0
    for k in range(0, n_clusters):
        for l in range(0, n_clusters):
            dens += den2(X, labels, centroids, k, l, stdev_val) /\
                        max(den1(X, labels, centroids, k, stdev_val),
                            den1(X, labels, centroids, l, stdev_val))

    dens /= n_clusters * (n_clusters - 1)
    return sigmas + dens
