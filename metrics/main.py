import random

from matplotlib import pyplot as plt

import numpy as np
import time

import metrics.c_index as c_index
import metrics.ch_index
import metrics.cop_index as cop_index
import metrics.cs_index as cs_index
import metrics.davies_bouldin_star as davies_bouldin_star
import metrics.dunn_index as dunn_index
import metrics.gD31_index as gD31_index
import metrics.gD33_index as gD33_index
import metrics.gD41_index as gD41_index
import metrics.gD43_index as gD43_index
import metrics.gD51_index as gD51_index
import metrics.gD53_index as gD53_index
import metrics.cluster_centroid as cluster_centroid
import metrics.davies_bouldin as davies_bouldin
import metrics.os_index as os_index
import metrics.s_dbw_index as s_dbw_index
import metrics.sil_index as sil_index
import metrics.sv_index as sv_index
import metrics.sym_db_index as sym_db_index
import metrics.sym_index as sym_index
import metrics.utils as utils

clusters = [[] for _ in range(3)]
for i in range(30):
    for j in range(30):
        clusters[0].append([i, j])
clusters[0].append([35, 15])

for i in range(40, 70):
    for j in range(30):
        clusters[1].append([i, j])

for i in range(70):
    for j in range(-40, -25):
        clusters[2].append([i, j])

#print(len(clusters[0]), len(clusters[1]), len(clusters[2]))

X = []
labels = []
for i in range(3):
    for j in range(len(clusters[i])):
        X.append(clusters[i][j])
        labels.append(i)
#print(cluster_centroid.cluster_centroid(np.array(X), labels, 3))


#print(len(clusters[0]), len(clusters[1]), len(clusters[2]))

newX = list(X)
newlabels = list(labels)
newlabels[900] = 1
#print(cluster_centroid.count_cluster_sizes(labels, 3))


start_time = time.time()
res = s_dbw_index.s_dbw(np.array(X), labels, 3)
print(time.time() - start_time)
print(res)


start_time = time.time()
res = s_dbw_index.update_s_dbw(X, labels, 3, [35, 15], 0, 1)
print(time.time() - start_time)
print(res)


start_time = time.time()
res = s_dbw_index.s_dbw(np.array(newX), newlabels, 3)
print(time.time() - start_time)
print(res)




