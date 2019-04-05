import numpy as np
from sklearn import cluster
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import metrics.ch_index as ch_index
import metrics.dunn_index as dunn_index
import metrics.davies_bouldin as davies_bouldin
import metrics.sil_index as sil_index
import metrics.c_index as c_index
import metrics.davies_bouldin as db
import metrics.gD41_index as gD41
import metrics.gD51_index as gD51
import metrics.gD53_index as gD53
import metrics.cs_index as cs
import metrics.davies_bouldin_star as dbs
import metrics.sym_index as sym
import metrics.cop_index as cop
import metrics.sv_index as sv
import metrics.sym_db_index as sdb
import metrics.s_dbw_index as sdbw
import metrics.os_index as os
import metrics.gD33_index as gD33
import metrics.gD43_index as gD43
import metrics.gD31_index as gD31

from Clusterization import clusterization
from Algorithms import GreedyAlgorithm, EvoOnePlusOne, EvoOnePlusFour

X = []
with open('s3.txt', 'r') as f:
    content = f.readlines()
    for x in content:
        row = x.split()
        res = []
        for i in row:
            res.append(int(i))
        X.append(res)

n_clusters = 15
X1 = StandardScaler().fit_transform(np.array(X))
connectivity = kneighbors_graph(X1, n_neighbors=2, include_self=False)
# res = KMeans(n_clusters=n_clusters, random_state=0).fit(X1)
# res = cluster.Birch(n_clusters=n_clusters).fit(X1)
# res = cluster.AgglomerativeClustering(
#             linkage="average", affinity="cityblock",
#             n_clusters=n_clusters, connectivity=connectivity).fit(X1)
res = cluster.SpectralClustering(n_clusters=n_clusters,
                                 eigen_solver='arpack',
                                 affinity="nearest_neighbors").fit(X1)
# res = DBSCAN().fit(X1)
labels = res.labels_
print(labels)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#

indicies = [gD33.Index(), gD43.Index(), gD31.Index(),
            c_index.Index(), ch_index.Index(), dunn_index.Index(),
            davies_bouldin.Index(), sil_index.Index(),
            db.Index(), gD41.Index(), gD51.Index(), gD53.Index(),
            cs.Index(), dbs.Index(), sym.Index(), cop.Index(),
            sv.Index(), sdb.Index(), sdbw.Index(), os.Index()]

i = 0
for index in indicies:
    i += 1
    print("Index " + str(i))
    cl = clusterization(X1, labels, n_clusters, index)
    m = cl.init_measure()
    # algo = GreedyAlgorithm(cl)
    # algo = EvoOnePlusOne(cl)
    algo = EvoOnePlusFour(cl)
    print(algo.run())