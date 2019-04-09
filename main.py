import numpy as np
from sklearn import cluster
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import metrics.ch_index as ch_index
import metrics.dunn_index as dunn_index
import metrics.davies_bouldin_star as db_star
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
import sys
from batch_tasks import tasks
import traceback

from Clusterization import clusterization
from Algorithms import GreedyAlgorithm, EvoOnePlusOne, EvoOnePlusFour

output_prefix = '.'

def run_config(fname, data, index, algo):
    print('Launching', fname, file=sys.stderr)
    for k in range(0, 20):
        with open(fname, 'a') as result:
            try:
                X = []
                with open(data, 'r') as f:
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
                res = cluster.SpectralClustering(n_clusters=n_clusters, random_state=k,
                                                 eigen_solver='arpack',
                                                 affinity="nearest_neighbors").fit(X1)
                # res = DBSCAN().fit(X1)
                labels = res.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                cl = clusterization(X1, labels, n_clusters, index)
                m = cl.init_measure()


                strategy = algo(cl, m)

                new_measure, iters, t = strategy.run()

                #result.write("Run of the algorithm  {}".format(strategy.__name__))
                result.write("Measure improvement   {}\n".format(abs(m - new_measure)))
                result.write("from                  {}\n".format(m))
                result.write("to                    {}\n".format(new_measure))
                result.write("Iterations performed  {}\n".format(iters))
                result.write("Time spent            {}\n".format(t))
                continue
            except:
                traceback.print_exc(file=result)


if __name__ == "__main__":
    # if len(sys.argv) == 1:
    #     print('Total', len(tasks), 'tasks to run')
    # elif len(sys.argv) == 2:
    #     print(tasks[int(sys.argv[1])][0])
    # else:
    #     output_prefix = sys.argv[2]
    #     eval(tasks[int(sys.argv[1])][1])

    for i in range(len(tasks)):
       print(tasks[i][0])
       output_prefix = '.'
       eval(tasks[i][1])


# indicies = [gD33.Index(), gD43.Index(), gD31.Index(),
#             c_index.Index(), ch_index.Index(),
#             dunn_index.Index(), db_star.Index(), sil_index.Index(),
#             db.Index(), gD41.Index(), gD51.Index(), gD53.Index(),
#             cs.Index(), dbs.Index(), sym.Index(), cop.Index(),
#             sv.Index(), sdb.Index(), sdbw.Index(), os.Index()]
#
# i = 0
# for index in indicies:
#     i += 1
#     print("Index " + str(i))
#     for Algo in GreedyAlgorithm, EvoOnePlusOne: #, EvoOnePlusFour:
#         cl = clusterization(X1, labels, n_clusters, index)
#         m = cl.init_measure()
#         algo = Algo(cl, m)
#         new_measure, iters, t = algo.run()
#         print("Run of the algorithm  {}".format(Algo.__name__))
#         print("Measure improvement   {}".format(m - new_measure))
#         print("from                  {}".format(m))
#         print("to                    {}".format(new_measure))
#         print("Iterations performed  {}".format(iters))
#         print("Time spent            {}".format(t))