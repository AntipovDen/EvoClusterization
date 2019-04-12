import numpy as np
from multiprocessing import Value, Pool
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
import datetime

from Clusterization import clusterization
from Algorithms import GreedyAlgorithm, EvoOnePlusOne, EvoOnePlusFour

output_prefix = '.'

def run_config(fname, data, index, algo, init):
    print('Launching', fname, file=sys.stderr)
    today = datetime.datetime.now()
    print(today.strftime("%Y-%m-%d %H.%M.%S") ) # 2017-04-05-00.18.00

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
            #connectivity = kneighbors_graph(X1, n_neighbors=2, include_self=False)

            res = init.fit(X1)
            labels = res.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            cl = clusterization(X1, labels, n_clusters, index)
            m = cl.init_measure()

            strategy = algo(cl, m)

            new_measure, iters, t = strategy.run()

            result.write("Measure improvement   {}\n".format(abs(m - new_measure)))
            result.write("from                  {}\n".format(m))
            result.write("to                    {}\n".format(new_measure))
            result.write("Iterations performed  {}\n".format(iters))
            result.write("Time spent            {}\n".format(t))
        except:
            traceback.print_exc(file=result)


# global counters
counter_one_threaded, counter_multi_threaded = None, None

# initializing the counter for the pool of processes.
def init(counter_1, counter_2):
    global counter_one_threaded, counter_multi_threaded
    counter_one_threaded = counter_1
    counter_multi_threaded = counter_2

# the function of the thread that runs tasks
def run_tasks(thread_number):
    global counter_one_threaded, counter_multi_threaded
    # if thread number is 0 or 1, we run the next possible one-threaded task
    if thread_number in [0, 1]:
        while True:
            with counter_one_threaded.get_lock():
                task_number = counter_one_threaded.value
                counter_one_threaded.value += 1
            task_number = task_number // 2 * 3 + task_number % 2
            if task_number > 3419: #170:
                return
            # print('({}, {}),'.format(thread_number, task_number))
            eval(tasks[task_number][1])
    else:
        while True:
            with counter_multi_threaded.get_lock():
                task_number = counter_multi_threaded.value
                counter_multi_threaded.value += 1
            task_number = task_number * 3 + 2
            if task_number > 3419:#170:
                return

            # print('({}, {}),'.format(thread_number, task_number))
            eval(tasks[task_number][1])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Total', len(tasks), 'tasks to run')
    elif len(sys.argv) == 2:
        print(tasks[int(sys.argv[1])][0])
    else:
        output_prefix = sys.argv[2]
        # 2 processes run 1-thread algorithms and 1 process runs a (1 + 4) EA
        with Pool(3, init, (Value('i', 0), Value('i', 0))) as pool:
            pool.map(run_tasks, range(3))
        # eval(tasks[int(sys.argv[1])][1])


    # for i in range(len(tasks)):
    #    print(tasks[i][0])
    #    output_prefix = '.'
    #    eval(tasks[i][1])


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