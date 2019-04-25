datas = [
    ('iris', 'iris.txt'),
    ('seeds', 'seeds.txt'),
    ('wholesale', 'wholesale_customers.txt'),
    ('glass', 'glass.txt'),
    ('ionosphere', 'ionosphere.txt')
    #('s3', 's3.txt'),
    #('g2', 'g2-2-50.txt'),
    #('bridge', 'bridge.txt')
]

indices = [
    ('silhouette', 'sil_index.Index()'),
    ('calinski_harabaz', 'ch_index.Index()'),
    ('davies_bouldin_star', 'db_star.Index()'),
    ('cop', 'cop.Index()')

    # ('davies_bouldin', 'db.Index()'),
    # ('generalized_dunn_41', 'gD41.Index()'),
    # ('generalized_dunn_43', 'gD43.Index()'),
    # ('generalized_dunn_51', 'gD51.Index()'),
    # ('generalized_dunn_53', 'gD53.Index()'),
    # ('generalized_dunn_31', 'gD31.Index()'),
    # ('generalized_dunn_33', 'gD33.Index()'),
    # ('c_index', 'c_index.Index()'),
    # ('os_index', 'os.Index()'),
    # ('s_dbw', 'sdbw.Index()'),
    # ('sym', 'sym.Index()'),
    # ('sym_db', 'sdb.Index()'),
    # ('sv_index', 'sv.Index()'),
    # ('cs', 'cs.Index()'),
]

#GreedyAlgorithm, EvoOnePlusOne: #, EvoOnePlusFour:
algos = [
    ('evo_one_four', 'EvoOnePlusFour'),
    ('greedy', 'GreedyAlgorithm'),
    ('evo_one_one', 'EvoOnePlusOne')
]


#initializations = [
#    ('birch', 'cluster.Birch()'),
#    ('agglomerative', 'cluster.AgglomerativeClustering(linkage="average", affinity="cityblock")')
#]

#for k in range(0, 9):
#    initializations.append(('spectral-' + str(k), 'cluster.SpectralClustering(random_state='+
#                            str(k) +', eigen_solver="arpack", affinity="nearest_neighbors")'))
#    initializations.append(('k-means-' + str(k), 'cluster.KMeans(random_state='+
#                            str(k)+')'))

# def get_file_name(data, index, algo, init):
#     return '{}-{}-{}-{}.txt'.format(data, index, algo, init)


def get_file_name(data, index, algo):
    return '{}-{}-{}.txt'.format(data, index, algo)


# tasks = [('state-of-the-art.txt', 'run_state_of_the_art([datas, indices])')]

#number_of_runs = 3

tasks = []
# for data_name, data in datas:
#     for index_name, index in indices:
#         for algo_name, algo in algos:
#             for init_name, init in initializations:
#                 fname = get_file_name(data_name, index_name, algo_name, init_name)
#                 tasks.append((fname, "run_config(output_prefix+ '/' + '{}', '{}', {}, {}, {})".format(fname,
#                                                                                                 data, index, algo, init)))
#for run_num in range(0, number_of_runs):
for data_name, data in datas:
    for index_name, index in indices:
        for algo_name, algo in algos:
            fname = get_file_name(data_name, index_name, algo_name)
            tasks.append((fname, "run_config(output_prefix+ '/' + '{}', '{}', {}, {})".format(fname, data, index, algo)))