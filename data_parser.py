from os import listdir
from numpy import mean


def read_run(file):
    lines = [file.readline() for _ in range(5)]
    measure_improvement = float(lines[0].split()[-1])
    iterations = int(lines[3].split()[-1])
    return measure_improvement, iterations


data_dir = 'output'
runs_per_config = 3
data = dict()
for filename in listdir(data_dir):
    dataset, measure, algo = filename[:-4].split('-')
    if dataset not in data:
        data[dataset] = dict()
    if measure not in data[dataset]:
        data[dataset][measure] = dict()
    with open(data_dir + '/' + filename, 'r') as f:
        # print('Reading {}'.format(filename))
        f.readline()
        res = [read_run(f) for _ in range(3)]
        f.readline()
        f.readline()
        res += [read_run(f) for _ in range(3)]
        f.readline()
        f.readline()
        res += [read_run(f) for _ in range(3)]

    approaches = ['iterative', 'full', 'full_long']

    for i in range(3):
        res_approach = [res[j * 3 + i] for j in range(runs_per_config)]
        algo_specified = algo + '-' + approaches[i]
        data[dataset][measure][algo_specified] = [mean([res_approach[k][j] for k in range(runs_per_config)]) for j in range(2)]


def print_histogram(means, ticks=[], plotname="", iters = None):
    s = '''\\begin{{tikzpicture}}
\\begin{{axis}} [
    title = {},
    enlargelimits=0.2,
    ylabel=measure improvement,
    ybar,
    symbolic x coords={{{}}},
    xtick=data,
    x tick label style={{rotate=45,anchor=east}},
    legend style={{at={{(0.5,-0.3)}},
    anchor=north,legend columns=-1}},
]
    \\addplot coordinates {{
            '''.format(plotname, ', '.join(ticks))
    algos = ['greedy', 'evo_one_one', 'evo_one_four']
    for i in range(3):
        s += '({}, {}) '.format(ticks[i], means[algos[i] + '-iterative'][0])
    s += '''
    };
    
    \\addplot coordinates {
            '''
    for i in range(3):
        s += '({}, {}) '.format(ticks[i], means[algos[i] + '-full'][0])
    s += '''
    };

    \\addplot coordinates {
            '''
    for i in range(3):
        s += '({}, {}) '.format(ticks[i], means[algos[i] + '-full_long'][0])
    s += '''
    };
    \\legend{iterative recalculation,full recalculation,full with no time limit}
\end{axis}
\end{tikzpicture}'''
    return s


for dataset in data:
    # print(data[dataset])
    for measure in data[dataset]:
        algo_ids = ['greedy', 'evo_one_one', 'evo_one_four']
        algo_names = ['greedy', '$(1 + 1)$', '$(1 + 4)$']
        # mean_values = [data[dataset][measure][algo + '-iterative'][0] for algo in algo_ids]
        print(print_histogram(data[dataset][measure], algo_names, '\\, '.join([dataset, measure.replace('_', '\\_')])))
        print('\\hskip 10pt')
    print('\\\\')