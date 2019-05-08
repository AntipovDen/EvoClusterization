from math import sqrt
from os import listdir
from statistics import median, mean, stdev

def read_run(file):
    lines = [file.readline() for _ in range(5)]
    measure_improvement = float(lines[0].split()[-1])
    iterations = int(lines[3].split()[-1])
    time_spent = float(lines[4].split()[-1])
    return measure_improvement, iterations, time_spent


data_dir = 'output'
approaches = ['iterative', 'full', 'full_long']
algo_names = ['greedy', '$(1 + 1)$', '$(1 + 4)$']
algo_ids = ['greedy', 'evo_one_one', 'evo_one_four']
measure_shortname = {'calinski_harabaz' : 'ch', 'cop': 'cop', 'davies_bouldin_star': 'db*', 'silhouette': 'sil'}
runs_per_config = 10
data_improvements = dict()
data_iterations = dict()
data_time = dict()
for filename in listdir(data_dir):
    dataset, measure, algo = filename[:-4].split('-')
    if dataset not in data_improvements:
        data_improvements[dataset] = dict()
        data_iterations[dataset] = dict()
        data_time[dataset] = dict()
    if measure not in data_improvements[dataset]:
        data_improvements[dataset][measure] = dict()
        data_iterations[dataset][measure] = dict()
        data_time[dataset][measure] = dict()
    with open(data_dir + '/' + filename, 'r') as f:
        # print('Reading {}'.format(filename))
        res = []
        for i in range(runs_per_config):
            f.readline()
            res += [read_run(f) for _ in range(3)]
            f.readline()
            # f.readline()
            # res += [read_run(f) for _ in range(3)]
            # f.readline()
            # f.readline()
            # res += [read_run(f) for _ in range(3)]

    for i in range(3):
        res_approach = [res[j * 3 + i] for j in range(runs_per_config)]
        algo_specified = algo + '-' + approaches[i]
        data_improvements[dataset][measure][algo_specified] = [res_approach[k][0] for k in range(runs_per_config)]
        data_iterations[dataset][measure][algo_specified] = [res_approach[k][1] for k in range(runs_per_config)]
        data_time[dataset][measure][algo_specified] = [res_approach[k][2] for k in range(runs_per_config)]


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
    %\\legend{iterative recalculation,full recalculation,full with no time limit}
\end{axis}
\end{tikzpicture}'''
    return s


def print_boxplot_imrovement(measure):
    s ='''\\begin{{tikzpicture}}[trim axis left]
\\begin{{axis}}[
    title={},
    scale only axis,
    height=5cm,
    width=0.8\\textwidth,
    boxplot/draw direction=y,
    ylabel=Measure improvement,
    axis y line=left,
    enlarge y limits,
    ymajorgrids,
    xtick={{{}}},
    xticklabels={{{}}},
    x tick label style={{rotate=45,anchor=east}},
    /pgfplots/boxplot/whisker range={{100000000}},
    /pgfplots/boxplot/every box/.style={{solid}},
    /pgfplots/boxplot/every whisker/.style={{solid}},
    /pgfplots/boxplot/every median/.style={{solid,thick}},
    legend entries = {{{}}},
    legend to name={{legend}},
    name=border
]
'''.format(measure.replace('_', '\\_'),
           ', '.join([str(7 * i + 3.5) for i in range(len(list(data_improvements.keys())))]),
           ', '.join([dataset for dataset in data_improvements]),
           ', '.join([name + ', ' + name + '*' for name in algo_names]))
    colors = ['red', 'blue', 'black', 'green', 'orange', 'purple']
    i = 0
    for dataset in data_improvements:
        j = 0
        for algo in reversed(sorted(list(data_improvements[dataset][measure]))):
            if 'iterative' in algo or 'full_long' in algo:
                color = colors[j]
                print(algo, color)
                s += '''    \\addplot+ [{}, boxplot={{draw position={}}}, mark options={{solid,mark=square,fill=white,draw={}}}]
        table [row sep=\\\\,y index=0] {{
            data\\\\
            {}\\\\
    }};
'''.format(color, i * 7 + j + 1, color, '\\\\ '.join([str(improvement) for improvement in data_improvements[dataset][measure][algo]]))
                j += 1
        i += 1
    s += '''\end{axis}
\\node[below right] at (border.north east) {\\ref{legend}};   
\end{tikzpicture}'''
    return s


def print_boxplot_iterations(dataset, measure):
    s = '''\\begin{{tikzpicture}}
    \\begin{{axis}}[
        title={},
        boxplot/draw direction=y,
        ylabel=Iterations performed,
        axis y line=left,
        enlarge y limits,
        ymajorgrids,
        xtick={{{}}},
        xticklabels={{{}}},
        x tick label style={{rotate=45,anchor=east}},
        /pgfplots/boxplot/whisker range={{100000000}},
        /pgfplots/boxplot/every box/.style={{solid}},
        /pgfplots/boxplot/every whisker/.style={{solid}},
        /pgfplots/boxplot/every median/.style={{solid,thick}},
        legend entries = {{{}}},
        legend to name={{legend}},
        name=border
    ]
    '''.format('Measure: {}\\, Dataset: {}'.format(measure.replace('_', '\\_'), dataset.replace('_', '\\_')),
               ', '.join([str(4 * i + 2) for i in range(len(list(data_iterations.keys())))]),
               ', '.join(algo_names),
               ', '.join(['iterative recalculation', 'full recalcualtion', 'full recalculation without time limit']))
    colors = ['red', 'blue', 'black']
    for i in range(3): #number of algo
        for j in range(3): #number of approach
            color = colors[j]
            s += '''    \\addplot+ [{}, boxplot={{draw position={}}}, mark options={{solid,mark=square,fill=white,draw={}}}]
            table [row sep=\\\\,y index=0] {{
                data\\\\
                {}\\\\
        }};
    '''.format(color, i * 4 + j + 1, color,
               '\\\\ '.join([str(iterations) for iterations in data_iterations[dataset][measure][algo_ids[i] + '-' + approaches[j]]]))
    s += '\end{axis}\n'
    if 'star' in measure:
        s += '\\node[below=50pt, right] at (border.south west) {\\ref{legend}};\n'
    elif 'silh' in measure:
        s += '\\node[below=71pt, right] at (border.south west) {};\n'
    s += '\end{tikzpicture}\n'
    return s


def print_boxplot_time(dataset, measure):
    s = '''\\begin{{tikzpicture}}
        \\begin{{axis}}[
            title={},
            boxplot/draw direction=y,
            ylabel=Time spent,
            axis y line=left,
            enlarge y limits,
            ymajorgrids,
            xtick={{{}}},
            xticklabels={{{}}},
            x tick label style={{rotate=45,anchor=east}},
            /pgfplots/boxplot/whisker range={{3}},
            /pgfplots/boxplot/every box/.style={{solid}},
            /pgfplots/boxplot/every whisker/.style={{solid}},
            /pgfplots/boxplot/every median/.style={{solid,thick}},
            legend entries = {{{}}},
            legend to name={{legend}},
            name=border
        ]
        '''.format('Measure: {}\\, Dataset: {}'.format(measure.replace('_', '\\_'), dataset.replace('_', '\\_')),
                   ', '.join([str(4 * i + 2) for i in range(len(list(data_time.keys())))]),
                   ', '.join(algo_names),
                   ', '.join(
                       ['iterative recalculation', 'full recalcualtion', 'full recalculation without time limit']))
    colors = ['red', 'blue', 'black']
    for i in range(3):  # number of algo
        for j in range(3):  # number of approach
            color = colors[j]
            s += '''    \\addplot+ [{}, boxplot={{draw position={}}}, mark options={{solid,mark=square,fill=white,draw={}}}]
                table [row sep=\\\\,y index=0] {{
                    data\\\\
                    {}\\\\
            }};
        '''.format(color, i * 4 + j + 1, color,
                   '\\\\ '.join([str(iterations) for iterations in
                                 data_time[dataset][measure][algo_ids[i] + '-' + approaches[j]]]))
    s += '\end{axis}\n'
    if 'star' in measure:
        s += '\\node[below=50pt, right] at (border.south west) {\\ref{legend}};\n'
    elif 'silh' in measure:
        s += '\\node[below=71pt, right] at (border.south west) {};\n'
    s += '\end{tikzpicture}\n'
    return s


def is_successful(dataset, measure):
    for algo in data_improvements[dataset][measure]:
        if 'iterative' in algo or 'full_long' in algo:
            for i in data_improvements[dataset][measure][algo]:
                if i != 0:
                    return True
    return False


# not needed
def median_data(dataset, measure, algo): # median value of improvement by full recalculation
    res = [data_improvements[dataset][measure][algo + '-full_long'][i] for i in range(10) if data_iterations[dataset][measure][algo + '-full_long'][i] != 0]
    if len(res) == 0:
        return '---'
    return '{:.3g}'.format(median(res))


def mean_data(dataset, measure, algo): # median value of improvement by full recalculation
    valid_runs = len([i for i in data_iterations[dataset][measure][algo + '-full_long'] if i > 0])
    res = [data_time[dataset][measure][algo + '-full_long'][i] for i in range(10) if data_iterations[dataset][measure][algo + '-full_long'][i] != 0]
    if valid_runs == 0:
        return '---'
    max_time = 25 * 60 if algo != 'evo_one_four' else 25 * 60 * 4
    successful_runs = len([i for i in res if i <= max_time])
    if successful_runs == 0:
        return '$>{}$'.format(max_time // 60)
    r = successful_runs / valid_runs
    m = mean([i for i in res if i <= max_time]) + (1 - r) / r * max_time
    return '${:.3g}$'.format(m / 60)


def deviation_data(dataset, measure, algo): # median value of improvement by full recalculation
    valid_runs = len([i for i in data_iterations[dataset][measure][algo + '-full_long'] if i > 0])
    res = [data_time[dataset][measure][algo + '-full_long'][i] for i in range(10) if
           data_iterations[dataset][measure][algo + '-full_long'][i] != 0]
    if valid_runs == 0:
        return '---'
    max_time = 25 * 60 if algo != 'evo_one_four' else 25 * 60 * 4
    successful_runs = len([i for i in res if i <= max_time])
    if successful_runs == 0:
        return 'unknown'
    r = successful_runs / valid_runs
    e_s = mean([i for i in res if i <= max_time])
    e = e_s + (1 - r) / r * max_time
    d_s = 0 if successful_runs == 1 else stdev([i for i in res if i <= max_time])
    d = sqrt(e_s ** 2 - e ** 2 + d_s ** 2 + (1 - r ) / r * (max_time ** 2 + 2 * max_time * e))
    return '${:.3g}$'.format(d / 60)


def percent_success_data(dataset, measure, algo): # median value of improvement by full recalculation
    valid_runs = len([i for i in data_iterations[dataset][measure][algo + '-full_long'] if i > 0])
    res = [data_time[dataset][measure][algo + '-full_long'][i] for i in range(10) if
           data_iterations[dataset][measure][algo + '-full_long'][i] != 0]
    if valid_runs == 0:
        return '---'
    max_time = 25 * 60 if algo != 'evo_one_four' else 25 * 60 * 4
    successful_runs = len([i for i in res if i <= max_time])
    return '{}/{}'.format(successful_runs, valid_runs)


def print_table():
    # unsuccessful_settings = ((dataset, measure) for dataset in data_improvements for measure in data_improvements[dataset] if not is_successful(dataset, measure))
    # for i in unsuccessful_settings:
    #     print(i)
    s = '''\\begin{tabular}{|c|c|rr|rr|rr|}
  \\hline
  \\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Measure} & \\multicolumn{2}{c|}{Mean, min} & \\multicolumn{2}{c|}{Deviation, min} & \\multicolumn{2}{c|}{Successful runs} \\\\ \\cline{3-8}
  & & $(1 + 1)$ & $(1 + 4)$ & $(1 + 1)$ & $(1 + 4)$ & $(1 + 1)$ & $(1 + 4)$  \\\\ \\hline
'''
    for dataset in data_improvements:
        # s += '  \\parbox[t]{{2mm}}{{\multirow{{3}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{}}}}}}}'.format(dataset.replace('_', '\\_'))
        s += '  \multirow{{3}}{{*}}{{{}}}'.format(dataset.replace('_', '\\_'))
        for measure in sorted(list(data_improvements[dataset])):
            s += ' & {} & '.format(measure.replace('_', '\\_'))
            # s += ' & '.join([median_data(dataset, measure, algo) for algo in algo_ids if 'evo' in algo])
            # s += ' & '
            s += ' & '.join([mean_data(dataset, measure, algo) for algo in algo_ids if 'evo' in algo])
            s += ' & '
            s += ' & '.join([deviation_data(dataset, measure, algo) for algo in algo_ids if 'evo' in algo])
            s += ' & '
            s += ' & '.join([percent_success_data(dataset, measure, algo) for algo in algo_ids if 'evo' in algo])
            s += ' \\\\\n'
        s += '\\hline\n'
    return s + '\end{tabular}\n'

print(data_time['glass']['silhouette']['evo_one_four-full_long'])
print(data_iterations['glass']['silhouette']['evo_one_four-full_long'])
# def check():
#     for dataset in data_improvements:
#         for measure in data_improvements[dataset]:
#             for algo in algo_ids:
#                 iterative_has_improved = sum(data_improvements[dataset][measure][algo + '-iterative']) > 0
#                 full_recalcuation_did_nothing = len([i for i in data_iterations[dataset][measure][algo + '-full_long'] if i == 0]) > 0
#                 if iterative_has_improved and full_recalcuation_did_nothing:
#                     print(dataset, measure, algo)
#
# for dataset in data_time:
#     for measure in data_time[dataset]:
#         for algo in algo_ids:
#             if len([i for i in data_time[dataset][measure][algo + '-full_long'] if i > 25 * 60]) == 10 and algo != 'evo_one_four' or \
#                len([i for i in data_time[dataset][measure][algo + '-full_long'] if i > 25 * 60 * 4]) == 10:
#                 print(dataset, measure, algo, data_time[dataset][measure][algo + '-full_long'])
#
# # check()
# exit(0)

with open('tables/improvement-table.tex', 'w') as f:
    f.write(print_table())
# measures = list(data_improvements[list(data_improvements.keys())[0]].keys())
# for measure in measures:
#     with open('plots/measure_{}.tex'.format(measure), 'w') as f:
#         f.write(print_boxplot_imrovement(measure))
#
# for dataset in data_iterations:
#     for measure in data_iterations[dataset]:
#         with open('plots/iters-{}-{}.tex'.format(dataset, measure), 'w') as f:
#             f.write(print_boxplot_iterations(dataset, measure))
#
# for dataset in data_time:
#     for measure in data_time[dataset]:
#         with open('plots/times-{}-{}.tex'.format(dataset, measure), 'w') as f:
#             f.write(print_boxplot_time(dataset, measure))
