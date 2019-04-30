from os import listdir
from numpy import mean


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
    s ='''\\begin{{tikzpicture}}
\\begin{{axis}}[
    title={},
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
\\node[below right, xshift=-10pt] at (border.north east) {\\ref{legend}};   
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
        legend style={{cells={{align=left}}}},
        name=border
    ]
    '''.format('Measure: {}\\, Dataset: {}'.format(measure.replace('_', '\\_'), dataset.replace('_', '\\_')),
               ', '.join([str(4 * i + 2) for i in range(len(list(data_improvements.keys())))]),
               ', '.join(algo_names),
               ', '.join(['iterative recalculation', 'full recalcualtion', 'full recalculation\\\\without time limit']))
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

measures = list(data_improvements[list(data_improvements.keys())[0]].keys())
for measure in measures:
    with open('plots/measure_{}.tex'.format(measure), 'w') as f:
        f.write(print_boxplot_imrovement(measure))

for dataset in data_iterations:
    for measure in data_iterations[dataset]:
        with open('plots/iters-{}-{}.tex'.format(dataset, measure), 'w') as f:
            f.write(print_boxplot_iterations(dataset, measure))

for dataset in data_time:
    for measure in data_time[dataset]:
        with open('plots/times-{}-{}.tex'.format(dataset, measure), 'w') as f:
            f.write(print_boxplot_time(dataset, measure))
