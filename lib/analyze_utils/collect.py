"""Result collection helpers.

Walks the trained-model save directories, loads pickled hidden-state
records, and dispatches them to the visualization runners in
lib.analyze_utils.orchestration.
"""
import os
import json
import glob
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.io
from scipy import stats
from .orchestration import run_visualization, run_shifted_visualization, run_stretched_visualization

def boxplot(data, x, y, palette=None, order=None):
    """Project-style box+swarm plot drawn onto the current matplotlib axes.

    Args:
        data: A long-format pandas DataFrame.
        x: Column name to use as the categorical x-axis.
        y: Column name to use as the numeric y-axis.
        palette: Optional seaborn palette name or list.
        order: Optional explicit category order for ``x``.

    Returns:
        None. Draws a grey box (fliers suppressed) with a translucent black
        swarm overlay onto the current axes.
    """
    sns.boxplot(data=data, x=x, y=y, color='lightgrey', width=0.5, fliersize=0, zorder=5, palette=palette, order=order) #, zorder=1)
    sns.swarmplot(data=data, x=x, y=y, color='black', size=5, alpha=0.25, zorder=20)

def violinplot(data, x, y, palette=None, order=None):
    """Project-style violin+swarm plot drawn onto the current matplotlib axes.

    Args:
        data: A long-format pandas DataFrame.
        x: Column name to use as the categorical x-axis.
        y: Column name to use as the numeric y-axis.
        palette: Optional seaborn palette name or list.
        order: Optional explicit category order for ``x``.

    Returns:
        None. Draws a grey violin (quartile lines, no extension past the
        data range) with a translucent black swarm overlay onto the
        current axes.
    """
    sns.violinplot(data=data, x=x, y=y, color='lightgrey', width=0.5, inner='quart', linewidth=1, cut=0, palette=palette, order=order, zorder=1)
#    sns.violinplot(data=data, x=x, y=y, color='lightgrey', inner="point", width=0.5, linewidth=1, cut=0)
    sns.swarmplot(data=data, x=x, y=y, color='black', size=5, zorder=2, alpha=0.7)


def post_analysis4periodicity(dfs, save_dir: str, constraints: list) -> None:
    """Walk the per-run periodicity npy dumps and assemble a CSV summary.

    For each run matching ``constraints``, finds every
    ``periodicity/<lambda>/*.npy`` under ``save_dir/dynamics/<run>/`` and
    extracts the detected inter-peak interval distribution per landmark.

    Args:
        dfs: DataFrame produced by ``read_results``; the ``name`` column
            enumerates run directory names.
        save_dir: Root directory where the per-run ``dynamics/`` outputs live.
        constraints: List of substrings every kept run-directory name must contain.

    Returns:
        None. Writes a per-direction periodicity-summary CSV into ``save_dir``.
    """
    dirs = sorted(dfs.name.unique())
    dirs = filter_path(dirs, constraints)
    for direction in ['']: #['left', 'right', '']:
        data = []
        for fname in dirs:
            dynamics_path = os.path.join(save_dir, 'dynamics', fname)
            periodicity_paths = glob.glob(os.path.join(dynamics_path, '*/*', f'{direction}periodicity', '*', '*.npy'))
            for path in periodicity_paths:
                landmark_interval = int(path.split('/')[-1].split('_')[0])

                # count zeros, LM. .. 
                # 12 +- 15% range.
                name = path.split('/')[-4].replace('visual_0_', '').replace('mental__0_', '').replace('visual__0_', '').replace('mental__0_', '')
                if 'base_hidden' in name:
                    name = 'dist RNN'
                elif name == 'hidden' or name == 'visual__0_hidden':
                    name = 'action RNN'
                elif name == 'fc_hidden':
                    name = 'fc_hidden'
                elif 'gs_one_hot' in name:
                    name = 'EC'
                else:
                    pass
                
                total_intervals = np.load(path) # (each distance, RNN units)
                if name != 'action RNN' and False :
                    # sample
                    indices = np.random.choice(total_intervals.shape[1], size=25, replace=False)
                    total_intervals = total_intervals[:, indices]
                # average across distance

                avg_intervals = total_intervals.sum(0) / ((total_intervals >0).sum(0) + 1e-16)
                non_zeros = (avg_intervals != 0).mean()
                periodicity_at_LM = ((avg_intervals >= landmark_interval * 0.85) & (avg_intervals <= landmark_interval * 1.15)).mean()
#                periodicity_at_LM = ((avg_intervals >= landmark_interval * 0.95) & (avg_intervals <= landmark_interval * 1.05)).mean()
#                    continue



                seed = int(fname.split('Ep')[0].split('S')[-1])
                data.append({'name': name,
                             'non_zeros': non_zeros,
                             'periodicity_at_LM': periodicity_at_LM,
                             'ratio': periodicity_at_LM / non_zeros,
                             'count': len(avg_intervals),
                             'fname': fname,
                             'seed': seed,
                             'direction': direction,
                             })
        if len(data) == 0:
            continue
        data = pd.DataFrame(data)
        dfs = data.melt(id_vars=['name', 'seed'], value_vars=['non_zeros', 'periodicity_at_LM', 'ratio', 'count'])
        data.to_csv(f'periodicity_summary_{direction}.csv')
    # draw barplot for non_zeros and periodicity_at_LM, respectively.
    L = dfs.name.unique()
    if 'EC' in dfs.name.to_list():
        if 'imgs' in dfs.name.to_list():
            order = ['imgs', 'EC', 'fc_hidden', 'dist RNN', 'action RNN']
        elif 'fc_hidden' in dfs.name.to_list() and False:
            order = ['EC', 'fc_hidden', 'dist RNN', 'action RNN']
        else:
            order = ['EC', 'dist RNN', 'action RNN']
        for target in ['non_zeros', 'periodicity_at_LM', 'ratio']:
            fig = plt.figure(figsize=(4,3), dpi=500)
            sns.barplot(data=dfs[dfs['variable'] == target], x='name', y='value', palette=['green', 'blue', 'red'], errorbar='se', order=order)
            plt. tight_layout()
            plt.savefig(os.path.join(save_dir, f'{target}_with_EC.pdf'))

            # box plot, violin plot

        # merge grid cell and dist RNN
        _name = 'dist RNN + EC'
        if len(order) == 4:
            _name += 'fc_hidden'
        for o in order[:-1]:
            # randomly drop except 25 neurons.
#            _data = data[data.name == o]
#            drop_indices = _data.sample(n=len(_data) - 36).index
#            data = data.drop(drop_indices)
            data.name[data.name == o] = _name
        data.non_zeros = data.non_zeros * data['count']
        data.periodicity_at_LM = data.periodicity_at_LM * data['count']
        _dfs = data[['name', 'non_zeros', 'periodicity_at_LM', 'count', 'seed']].groupby(['name', 'seed']).sum().reset_index()
        _dfs['non_zeros'] = _dfs['non_zeros'] / _dfs['count']
        _dfs['periodicity_at_LM'] = _dfs['periodicity_at_LM'] / _dfs['count']
        _dfs['ratio'] = _dfs['periodicity_at_LM'] / _dfs['non_zeros']
        dfs = _dfs.melt(id_vars=['name', 'seed'], value_vars=['non_zeros', 'periodicity_at_LM', 'ratio', 'count'])

        order = [_name, 'action RNN']
    else:
        order = ['dist RNN', 'action RNN']

#    plt.savefig(os.path.join(save_dir, f'periodicity_summary_periodic.pdf'))
    
    # drop 'fc_hidden' and 'imgs' if exist
    dfs = dfs[~dfs.name.str.contains('fc_hidden')]
    dfs = dfs[~dfs.name.str.contains('imgs')]
    for target in ['non_zeros', 'periodicity_at_LM', 'ratio']:
        fig = plt.figure(figsize=(4,3), dpi=500)
        sns.barplot(data=dfs[dfs['variable'] == target], x='name', y='value', palette=['#33669a', '#d95527'], errorbar='se', order=order)
        plt. tight_layout()
        plt.savefig(os.path.join(save_dir, f'{target}.pdf'))


        fig = plt.figure(figsize=(4,3), dpi=500)
        violinplot(data=dfs[dfs['variable'] == target], x='name', y='value', palette=['#33669a', '#d95527'], order=order)
        plt. tight_layout()
        plt.savefig(os.path.join(save_dir, f'{target}_violin.pdf'))
        plt.savefig(os.path.join('./', f'{target}_violin.png'))
        plt.savefig(os.path.join('./', f'{target}_violin.pdf'))

        # ratio_box intentionally skipped; only non_zeros_box and
        # periodicity_at_LM_box are used in the paper.
        if target != 'ratio':
            fig = plt.figure(figsize=(4,3), dpi=500)
            boxplot(data=dfs[dfs['variable'] == target], x='name', y='value', palette=['#33669a', '#d95527'], order=order)
            plt. tight_layout()
            plt.savefig(os.path.join(save_dir, f'{target}_box.pdf'))
            plt.savefig(os.path.join('./', f'{target}_box.png'))
            plt.savefig(os.path.join('./', f'{target}_box.pdf'))

    #    plt.bar(data=dfs[dfs['variable'] == 'periodicity_at_LM'], x='name', y='value', palette=['blue', 'red'], errorbar='se')
#    plt.savefig(os.path.join(save_dir, f'periodicity_summary_at_LM.pdf'))
    breakpoint()


def read_results(dirname: str, constraint: list = [], seeds=None):
    """Read every per-run training log under ``dirname`` into one long DataFrame.

    Each subdirectory of ``dirname`` should contain ``log.csv`` and
    ``arguments.json`` written by train.py. Only runs whose path contains
    every substring in ``constraint`` (and, if ``seeds`` is given, whose
    seed matches one of the values) are kept.

    Args:
        dirname: Parent directory holding one subdirectory per run.
        constraint: List of substrings every kept run path must contain.
        seeds: Optional iterable of seed ints. If given, runs whose
            ``arguments.json:seed`` is not in this set are skipped.

    Returns:
        pandas.DataFrame: Long-format table with one row per (run, epoch)
        and an extra ``name`` column for the run-directory base name.
        Empty DataFrame if no matching runs.
    """
    paths = glob.glob(f'{dirname}/*')
    logs = []
    for path in paths:
        if constraint is not None:
            skip = False
            for c in constraint:
                if c not in path:
                    skip = True
                    break
            if skip:
                continue
        try:
            argument = json.load(open(f'{path}/arguments.json'))
            log = pd.read_csv(f'{path}/log.csv')
        except Exception:
            print(path, "does not have log.csv or arguments.json")
            continue
        log = log.drop(columns={'Unnamed: 0'})
        log['name'] = argument['name']
        log['seed'] = argument['seed']
#        if argument['seed'] == 12:
#            continue
        if seeds is not None:
            if argument['seed'] not in seeds:
                continue
        image_interval = argument['input_dim'] + argument['interval_dim']
        image_interval = image_interval / argument['step_size']
        log['image_interval'] = int(image_interval)
        log['epochs'] = argument['epochs']
        log['name_wo_seed'] = argument['name'].replace(f'S{argument["seed"]}Ep', 'Ep')
        log['lambdas'] = str(argument['lambdas'])
        log['contain_period'] = image_interval in argument['lambdas']
        log['resolution'] = argument.get('resolution', 1)
        # Post-refactor arguments.json drops 'gcpc' and 'velocity' (they're
        # implicit in the model name now); default to sensible values so
        # downstream aggregation still works on both pre- and post-refactor runs.
        log['gcpc'] = argument.get('gcpc', 'g' if str(argument.get('model', '')).startswith('VHA') else 'none')
        log['velocity'] = argument.get('velocity', 'none')
        log['default_internal_vel'] = argument.get('default_internal_vel', 1)
        log['model'] = argument.get('model', '')

        # manipulate data
#        if 11 not in argument['lambdas'] or 13 not in argument['lambdas']:
#            continue
 #       if image_interval > 20:
#            continue
        print(path)
        logs.append(log)
    dfs = pd.concat(logs)
    return dfs 


def draw_plot(ax, target, env_ids, colors, linestyle: str = '-', label=None, limits=None, split: str = 'all') -> None:
    """Plot per-environment mean +/- SEM curves on a shared axes.

    Args:
        ax: Matplotlib axes to draw into.
        target: DataFrame produced by ``get_stat``, with one row per epoch
            and columns ``<split>_<env_id>_mean`` / ``<split>_<env_id>_se``.
        env_ids: Iterable of environment IDs to plot (one curve per).
        colors: List of colours, one per environment.
        linestyle: Line style forwarded to matplotlib (default ``'-'``).
        label: Legend label for the first plotted curve. ``None`` means
            no legend entry.
        limits: Optional per-environment epoch window size; truncates
            each env to ``[limits*i, limits*(i+1))``.
        split: Split key (``'all'``, ``'seen'``, ``'unseen'``, ...) used
            to look up columns in ``target``.

    Returns:
        None. Draws one ``ax.plot`` per environment plus an SEM ribbon.
    """
    for i, env_id in enumerate(env_ids):
        if limits is not None: # limit epoch
            aggregated = target[target['epoch'] < limits*(i+1)]
            aggregated = aggregated[aggregated['epoch'] >= limits*i]
        else:
            aggregated = target
        ax.plot(aggregated['epoch'], aggregated[f'{split}_{env_id}_mean'], color=colors[i], linestyle=linestyle, label=label)
        ax.fill_between(aggregated['epoch'], aggregated[f'{split}_{env_id}_mean'] - aggregated[f'{split}_{env_id}_se'], aggregated[f'{split}_{env_id}_mean'] + aggregated[f'{split}_{env_id}_se'], color=colors[i], alpha=0.3)

def get_stat(df, split: str = 'all', target: str = 'accuracy'):
    """Group rows by epoch and return per-env mean + SEM of one metric.

    Args:
        df: Long-format DataFrame from ``read_results``.
        split: Train/test split selector (``'all'``, ``'seen'``,
            ``'unseen'``, ...).
        target: Metric name to aggregate (e.g. ``'accuracy'``, ``'loss'``).

    Returns:
        pandas.DataFrame with columns ``epoch``, ``<split>_<env>_mean``,
        ``<split>_<env>_se`` for env in ``{0, 1, 2}``. Ready to feed into
        ``draw_plot``.
    """
    aggregated = df.groupby('epoch').agg({
    f'visual/{split}/0/{target}': ['mean', 'sem'],
    f'visual/{split}/1/{target}': ['mean', 'sem'],
    f'visual/{split}/2/{target}': ['mean', 'sem']
    }).reset_index()
    aggregated.columns = ['epoch', 
                      f'{split}_0_mean', f'{split}_0_se',
                      f'{split}_1_mean', f'{split}_1_se',
                      f'{split}_2_mean', f'{split}_2_se']
    return aggregated



def calculate_average(dfs, target: str = 'accuracy'):
    """
    Mix seen/unseen splits into a single ``all/`` entry per env.

    Uses the canonical 0.8/0.2 weighting (seen_ratio = 0.8). The
    previous ``ratio`` parameter is gone -- the body hardcoded the
    weights and ignored the kwarg.
    """
    for mode in ['visual', 'mental']:
        for env in range(3):
            seen = dfs[f'{mode}/seen/{env}/{target}']
            unseen = dfs[f'{mode}/unseen/{env}/{target}']
            all = seen * 0.8 + unseen * 0.2
            dfs[f'{mode}/all/{env}/{target}'] = all
    return dfs


def compute_regression_coef(trajs, goals):
    """
    Compute regression coefficient a from Y = aX + b.
    X: true distance |start - goal|
    Y: actual distance |start - stop| (first and last trajectory indices)
    Returns slope a.
    """
    starts = np.array([t[0] for t in trajs], dtype=float)
    stops = np.array([t[-1] for t in trajs], dtype=float)
    goals_arr = np.array(goals, dtype=float).flatten()
    X = np.abs(starts - goals_arr)
    Y = np.abs(starts - stops)
    if len(X) < 2 or X.std() < 1e-10:
        return np.nan
    a, _ = np.polyfit(X, Y, 1)
    return a


def read_regression_results(dirname: str, constraints: list = [], seeds=None, epoch_str: str = '001999'):
    """
    Read run_info .pth files and compute regression coefficients per condition.
    Returns a DataFrame with columns like visual/seen/0/reg_coef, mental/all/0/reg_coef, etc.
    Call calculate_average(dfs, target='reg_coef') afterwards to populate the /all/ columns.
    """
    paths = glob.glob(f'{dirname}/*')
    logs = []
    for path in paths:
        skip = False
        for c in constraints:
            if c not in path:
                skip = True
                break
        if skip:
            continue
        try:
            argument = json.load(open(f'{path}/arguments.json'))
        except Exception:
            print(path, "does not have arguments.json")
            continue
        seed = argument['seed']
        if seeds is not None and seed not in seeds:
            continue
        run_info_files = sorted(glob.glob(f'{path}/run_info_{epoch_str}.pth'))
        if not run_info_files:
            print(f'{path}: no run_info_{epoch_str}.pth found')
            continue
        for run_info_file in run_info_files:
            epoch = int(run_info_file.split('run_info_')[1].split('.')[0])
            try:
                run_info = torch.load(run_info_file, map_location='cpu', weights_only=False)
            except Exception:
                try:
                    run_info = torch.load(run_info_file, map_location='cpu')
                except Exception:
                    print(f'Failed to load {run_info_file}')
                    continue
            log = {
                'seed': seed,
                'epoch': epoch,
                'name': argument['name'],
                'gcpc': argument.get('gcpc', ''),
                'velocity': argument.get('velocity', ''),
                'resolution': argument.get('resolution', 1),
                'default_internal_vel': argument.get('default_internal_vel', 1),
            }
            for mode in ['visual', 'mental']:
                for observe in ['seen', 'unseen']:
                    for env in range(10):
                        prefix = f'{mode}/{observe}/{env}/'
                        traj_key = f'{prefix}trajs'
                        goal_key = f'{prefix}goals'
                        if traj_key not in run_info or goal_key not in run_info:
                            break
                        trajs = run_info[traj_key]
                        goals = run_info[goal_key]
                        if len(trajs) == 0:
                            continue
                        log[f'{mode}/{observe}/{env}/reg_coef'] = compute_regression_coef(trajs, goals)
            logs.append(log)
    if not logs:
        return pd.DataFrame()
    return pd.DataFrame(logs)


def generalization1_regression(data, filename, colors=None, options={}):
    """
    Bar chart: regression coefficient (slope a from Y=aX+b) for visual vs mental trials.
    data: list of DataFrames with 'visual/all/0/reg_coef' and 'mental/all/0/reg_coef' columns.
    """
    rows = []
    breakpoint()
    for i, datum in enumerate(data):
        for _, row in datum.iterrows():
            rows.append({'seed': row['seed'], 'condition': 'visual', 'reg_coef': row.get('visual/all/0/reg_coef', np.nan), 'group': i})
            rows.append({'seed': row['seed'], 'condition': 'mental', 'reg_coef': row.get('mental/all/0/reg_coef', np.nan), 'group': i})
    df = pd.DataFrame(rows).dropna(subset=['reg_coef'])
    fig, ax = plt.subplots()
    hue = 'group' if len(data) > 1 else None
    sns.barplot(df, x='condition', y='reg_coef', hue=hue, errorbar='se', ax=ax,
                palette=colors if colors is not None else None)
    ax.set_ylabel('Regression coefficient (a)')
    ax.set_xlabel('')
    if 'ylim' in options:
        ax.set_ylim(options['ylim'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generalization2_regression(data, filename, options={}):
    """
    Bar chart: regression coefficient for seen vs unseen visual trials.
    data: list of DataFrames with 'visual/seen/0/reg_coef' and 'visual/unseen/0/reg_coef' columns.
    """
    rows = []
    for i, datum in enumerate(data):
        for _, row in datum.iterrows():
            rows.append({'seed': row['seed'], 'condition': 'seen', 'reg_coef': row.get('visual/seen/0/reg_coef', np.nan), 'group': i})
            rows.append({'seed': row['seed'], 'condition': 'unseen', 'reg_coef': row.get('visual/unseen/0/reg_coef', np.nan), 'group': i})
    df = pd.DataFrame(rows).dropna(subset=['reg_coef'])
    fig, ax = plt.subplots()
    hue = 'group' if len(data) > 1 else None
    sns.barplot(df, x='condition', y='reg_coef', hue=hue, errorbar='se', ax=ax)
    ax.set_ylabel('Regression coefficient (a)')
    ax.set_xlabel('')
    if 'ylim' in options:
        ax.set_ylim(options['ylim'])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def generalization3_regression(data, filename1, filename2, options={}):
    """
    Bar charts: regression coefficient across 3 environments.
    filename1: visual seen across env 0,1,2
    filename2: visual all (env 0 only)
    data: list of DataFrames with visual/{seen,all}/{0,1,2}/reg_coef columns.
    """
    # filename1: seen across envs
    rows = []
    for i, datum in enumerate(data):
        for _, row in datum.iterrows():
            for env in range(3):
                rows.append({'seed': row['seed'], 'env': f'Env{env+1}', 'reg_coef': row.get(f'visual/seen/{env}/reg_coef', np.nan), 'group': i})
    df = pd.DataFrame(rows).dropna(subset=['reg_coef'])
    fig, ax = plt.subplots()
    hue = 'group' if len(data) > 1 else None
    sns.barplot(df, x='env', y='reg_coef', hue=hue, errorbar='se', ax=ax)
    ax.set_ylabel('Regression coefficient (a)')
    ax.set_xlabel('Environment')
    if 'ylim' in options:
        ax.set_ylim(options['ylim'])
    plt.tight_layout()
    plt.savefig(filename1)
    plt.close()

    # filename2: all (env 0 only)
    rows = []
    for i, datum in enumerate(data):
        for _, row in datum.iterrows():
            rows.append({'seed': row['seed'], 'env': 'Env1', 'reg_coef': row.get('visual/all/0/reg_coef', np.nan), 'group': i})
    df = pd.DataFrame(rows).dropna(subset=['reg_coef'])
    fig, ax = plt.subplots()
    hue = 'group' if len(data) > 1 else None
    sns.barplot(df, x='env', y='reg_coef', hue=hue, errorbar='se', ax=ax)
    ax.set_ylabel('Regression coefficient (a)')
    ax.set_xlabel('Environment')
    if 'ylim' in options:
        ax.set_ylim(options['ylim'])
    plt.tight_layout()
    plt.savefig(filename2)
    plt.close()


def generalization1(data, epochs, filename, linestyles=None, colors=None, options={}):
    """
        Draw generalization plot.
        1. Visual vs. Mental
    """
    fig = plt.figure()
    if linestyles is None:
        linestyles = ['--', 'solid', 'dotted', 'dashdot']
#    if colors is None:
#        colors = ['black', 'red']

    for i, (datum, linestyle) in enumerate(zip(data, linestyles)):
        if colors is not None:
            c1 = c2 = colors[i]
        else:
            c1 = 'black'
            c2 = 'red'
        # filter error log
#        print(datum.index[datum.index.duplicated()])
#        datum = datum.reset_index(drop=True)
#        mask = (datum.epoch == 0) | (datum.epoch >= 100)
#        if len(mask) != mask.sum():
#            print(datum[~mask])
#            datum = datum[mask]
        sns.lineplot(datum, x='epoch', y='visual/all/0/accuracy', color=c1, linestyle=linestyle, errorbar='se')
        sns.lineplot(datum, x='epoch', y='mental/all/0/accuracy', color=c2, linestyle=linestyle, errorbar='se')

    plt.xlabel('Blocks')
    plt.ylabel('Model Performance (%)')
    plt.xlim(0, epochs)

    if 'xlim' in options:
        plt.xlim(options['xlim'])
    if 'ylim' in options:
        plt.ylim(options['ylim'])
    plt.savefig(filename)

def generalization2(data, epochs, filename, linestyles=None, options={}):
    """
        Draw generalization plot.
        2. Seen vs. Unseen
    """
    blue = (46/255,66/255,130/255)
    fig = plt.figure()
    if linestyles is None:
        linestyles = ['--', 'solid', 'dotted', 'dashdot']

    for data, linestyle in zip(data, linestyles):
        sns.lineplot(data, x='epoch', y='visual/seen/0/accuracy', color='red', linestyle=linestyle, errorbar='se')
        sns.lineplot(data, x='epoch', y='visual/unseen/0/accuracy', color=blue, linestyle=linestyle, errorbar='se')

    plt.xlim(0, epochs)
    plt.xlabel('Trials')
    plt.ylabel('Performance (%)')

    if 'xlim' in options:
        plt.xlim(options['xlim'])
    if 'ylim' in options:
        plt.ylim(options['ylim'])
    plt.savefig(filename)

def generalization3(data, epochs, filename1, filename2, linestyles=None, options={}):
    """
        Draw generalization plot.
        3. 3 envs
    """
    if linestyles is None:
        linestyles = ['--', 'solid', 'dotted', 'dashdot']
    
    fig = plt.figure()
    ax = plt.gca()
    for datum, linestyle in zip(data, linestyles):
        aggregated = get_stat(datum)
        draw_plot(ax, aggregated, [0, 1, 2], ['black', '#118241', '#72bf45'], linestyle=linestyle, label='visual/seen', limits=epochs)

    # change x axis to  Env1, Env2, Env3
    plt.xticks([epochs//2, int(1.5*epochs), int(2.5*epochs)], ['Env1', 'Env2', 'Env3'])
    plt.vlines(x=epochs, ymin=0, ymax=1, color='grey', linestyle='--')
    plt.vlines(x=2*epochs, ymin=0, ymax=1, color='grey', linestyle='--')
    plt.xlabel('Environment')
    plt.ylabel('Performance (%)')
    if 'xlim' in options:
        plt.xlim(options['xlim'])
    if 'ylim' in options:
        plt.ylim(options['ylim'])
    plt.savefig(filename1)

    fig = plt.figure()
    ax = plt.gca()
    for datum, linestyle in zip(data, linestyles):
        aggregated = get_stat(datum)
        draw_plot(ax, aggregated, [0], ['black'], linestyle=linestyle, label='visual/seen')

    # change x axis to  Env1, Env2, Env3
    plt.xticks([epochs//2, 1.5*epochs, 2.5 * epochs], ['Env1', 'Env2', 'Env3'])
    plt.vlines(x=epochs, ymin=0, ymax=1, color='grey', linestyle='--')
    plt.vlines(x=2*epochs, ymin=0, ymax=1, color='grey', linestyle='--')
    plt.xlabel('Environment')
    plt.ylabel('Performance (%)')

    if 'xlim' in options:
        plt.xlim(options['xlim'])
    if 'ylim' in options:
        plt.ylim(options['ylim'])
    plt.savefig(filename2)



def draw_generalization(dfs, dirname, epoch=2000, _format='pdf', data=None, linestyles=None, colors=None, postfix='', options={}, mode='accuracy'):
    """
        Draw generalization plot.
        1. Visual vs. Mental
        2. Seen vs. Unseen
        3. 3 envs
        mode: 'accuracy' (line plots over epochs) or 'regression' (bar charts of regression coefficient)
    """
    if mode == 'regression':
        os.makedirs(dirname, exist_ok=True)
        generalization1_regression(data, os.path.join(dirname, f'generalization1_regression{postfix}.{_format}'), colors=colors, options=options)
        generalization2_regression(data, os.path.join(dirname, f'generalization2_regression{postfix}.{_format}'), options=options)
        filename1 = os.path.join(dirname, f'generalization3-1_regression{postfix}.{_format}')
        filename2 = os.path.join(dirname, f'generalization3-2_regression{postfix}.{_format}')
        generalization3_regression(data, filename1, filename2, options=options)
        return

    if data is None:
        baseline = dfs[dfs['gcpc'] == ""].explode('epoch')
        ours = dfs[dfs['gcpc'] == "g"].explode('epoch')
        data = [baseline, ours]
    generalization1(data, epoch, os.path.join(dirname, f'generalization1{postfix}.{_format}'), linestyles=linestyles, colors=colors, options=options)
    generalization2(data, epoch, os.path.join(dirname, f'generalization2{postfix}.{_format}'), linestyles=linestyles, options=options)
    filename1 = os.path.join(dirname, f'generalization3-1{postfix}.{_format}')
    filename2 = os.path.join(dirname, f'generalization3-2{postfix}.{_format}')
    generalization3(data, epoch, filename1, filename2, linestyles=linestyles, options=options)


def draw_scaling_factor(dfs, save_dir: str, epochs: int = 5000, _format: str = 'pdf') -> None:
    """
        Draw scaling factor Performance plot. (Figure 6.b)

    """
    dfs = dfs[dfs['default_internal_vel'] == 1]
    dfs = dfs[dfs['resolution'] != 1]
    for res in [2, 3]:
        _dfs = dfs[dfs['resolution'] == res]
        baseline = _dfs[_dfs['velocity'] == 'none']
#        ours = _dfs[_dfs['velocity'] == 'internal_corr_suc_3']
        ours = _dfs[_dfs['velocity'] != 'none']
        filename = os.path.join(save_dir, f'generalization1_res{res}.{_format}')
        fig = plt.figure()
#        sns.lineplot(baseline, x='epoch', y='visual/seen/0/accuracy', color='black', linestyle='--', errorbar='se')
        sns.lineplot(baseline, x='epoch', y='mental/seen/0/accuracy', color='blue', linestyle='--', errorbar='se')

#        sns.lineplot(ours, x='epoch', y='visual/seen/0/accuracy', color='black', errorbar='se')
        sns.lineplot(ours, x='epoch', y='mental/seen/0/accuracy', color='green', errorbar='se')
        plt.xlabel('Trials')
        plt.ylabel('Performance (%)')
        plt.xlim(0, epochs)
        plt.savefig(filename)


#        generalization1(baseline, ours, 5000, filename)
        generalization2([baseline, ours], epochs, os.path.join(save_dir, f'generalization2_res{res}.{_format}'))
        filename1 = os.path.join(save_dir, f'generalization3-1_res{res}.{_format}')
        filename2 = os.path.join(save_dir, f'generalization3-2_res{res}.{_format}')
        generalization3([baseline, ours], epochs, filename1, filename2)

        continue

        fig = plt.figure()
        ax = plt.gca()
        draw_plot(ax, get_stat(baseline), [0, 1, 2], ['green', 'blue', 'red'], linestyle='--', label='baseline')
        draw_plot(ax, get_stat(baseline), [0, 1, 2], ['green', 'blue', 'red'], linestyle='--', label='baseline')

        unique_lambdas = ours.lambdas.unique()
        linestyles = ['solid', 'dotted', 'dashdot'][:len(unique_lambdas)]
        for ls, lambdas in zip(linestyles, unique_lambdas):
            _ours = ours[ours['lambdas'] == lambdas]
            draw_plot(ax, get_stat(_ours), [0, 1, 2], ['green', 'blue', 'red'], label=lambdas, linestyle=ls)

        plt.xlabel('Trials')
        plt.ylabel('Performance (%)')

        plt.xticks([2500, 7500, 12500], ['Env1', 'Env2', 'Env3'])
        plt.vlines(x=5000, ymin=0, ymax=1, color='grey', linestyle='--')
        plt.vlines(x=10000, ymin=0, ymax=1, color='grey', linestyle='--')
        plt.xlabel('Environment')
        plt.ylabel('Performance (%)')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'scaling_factor_{res}.{_format}'), format=_format)


def draw_periodicity(dfs, save_dir: str, vis_env1_only: bool = True, _format: str = 'pdf') -> None:
    """
        Draw scaling factor plot.

    """
    # image interval
    if vis_env1_only:
        dfs = dfs[dfs.epoch < 2000]
    dfs = dfs[dfs.image_interval != 24]
    baseline = dfs[~dfs.contain_period]
    ours = dfs[dfs.contain_period]
    fig = plt.figure()
    ax = plt.gca()
    draw_plot(ax, get_stat(baseline), [0], ['blue'], linestyle='solid', label='not contain')
    draw_plot(ax, get_stat(ours), [0], ['red'], linestyle='solid', label='contain')
    plt.xlabel('Trials')
    plt.ylabel('Performance (%)')
    
    if not vis_env1_only:
        plt.xticks([1000, 3000, 5000], ['Env1', 'Env2', 'Env3'])
        plt.vlines(x=2000, ymin=0, ymax=1, color='grey', linestyle='--')
        plt.vlines(x=4000, ymin=0, ymax=1, color='grey', linestyle='--')
        plt.xlabel('Environment')
    plt.ylabel('Performance (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'periodiy_accum.{_format}'), format=_format)

    image_intervals = dfs.image_interval.unique()
    for interval in image_intervals:
        # group by lambdas.

        fig = plt.figure()
        ax = plt.gca()
        _dfs = dfs[dfs.image_interval == interval]
        lambdas = _dfs.lambdas.unique()
        contain = []
        not_contain = []
        mod_contain_up = []
        mod_contain_down = []
        for lamb in lambdas:
            _lamb = eval(lamb)
            if interval in eval(lamb):
                contain.append(lamb)
            else:
                # check mod contain
                mod_up = False
                mod_down = False
                for j in range(2, 17):
                    if j*interval in _lamb:
                        mod_up = True
                        break
                for j in range(2, 17):
                    if interval % j == 0 and interval/j in _lamb:
                        mod_down = True
                        break
                if mod_up:
                    mod_contain_up.append(lamb)
                elif mod_down:
                    mod_contain_down.append(lamb)
                else:
                    not_contain.append(lamb)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'brown', 'pink', 'grey', 'yellow']
        for i, nc in enumerate(not_contain):
            __dfs = _dfs[_dfs.lambdas == nc]
            draw_plot(ax, get_stat(__dfs), [0], ['grey'], linestyle=(0, (i+1, 1)), label=nc)

        for i, nc in enumerate(mod_contain_up):
            __dfs = _dfs[_dfs.lambdas == nc]
            draw_plot(ax, get_stat(__dfs), [0], ['blue'], linestyle=(0, (i+1, 1)), label=nc)
        for i, nc in enumerate(mod_contain_down):
            __dfs = _dfs[_dfs.lambdas == nc]
            draw_plot(ax, get_stat(__dfs), [0], ['green'], linestyle=(0, (i+1, 1)), label=nc)
        for c in contain:
            __dfs = _dfs[_dfs.lambdas == c]
            draw_plot(ax, get_stat(__dfs), [0], ['red'], linestyle='solid', label=c)
        
        if vis_env1_only:
            plt.xlabel('Trials')
        else:
            plt.xticks([1000, 3000, 5000], ['Env1', 'Env2', 'Env3'])
            plt.vlines(x=2000, ymin=0, ymax=1, color='grey', linestyle='--')
            plt.vlines(x=4000, ymin=0, ymax=1, color='grey', linestyle='--')
            plt.xlabel('Environment')
        plt.ylabel('Performance (%)')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'periodiy_{interval}.png'))
#    breakpoint()


 

def run_analysis(dfs, dirname: str, save_dir: str, use_cuda: bool = True, target: str = 'hidden', save_indiv: bool = False, _format: str = 'pdf', visual_only: bool = False, constraints: list = [], force: bool = False, options: dict = {}) -> None:
    """Dispatch per-seed analyses for every run in ``dfs`` that matches ``constraints``.

    For each matching run, loads its ``run_info_*.pth`` checkpoint(s),
    rebuilds the run-info dict, and calls ``run_visualization`` to produce
    the TDR / PCA / firing-rate / autocorr / regression / neuron figures.

    Args:
        dfs: DataFrame from ``read_results`` enumerating candidate runs.
        dirname: Parent directory holding the candidate runs.
        save_dir: Output root for the per-run analysis dumps.
        use_cuda: If True, move loaded tensors to GPU before analysis.
        target: Hidden-state field passed through to ``run_visualization``
            (``'hidden'``, ``'gs'``, ``'ps'``, ...).
        save_indiv: Forwarded; if True, firing-rate plots also write per-neuron figures.
        _format: Figure extension forwarded to the analyses.
        visual_only: If True, restrict to the ``visual`` mode.
        constraints: List of substrings filtering which runs are analyzed.
        force: If True, re-run analyses even if their outputs already exist.
        options: Per-analysis option bundles forwarded into ``run_visualization``.

    Returns:
        None. Writes one analysis sub-directory per matching run under ``save_dir``.
    """
    dirs = sorted(dfs.name.unique())
    dirs = filter_path(dirs, constraints)
    for _fname in dirs: #[len(dirs)//2:]:
#        if 'random_vector_ctrnn_N6L384I384Step64A0.9lr0.001F256S13Ep5000_CTRes2SubpositiveE256_gNs384Np400D1Sig0.0_22_24_26' in _fname:
#            breakpoint()
        path = os.path.join(dirname, _fname)
        argument = json.load(open(f'{path}/arguments.json'))
        flist = sorted(glob.glob(os.path.join(path, 'run_info*.pth')))
        for fname in flist[:1]:
            if use_cuda and torch.cuda.is_available():
                run_info = torch.load(fname, weights_only=False)
            else:
                run_info = torch.load(fname, map_location='cpu')
            # read run_info
            epoch = fname.split('/run_info_')[1].split('.')[0]
            _save_dir = os.path.join(save_dir, 'dynamics',  _fname, epoch)
            if os.path.exists(os.path.join(_save_dir, f'mental__0_{target}')) or os.path.exists(os.path.join(_save_dir, f'visual__0_{target}')):
                print(f'{_save_dir} already exists')
                if not force:
                    continue
            print(fname)
            os.makedirs(_save_dir, exist_ok=True)
            # In case of scaling factor.
            argument['suc_only'] = options.get('suc_only', True)
            argument['uni_only'] = options.get('uni_only', True)
            argument['hidden'] = target
            save_indiv = '1999' in fname
            run_visualization(run_info, argument, _save_dir, save_indiv=save_indiv, _format=_format, visual_only=visual_only, options=options)


def run_shifted_analysis(dfs, dirname: str, save_dir: str, target: str = 'hidden', _format: str = 'pdf', constraints: list = [], options: dict = {}) -> None:
    """
    Independent of run_visualization: loads run_info and calls run_shifted_visualization
    (which calls run_firing_rate_shifted) without re-running the full analysis pipeline.

    Usage:
        dfs = read_results(dirname, constraints, seeds=[43])
        run_shifted_analysis(dfs, dirname, save_dir, target='hidden', options={
            'firing_rate': {'ignore_last': True, 'smoothing_sigma': 2, 'smoothing_truncate': 2}
        })
    """
    dirs = sorted(dfs.name.unique())
    dirs = filter_path(dirs, constraints)
    for _fname in dirs:
        path = os.path.join(dirname, _fname)
        argument = json.load(open(f'{path}/arguments.json'))
        flist = sorted(glob.glob(os.path.join(path, 'run_info*.pth')))
        for fname in flist[:1]:
            run_info = torch.load(fname, map_location='cpu', weights_only=False)
            epoch = fname.split('/run_info_')[1].split('.')[0]
            _save_dir = os.path.join(save_dir, 'dynamics', _fname, epoch)
            os.makedirs(_save_dir, exist_ok=True)
            argument['hidden'] = target
            run_shifted_visualization(run_info, _save_dir, argument, _format=_format, options=options)


def run_stretched_analysis(dfs, dirname: str, save_dir: str, target: str = 'hidden', _format: str = 'pdf', constraints: list = [], options: dict = {}) -> None:
    """
    Independent of run_visualization: loads run_info and calls run_stretched_visualization
    (which calls run_firing_rate_stretched). No shifting — only X/Y scaling.

    Usage:
        dfs = read_results(dirname, constraints, seeds=[43])
        run_stretched_analysis(dfs, dirname, save_dir, target='hidden', options={
            'firing_rate': {'ignore_last': True, 'smoothing_sigma': 2, 'smoothing_truncate': 2,
                            'sx_range': np.linspace(0.5, 4.0, 71)}
        })
    """
    dirs = sorted(dfs.name.unique())
    dirs = filter_path(dirs, constraints)
    for _fname in dirs:
        path = os.path.join(dirname, _fname)
        argument = json.load(open(f'{path}/arguments.json'))
        flist = sorted(glob.glob(os.path.join(path, 'run_info*.pth')))
        for fname in flist[:1]:
            run_info  = torch.load(fname, map_location='cpu', weights_only=False)
            epoch     = fname.split('/run_info_')[1].split('.')[0]
            _save_dir = os.path.join(save_dir, 'dynamics', _fname, epoch)
            os.makedirs(_save_dir, exist_ok=True)
            argument['hidden'] = target
            run_stretched_visualization(run_info, _save_dir, argument, _format=_format, options=options)


def filter_path(paths, constraint):
    """
    Filter paths based on the constraint.
    """
    if constraint is not None:
        new_paths = []
        for path in paths:
            skip = False
            for c in constraint:
                if c not in path:
                    skip = True
                    break
            if not skip:
                new_paths.append(path)
        paths = new_paths
    return paths

def draw_aggregated_periodicity(seeds, dirname, constraint=[], target='mental__0_hidden', resolutions=[2,3], save_dir='./', _format='pdf', use_half_only=False, use_long_only=False, mode='kde', target_seeds=[11], postfix=''):
    """
    Draw aggregated periodicity for the given target.
    """
    graph_mode = mode

    def _draw_aggregated_periodicity(name, names_wo_seed, colormap_name, paths, ax, ax_mode):
        indices = np.where(names_wo_seed == name)[0]
        print(name)
        periodicities = []
        if seeds is None or len(seeds) == 0:
            colors = sns.color_palette(colormap_name, len(indices)+1)[1:]
        else:
            colors = sns.color_palette(colormap_name, len(seeds)+1)[1:]
        j = 0
        modes = []
        n_zero_total = 0   # count of avg_period==0 units aggregated across seeds
        n_units_total = 0  # total avg_period entries (denominator for the fraction)
        for index in indices:
            # filter based on seeds
            seed = int(paths[index].split('/')[-6].split('Ep')[0].split('S')[-1])
            if seeds is not None and len(seeds) > 0 and seed not in seeds:
                continue

            period = np.load(paths[index])
            if use_long_only: # only consider longer distances.
                period = period[2:]

            avg_period = period.sum(0) / ((period > 0).sum(0) + 1e-16)
            n_zero_total += int((avg_period == 0).sum())
            n_units_total += int(avg_period.size)
            print(paths[index], (avg_period == 0).sum(), (avg_period == 0).mean())
            avg_period = avg_period[avg_period > 0]

            if seed in target_seeds:
                seed_fig = plt.figure()
                seed_ax = seed_fig.gca()
                plt.xlabel('Perioidicity (a.u.)')
                plt.ylabel('# of RNN units')
                locs, labels = plt.xticks()
                n, bins, patches = plt.hist(avg_period, bins=np.arange(avg_period.max()+1), color='black', alpha=0.2)
#                seed_ax.hist(avg_period, range=(0,avg_period.max()+1), color='black', alpha=0.2)

                for g in [11, 12, 13]:
                    seed_ax.axvline(x=g+0.5, color='blue')
                seed_ax.axvline(x=12 + 0.5, color='red', linestyle='--')

                locs, labels = seed_ax.get_xticks(), seed_ax.get_xticklabels()
                seed_ax.set_xticks([e+0.5 for e in locs[1:]], [int(e) for e in locs[1:]])
                seed_ax.set_xlim(-0.5, 36)
                if 'Res' in name:
                    res = name.split('Res')[1][0]
                    seed_fig.savefig(f'{save_dir}/S{seed}Res{res}_periodicity_{half}{target}{postfix}.{_format}', format=_format)
                else:
                    seed_fig.savefig(f'{save_dir}/S{seed}_periodicity_{half}{target}{postfix}.{_format}', format=_format)
                
            mode = stats.mode(avg_period)[0]
            modes.append(float(mode))
            periodicities.append(period)
#            ax.hist(avg_period, bins=np.arange(avg_period.max()+1), alpha=0.2, label=name, color=colors[j])
            if graph_mode == 'kde':
                sns.kdeplot(avg_period, ax=ax, label=name, color=colors[j])
            elif graph_mode == 'hist':
#                sns.histplot(avg_period, ax=ax, label=name, bins=np.arange(avg_period.max()+1), color=colors[j], alpha=0.2, binwidth=1)
                sns.histplot(avg_period, ax=ax, label=name, binrange=(0, avg_period.max()+1), color=colors[j], alpha=0.2, binwidth=1)
            j += 1
        if len(seeds) != j:
            print("Warning: number of seeds does not match the number of periodicities drawn.", seeds, len(seeds), j)

        if len(modes) > 0:
            sns.histplot(modes, ax=ax_mode, label=name, binrange=(0, max(modes)+1), binwidth=1, alpha=0.2)

        return modes, n_zero_total, n_units_total

    colormap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'YlOrBr', 'YlGnBu']

    if use_half_only:
        half = "half_"
    else:
        half = ""

    paths = glob.glob(f'{dirname}/dynamics/*/{target}/{half}periodicity/*/*.npy')
    paths += glob.glob(f'{dirname}/dynamics/*/*/{target}/{half}periodicity/*/*.npy')
    paths = filter_path(paths, constraint)
    print(paths)
    names = [e.split('/')[-6] for e in paths]

    names_wo_seed = []
    for name in names:
        seed = int(name.split('Ep')[0].split('S')[-1])
        _name = name.replace(f'S{seed}Ep', 'Ep')
        names_wo_seed.append(_name)
    names_wo_seed = np.asarray(names_wo_seed)

    unique_name = np.unique(names_wo_seed)
    # read all periodicity for each name
    aggs = []
    # total
    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.gca()
    ax_mode = fig2.gca()
    i = 0
    zero_agg, unit_agg = 0, 0
    for name in unique_name:
        modes, n_zero, n_units = _draw_aggregated_periodicity(
            name, names_wo_seed, colormap_names[i], paths, ax, ax_mode)
        zero_agg += n_zero
        unit_agg += n_units
        i = (i + 1)
    if mode == 'hist':
        offset = 0.5
    else:
        offset = 0
        ax.set_ylim(0, 0.5)
    ax.axvline(x=12 + offset, color='red', linestyle='--')
    ax_mode.axvline(x=12 + offset, color='red', linestyle='--')
    ax.set_xlim(0, 40)
    locs, labels = ax.get_xticks(), ax.get_xticklabels()
    ax.set_xticks([e+offset for e in locs[1:]], [int(e) for e in locs[1:]])
    locs, labels = ax_mode.get_xticks(), ax_mode.get_xticklabels()
    ax_mode.set_xticks([e+offset for e in locs[1:]], [int(e) for e in locs[1:]])
    ax_mode.set_ylim(0, len(seeds))
    # Report the count of RNN units with zero avg periodicity (already filtered
    # out of the KDE / hist above) so the reader can see how many units the
    # curve excludes.
    zero_title = f'# zero units = {zero_agg} / {unit_agg}' if unit_agg else 'no data'
    ax.set_title(zero_title)
    ax_mode.set_title(zero_title)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f'{save_dir}/aggregated_periodicity_{half}{target}{postfix}.{_format}', format=_format)
    # mode_periodicity_<target>.pdf intentionally not written -- only the
    # aggregated variant is used in the paper.
    print(f"Save {save_dir}/aggregated_periodicity_{half}{target}{postfix}.{_format}")

    # different resolution
    for res in resolutions:
        # figure
        fig = plt.figure()
        ax = fig.gca()
        fig2 = plt.figure()
        ax_mode = fig2.gca()
        i = 0
        zero_agg_res, unit_agg_res = 0, 0
        for name in unique_name:
            if f'Res{res}' not in name or f'{11*res}_{12*res}' not in name or 'def' in name:
                continue
            _, n_zero_r, n_units_r = _draw_aggregated_periodicity(
                name, names_wo_seed, colormap_names[i], paths, ax, ax_mode)
            zero_agg_res += n_zero_r
            unit_agg_res += n_units_r
            i = (i + 1)
            # histogram for each periodicity
        if mode == 'hist':
            offset = 0.5
        else:
            offset = 0
            ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 40)
        ax_mode.set_xlim(0, 25)

        locs, labels = ax.get_xticks(), ax.get_xticklabels()
        ax.set_xticks([e+offset for e in locs[1:]], [int(e) for e in locs[1:]])
        locs, labels = ax_mode.get_xticks(), ax_mode.get_xticklabels()
        ax_mode.set_xticks([e+offset for e in locs[1:]], [int(e) for e in locs[1:]])

        ax.axvline(x=12+offset, color='red', linestyle='--')
        ax_mode.axvline(x=12 + offset, color='red', linestyle='--')
        ax_mode.set_ylim(0, len(seeds))
        zero_title_res = (f'# zero units = {zero_agg_res} / {unit_agg_res}'
                          if unit_agg_res else 'no data')
        ax.set_title(zero_title_res)
        ax_mode.set_title(zero_title_res)
        fig.savefig(f'{save_dir}/aggregated_periodicity_res{res}_{half}{target}{postfix}.{_format}', format=_format)
        # mode_periodicity_res<res>_<target>.pdf intentionally not written.
        print(f"Save {save_dir}/aggregated_periodicity_res{res}_{half}{target}{postfix}.{_format}")
        fig.clear()
        fig2.clear()


def export_run_info_to_mat(dirname: str, constraint: list, save_dir: str = './') -> None:
    """Export tagged per-seed run-info dumps to MATLAB ``.mat`` for collaborators.

    Iterates over runs in ``dirname`` matching every substring in ``constraint``,
    loads each ``run_info_*.pth``, and writes a flattened ``.mat`` snapshot
    into ``save_dir``.

    Args:
        dirname: Parent directory holding the candidate runs.
        constraint: List of substrings; only runs whose path contains all
            of them are exported.
        save_dir: Output directory for the ``.mat`` files.

    Returns:
        None.
    """
    paths = glob.glob(f'{dirname}/*')
    print(len(paths))
    total_data = {}
    for path in paths:
        if constraint is not None:
            skip = False
            for c in constraint:
                if c not in path:
                    skip = True
                    break
            if skip:
                continue
        data = {}
        seed = int(path.split('Ep')[0].split('S')[-1])
        lambdas = [int(e) for e in path.split('Noise')[0].split('_')[-3:]]
#        lambdas = [int(e) for e in path.split('_')[-3:]]
        run_info = glob.glob(f'{path}/run_info_001999.pth')
#        run_info = glob.glob(f'{path}/run_info_003999.pth')
        save_path = path.replace(f'S{seed}Ep', 'Ep').split('/')[-1] + '.mat'
        save_path = path.split('/')[-1] + '.mat'
        save_path = os.path.join(save_dir, save_path)
        if len(run_info) == 0:
            continue
        model_path = run_info[0].replace('run_info_', '')
        model_weights = torch.load(model_path, map_location='cpu')['model']

        if 'dist_classifier.weight' in model_weights:
            data['dist_classifier.weight'] = model_weights['dist_classifier.weight'].numpy()
            data['dist_classifier.bias'] = model_weights['dist_classifier.bias'].numpy()
        data['classifier.weight'] = model_weights['classifier.weight'].numpy()
        data['classifier.bias'] = model_weights['classifier.bias'].numpy()
        map_loc = 'cpu' if not torch.cuda.is_available() else 'cuda'
        try:
            run_info = torch.load(run_info[0], map_location=map_loc)
        except Exception:
            run_info = torch.load(run_info[0], weights_only=False, map_location=map_loc)
        

        # suc_only, uni_only
        arguments = sorted(set([e.split('/')[-1] for e in run_info.keys()]))
        print(path)
        mode = 'visual'
        for argname in arguments:
            datum = []
            env_ids = []
            is_seen = []
            arr = None
            for env in range(1000):
                for observe in ['seen', 'unseen']:
                    key = f'{mode}/{observe}/{env}/{argname}'
                    if key not in run_info:
                        break
                    if len(run_info[key]) == 0 or (type(run_info[key][0]) is list and  run_info[key][0] == []):
                        break
                    d = run_info[key]
                    if argname == 'gs':
                        d = [e[:, :3] for e in d]
                        ds = []
                        for _d in d:
                            if _d.max() > max(lambdas):
                                breakpoint()
                            _d[:,1:] += lambdas[0]
                            _d[:,2] += lambdas[1]
                            new_d = F.one_hot(torch.from_numpy(_d), sum(lambdas)).sum(1)
                            ds.append(new_d)
                        d = ds
                    elif argname == 'trajs':
                        d = [np.array([e[0], e[-1]]) for e in d]
                    elif argname == 'ps':
                        d = [e[:,:,1] for e in d]

                    datum.extend(d)
                    is_seen += [1 if observe == 'seen' else 0] * len(d)
                    env_ids += [env] * len(d)
                # concatenate
            if len(datum) == 0:
                continue
            if type(datum[0]) is np.ndarray:
                Ls = [len(e) for e in datum]
                arr = -np.ones([len(datum), max(Ls)] + list(datum[0].shape[1:])) * 12345
                for i, e in enumerate(datum):
                    arr[i, -len(e):] = e
                # padding
                arr[arr == -12345] = np.nan
                datum = arr
            elif type(datum[0]) is torch.Tensor:
                Ls = [len(e) for e in datum]
                if datum[0].ndim != 1:
                    arr = -torch.ones([len(datum), max(Ls)] + list(datum[0].shape[1:])) * 12345
                    for i, e in enumerate(datum):
                        arr[i, -len(e):] = e
                    arr[arr == -12345] = torch.nan
                    if arr.shape[2] == 1:
                        arr = arr[:,:,0]
                else:
                    breakpoint()
                arr = arr.numpy()
            else:
                arr = np.asarray(datum).reshape(len(datum), -1)

            data[argname] = arr
        total_data[str(seed)] = data
        scipy.io.savemat(save_path, total_data[str(seed)])
#        scipy.io.savemat(save_path, total_data[str(seed)])


