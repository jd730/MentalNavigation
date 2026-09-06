"""Orchestrators that drive the analysis pipeline.

`run_visualization` is the lib version, called by lib.analyze_utils.collect
with an arguments dict; `run_visualization_cli` is the argparse-Namespace
version, called by train.py and lib.exp during training.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .dim_reduction import run_non_linear_reduction
from .autocorr import run_autocorr
from .regression import run_regression
from .firing_rate import (
    run_firing_rate,
    run_firing_rate_each_trial,
    run_firing_rate_shifted,
    run_firing_rate_stretched,
)
from .neurons import analyze_speed, analyze_var


# ---------------------------------------------------------------------------
# Per-env analysis helpers
#
# Each helper does ONE analysis for ONE (env_id, mode, observe) tuple, taking
# the prepared `hiddens` / `new_traj` / `new_goals` and any other state it
# needs. `run_visualization` below picks which helpers to call via the
# `arguments['analyses']` selector (defaulting to the legacy vis_corr /
# vis_tdr / vis_pca flags for backward compat). This is what makes
# regenerating one analysis without re-running the others possible.
# ---------------------------------------------------------------------------


def _run_corr_per_env(hiddens, new_traj, save_dir, *, image_interval,
                      auto_corr_filename, lambdas, save_indiv, _format, options):
    """Autocorrelation + firing-rate for one (env, mode, observe) tuple.

    Honors ``options['autocorr'].separate_direction`` and
    ``options['firing_rate'].separate_direction`` to split left- and
    right-going trials into separate runs of the underlying analyses.
    Preserved breakpoints + un-onset/onset firing-rate pairing match the
    pre-refactor behavior.
    """
    breakpoint()
    direction = [traj[-1] > traj[0] for traj in new_traj]
    vis = True
    if options.get('autocorr', {}).get('separate_direction', False):
        left_hiddens = [h for h, d in zip(hiddens, direction) if not d]
        right_hiddens = [h for h, d in zip(hiddens, direction) if d]
        option = options.get('autocorr', {}).copy()
        option['direction'] = 'left'
        if len(left_hiddens) > 0:
            run_autocorr(left_hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, options=option)
        option['direction'] = 'right'
        if len(right_hiddens) > 0:
            run_autocorr(right_hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, options=option)
    else:
#                    analyze_neurons(hiddens, save_dir, _format=_format)
        run_autocorr(hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, options=options.get('autocorr', {}))
#                        run_autocorr(hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, first_half=True, options=options.get('autocorr', {}))
    if options.get('firing_rate', {}).get('separate_direction', False):
        left_hiddens = [h for h, d in zip(hiddens, direction) if not d]
        right_hiddens = [h for h, d in zip(hiddens, direction) if d]
        breakpoint()
        option = options.get('firing_rate', {}).copy()
        option['direction'] = 'left'
        if len(left_hiddens) > 0:
            run_firing_rate(left_hiddens, save_dir, save_indiv=save_indiv, _format=_format, options=option)
            run_firing_rate(left_hiddens, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=option)
        option['direction'] = 'right'
        if len(right_hiddens) > 0:
            run_firing_rate(right_hiddens, save_dir, save_indiv=save_indiv, _format=_format, options=option)
            run_firing_rate(right_hiddens, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=option)
    else:
        run_firing_rate(hiddens, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=options.get('firing_rate', {}))
#                    run_firing_rate(hiddens, save_dir, _format=_format)
        run_firing_rate(hiddens, save_dir, save_indiv=save_indiv, _format=_format, options=options.get('firing_rate', {}))


#: Registry of per-env analyses. ``arguments['analyses']`` in run_visualization
#: is a list of these names; the helper for each runs once per env. Kept
#: separate from the helpers themselves so collect_abs_result.py can
#: introspect the set without importing them.
ANALYSES = ('corr',)


def run_visualization(run_info, arguments={}, _save_dir='./', save_indiv=False, _format='pdf', visual_only=False, options={}):
    """Run the full per-seed analysis sweep from a ``run_info`` dump (library entry).

    Iterates over (mode, observe, env_id) keys in ``run_info`` and
    dispatches to the per-theme analyses (TDR, PCA, firing rate,
    autocorrelation, regression, neuron analyses).

    Args:
        run_info: Dict written by training; keys follow the pattern
            ``'<mode>/<observe>/<env_id>/<field>'`` with fields like
            ``hidden``, ``actions``, ``trajs``, ``suc``, ``unidirection``.
        arguments: Caller-side options dict — supported keys include
            ``analyses`` (list[str]: a subset of ``orchestration.ANALYSES``
            naming which per-env analyses to run; lets you regenerate one
            without re-running the others), ``vis_corr`` (legacy bool: when
            ``analyses`` is None, it is translated into ``['corr']``),
            ``suc_only``, ``uni_only``, ``hidden`` (which hidden field to
            read), ``merge_seen_unseen``, ``lambdas``, ``epochs``,
            ``input_dim``, ``interval_dim``, ``step_size``, ``seed``.
        _save_dir: Output root; per-(mode, observe, env_id) subdirectories
            are created beneath it.
        save_indiv: Forwarded to firing-rate plots; if True, also writes
            per-neuron individual figures.
        _format: Figure extension forwarded to all plotting helpers.
        visual_only: If True, restrict to the ``visual`` mode only.
        options: Per-analysis option bundles; supported keys are
            ``firing_rate``, ``autocorr``.

    Returns:
        None. Writes the per-analysis figures and caches into ``_save_dir``.
    """
    suc_only = arguments.get('suc_only', True)
    uni_only = arguments.get('uni_only', True)
    vis_corr = arguments.get('vis_corr', True)
    hidden_name = arguments.get('hidden', 'hidden')
    merge_seen_unseen = arguments.get('merge_seen_unseen', True)
    lambdas = arguments.get('lambdas', [11, 12, 13])
    epochs = arguments.get('epochs', 5000)
    input_dim = arguments.get('input_dim', 384)
    interval_dim = arguments.get('interval_dim', 384)
    step_size = arguments.get('step_size', 64)

    image_interval = (input_dim + interval_dim) // step_size

    auto_corr_filename = f'{image_interval}_{arguments["seed"]}_{input_dim}_{interval_dim}_{step_size}.npy'

    # Pick which analyses to run. New callers can pass arguments['analyses'];
    # for back-compat we translate the legacy vis_corr boolean into the same
    # list shape. (vis_tdr / vis_pca were dropped along with the TDR and PCA
    # per-env helpers - the analyses are no longer wired through this entry.)
    analyses = arguments.get('analyses', None)
    if analyses is None:
        analyses = []
        if vis_corr:
            analyses.append('corr')
    for env_id in range(1000):
        if f'visual/seen/{env_id}/{hidden_name}' not in run_info:
            break
        if visual_only:
            modes = ['visual']
        else:
            modes = ['visual', 'mental']
        for mode in modes:
            for observe in ['seen', 'unseen']:
                prefix = f'{mode}/{observe}/{env_id}'
                print(prefix)
                hidden = run_info[f'{prefix}/{hidden_name}']
                actions = run_info[f'{prefix}/actions']
                unidirection = run_info[f'{prefix}/unidirection']
                trajs = run_info[f'{prefix}/trajs']
                success = run_info[f'{prefix}/suc']
                goals = run_info.get(f'{prefix}/goals', None)
                if merge_seen_unseen:
                    if observe == 'unseen':
                        continue
                    prefix = f'{mode}/unseen/{env_id}'
                    print(prefix)
                    hidden += run_info[f'{prefix}/{hidden_name}']
                    actions += run_info[f'{prefix}/actions']
                    unidirection += run_info[f'{prefix}/unidirection']
                    trajs += run_info[f'{prefix}/trajs']
                    success += run_info[f'{prefix}/suc']
                    _goals = run_info.get(f'{prefix}/goals', None)
                    if _goals is not None and goals is not None:
                        goals += _goals 

                hiddens = []
                new_traj = []
                new_goals = []
                print(success, unidirection)
                for i, (hid, suc, uni, traj) in enumerate(zip(hidden, success, unidirection, trajs)):
                    if (not suc_only or suc) and (not uni_only or uni):
                        hiddens.append(hid)
                        new_traj.append(traj)
                        print(mode, observe, env_id, len(traj), traj[0], traj[-1])
                        if goals is not None:
                            new_goals.append(goals[i])
#                        print(len(hid), len(traj))
                
                if hidden_name == 'imgs':
                    hiddens = [e[:, :, :1] for e in hiddens]
                if hidden_name == 'ps' or hidden_name == 'images':
                    hiddens = [e.transpose(0, 2, 1).reshape(len(e), 1, -1) for e in hiddens]
                elif hidden_name != 'hidden':
                    hiddens = [e.reshape(len(e), 1, -1) for e in hiddens]
                print(len(hiddens))
                if len(hiddens) == 0:
                    continue
                if type(hiddens[0]) is torch.Tensor and hiddens[0].device.type == 'cuda':
                    hiddens = [h.cpu() for h in hiddens]
#                _name = name.split('_')[-1].replace('.pth','') + '_' + prefix.replace('/', '_') + '_' + hidden_name
                _name = prefix.replace('/', '_') + '_' + hidden_name
                if merge_seen_unseen:
                    _name = _name.replace('unseen','').replace('seen','')


                vis = True
                if hidden_name == 'gs': # and False:
                    one_hot_hiddens = []
                    # convert one_hot encoding
                    original_hiddens = hiddens
                    for hid in hiddens:
                        _hid = hid[:,0, :len(lambdas)].copy()
                        _hid[:, 1:] += lambdas[0]
                        _hid[:, 2:] += lambdas[1]
                        one_hot_hiddens.append(F.one_hot(torch.from_numpy(_hid).long(), sum(lambdas)).sum(1).float().unsqueeze(1))
                    hiddens = one_hot_hiddens                    
                    save_dir = os.path.join(_save_dir, _name + '_one_hot')
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    print(save_dir)
                    if False:
                        analyze_speed(hiddens, save_dir)

                    if vis_corr:
#                        analyze_neurons(hiddens, save_dir)
#                        run_firing_rate(hiddens, save_dir, is_onset=True)
                        run_firing_rate_each_trial(hiddens, new_traj, save_dir, _format=_format, options=options.get('firing_rate', {}))
                        run_firing_rate(hiddens, save_dir, save_indiv=save_indiv, _format=_format, options=options.get('firing_rate', {}))
                        run_regression(save_dir, epochs, _format=_format)
                        run_autocorr(hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, options=options.get('autocorr', {}))
#                        run_autocorr(hiddens, save_dir, vis, image_interval, auto_corr_filename, lambdas, _format=_format, first_half=True, options=options.get('autocorr', {}))

                    hiddens = original_hiddens

                save_dir = os.path.join(_save_dir, _name)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                print(save_dir)


                if True and False:
                    # use hiden 
                    analyze_var(hiddens)
                    analyze_speed(hiddens, save_dir)


                if 'corr' in analyses:
                    _run_corr_per_env(
                        hiddens, new_traj, save_dir,
                        image_interval=image_interval,
                        auto_corr_filename=auto_corr_filename,
                        lambdas=lambdas, save_indiv=save_indiv,
                        _format=_format, options=options,
                    )
                    continue

def extract_hiddens(run_info, hidden_name='hidden', suc_only=True, uni_only=True,
                    merge_seen_unseen=True, visual_only=True):
    """
    Extract (hiddens, trajs) from run_info using the same filtering logic as run_visualization.
    Returns a dict keyed by (mode, env_id) -> (hiddens_list, trajs_list).
    """
    results = {}
    for env_id in range(1000):
        if f'visual/seen/{env_id}/{hidden_name}' not in run_info:
            break
        modes = ['visual'] if visual_only else ['visual', 'mental']
        for mode in modes:
            for observe in ['seen', 'unseen']:
                if merge_seen_unseen and observe == 'unseen':
                    continue
                prefix = f'{mode}/{observe}/{env_id}'
                hidden     = list(run_info[f'{prefix}/{hidden_name}'])
                unidirect  = list(run_info[f'{prefix}/unidirection'])
                trajs      = list(run_info[f'{prefix}/trajs'])
                success    = list(run_info[f'{prefix}/suc'])
                if merge_seen_unseen:
                    _prefix = f'{mode}/unseen/{env_id}'
                    hidden    += list(run_info[f'{_prefix}/{hidden_name}'])
                    unidirect += list(run_info[f'{_prefix}/unidirection'])
                    trajs     += list(run_info[f'{_prefix}/trajs'])
                    success   += list(run_info[f'{_prefix}/suc'])

                hiddens, new_traj = [], []
                for hid, suc, uni, traj in zip(hidden, success, unidirect, trajs):
                    if (not suc_only or suc) and (not uni_only or uni):
                        hiddens.append(hid)
                        new_traj.append(traj)

                if hidden_name == 'ps' or hidden_name == 'images':
                    hiddens = [e.transpose(0, 2, 1).reshape(len(e), 1, -1) for e in hiddens]
                elif hidden_name != 'hidden':
                    hiddens = [e.reshape(len(e), 1, -1) for e in hiddens]

                # do NOT strip [:-1] here — run_visualization does not do this.
                # run_firing_rate handles ignore_last internally (strips [:-2]).
#                hiddens = [e[:-1] for e in hiddens]
                if any(isinstance(h, torch.Tensor) and h.device.type == 'cuda' for h in hiddens):
                    hiddens = [h.cpu() for h in hiddens]

                if len(hiddens) == 0:
                    continue
                key = (mode, env_id)
                results[key] = (hiddens, new_traj)
    return results


def run_shifted_visualization(run_info, _save_dir, arguments, _format='pdf', options={}):
    """
    Standalone alternative to run_visualization: only runs run_firing_rate_shifted.
    Does NOT re-run the full pipeline. Call this directly when you only need the
    aligned/shifted firing-rate plots.
    """
    hidden_name       = arguments.get('hidden', 'hidden')
    suc_only          = arguments.get('suc_only', True)
    uni_only          = arguments.get('uni_only', True)
    merge_seen_unseen = arguments.get('merge_seen_unseen', True)

    all_hiddens = extract_hiddens(run_info, hidden_name=hidden_name,
                                  suc_only=suc_only, uni_only=uni_only,
                                  merge_seen_unseen=merge_seen_unseen, visual_only=True)
    for (mode, env_id), (hiddens, trajs) in all_hiddens.items():
        _name     = f'{mode}__{env_id}_{hidden_name}'
        save_dir  = os.path.join(_save_dir, _name)
        os.makedirs(save_dir, exist_ok=True)
        run_firing_rate_shifted(hiddens, save_dir, save_indiv=True, _format=_format,
                                options=options.get('firing_rate', {}))
#    breakpoint()


def run_stretched_visualization(run_info, _save_dir, arguments, _format='pdf', options={}):
    """
    Standalone alternative to run_visualization: only runs run_firing_rate_stretched.
    No shifting — only X (time) and Y (amplitude) scaling.
    """
    hidden_name       = arguments.get('hidden', 'hidden')
    suc_only          = arguments.get('suc_only', True)
    uni_only          = arguments.get('uni_only', True)
    merge_seen_unseen = arguments.get('merge_seen_unseen', True)

    all_hiddens = extract_hiddens(run_info, hidden_name=hidden_name,
                                  suc_only=suc_only, uni_only=uni_only,
                                  merge_seen_unseen=merge_seen_unseen, visual_only=True)
    for (mode, env_id), (hiddens, trajs) in all_hiddens.items():
        _name    = f'{mode}__{env_id}_{hidden_name}'
        save_dir = os.path.join(_save_dir, _name)
        os.makedirs(save_dir, exist_ok=True)
        run_firing_rate_stretched(hiddens, save_dir, save_indiv=True, _format=_format,
                                  options=options.get('firing_rate', {}))


