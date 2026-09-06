"""Seed / env / mode dispatcher with granular skip and force control.

Mirrors the (env, mode, observe) iteration of
:func:`lib.analyze_utils.orchestration.run_visualization` but replaces
the hard-bundled ``_run_corr_per_env`` call with a loop over selected
analyses; each analysis skips itself if its output files already exist
(unless the caller pinned it in the ``force`` set).

Two levels of entry point:

* :func:`run_seed` -- given an already-loaded ``run_info`` dict and a
  destination directory, iterate over its (env, mode, observe) tuples
  and run the selected analyses on each.

* :func:`run_dir` -- walk a training save-directory, loading each
  seed's latest ``run_info_*.pth`` and calling :func:`run_seed` on it.

Neither entry point touches the legacy
:mod:`lib.analyze_utils.orchestration.run_visualization` -- callers
that want the historical bundled behaviour keep using that.
"""
import glob
import json
import os

import numpy as np
import torch

from .registry import ANALYSES


def _apply_hidden_reshape(hiddens, hidden_name):
    """Match the reshape logic in run_visualization for the non-hidden cases.

    Skips empty-list entries: they arise when a single-CTRNN model (VHA,
    VHA-ReLU, RNN) is queried for a double-CTRNN-only field like
    ``base_hidden`` -- the training loop stashes ``[]`` there per trial.
    """
    hiddens = [e for e in hiddens if not (isinstance(e, list) and len(e) == 0)]
    if hidden_name == 'imgs':
        return [e[:, :, :1] for e in hiddens]
    if hidden_name == 'ps' or hidden_name == 'images':
        return [e.transpose(0, 2, 1).reshape(len(e), 1, -1) for e in hiddens]
    if hidden_name != 'hidden':
        return [e.reshape(len(e), 1, -1) for e in hiddens]
    return hiddens


def _collect_per_env(run_info, arguments, visual_only, hidden_name):
    """Iterate (env_id, mode, observe), yielding filtered ``(hiddens, new_traj, name)``.

    Reproduces the filter chain in run_visualization: suc_only,
    uni_only, optional seen+unseen merge, tensor->cpu conversion, and
    the hidden_name-specific reshape.
    """
    suc_only          = arguments.get('suc_only', True)
    uni_only          = arguments.get('uni_only', True)
    merge_seen_unseen = arguments.get('merge_seen_unseen', True)

    for env_id in range(1000):
        if f'visual/seen/{env_id}/{hidden_name}' not in run_info:
            break
        modes = ['visual'] if visual_only else ['visual', 'mental']
        for mode in modes:
            for observe in ['seen', 'unseen']:
                prefix = f'{mode}/{observe}/{env_id}'
                hidden        = list(run_info[f'{prefix}/{hidden_name}'])
                actions       = list(run_info[f'{prefix}/actions'])
                unidirection  = list(run_info[f'{prefix}/unidirection'])
                trajs         = list(run_info[f'{prefix}/trajs'])
                success       = list(run_info[f'{prefix}/suc'])

                if merge_seen_unseen:
                    if observe == 'unseen':
                        continue
                    _prefix = f'{mode}/unseen/{env_id}'
                    hidden       += list(run_info[f'{_prefix}/{hidden_name}'])
                    actions      += list(run_info[f'{_prefix}/actions'])
                    unidirection += list(run_info[f'{_prefix}/unidirection'])
                    trajs        += list(run_info[f'{_prefix}/trajs'])
                    success      += list(run_info[f'{_prefix}/suc'])

                hiddens, new_traj = [], []
                for hid, suc, uni, traj in zip(hidden, success, unidirection, trajs):
                    if (not suc_only or suc) and (not uni_only or uni):
                        hiddens.append(hid)
                        new_traj.append(traj)

                hiddens = _apply_hidden_reshape(hiddens, hidden_name)
                if len(hiddens) == 0:
                    continue
                if any(isinstance(h, torch.Tensor) and h.device.type == 'cuda' for h in hiddens):
                    hiddens = [h.cpu() for h in hiddens]

                name = prefix.replace('/', '_') + '_' + hidden_name
                if merge_seen_unseen:
                    name = name.replace('unseen', '').replace('seen', '')
                yield env_id, mode, observe, hiddens, new_traj, name


def _should_skip(name, save_dir, ctx, force):
    """True if the analysis has already been done and isn't in ``force``."""
    if name in force:
        return False
    _, exists = ANALYSES[name]
    return exists(save_dir, ctx)


def run_seed(run_info, save_dir, target='hidden', analyses=None, force=None,
             arguments=None, visual_only=False, save_indiv=False,
             _format='pdf', options=None, verbose=True):
    """Run the selected analyses for one seed's ``run_info`` dump.

    Iterates over the (env_id, mode, observe) tuples in ``run_info``;
    for each one it prepares a per-env ``save_dir`` and calls each
    analysis in ``analyses`` in registry order, skipping analyses
    whose outputs already exist unless the analysis name is in
    ``force``.

    Args:
        run_info: The ``run_info_*.pth`` dict loaded via ``torch.load``.
        save_dir: Root output dir. Per-env subdirs are created beneath.
        target: Which hidden field to read (``hidden``, ``base_hidden``,
            ``gs``, ...).
        analyses: List of analysis names from :data:`ANALYSES`;
            defaults to all registered analyses (in registry order).
        force: Iterable of analysis names to force-regenerate even if
            their outputs exist. Pass an empty iterable / ``None`` to
            skip everything that's already been generated.
        arguments: Caller-side options dict; carries the training
            arguments (lambdas, epochs, input_dim, interval_dim,
            step_size, seed, suc_only, uni_only, merge_seen_unseen).
            Same shape as the legacy ``arguments`` param.
        visual_only: If True, skip the ``mental`` mode.
        save_indiv: If True, per-d individual firing-rate figures are
            emitted.
        _format: Figure extension.
        options: Nested per-analysis options bundle; supported keys are
            ``autocorr`` and ``firing_rate``.
        verbose: If True, log per-env / per-analysis run/skip decisions.

    Returns:
        None. Writes figures + caches into ``save_dir``.
    """
    if arguments is None:
        arguments = {}
    if options is None:
        options = {}
    if analyses is None:
        analyses = list(ANALYSES.keys())
    force = set(force) if force else set()

    unknown = [a for a in analyses if a not in ANALYSES]
    if unknown:
        raise KeyError(f'unknown analyses: {unknown!r} (registered: {list(ANALYSES.keys())!r})')
    unknown = [a for a in force if a not in ANALYSES]
    if unknown:
        raise KeyError(f'unknown force entries: {unknown!r}')

    lambdas      = arguments.get('lambdas', [11, 12, 13])
    epochs       = arguments.get('epochs', 5000)
    input_dim    = arguments.get('input_dim', 384)
    interval_dim = arguments.get('interval_dim', 384)
    step_size    = arguments.get('step_size', 64)
    image_interval = (input_dim + interval_dim) // step_size
    auto_corr_filename = f'{image_interval}_{arguments.get("seed", 0)}_{input_dim}_{interval_dim}_{step_size}.npy'

    ctx = {
        'image_interval':     image_interval,
        'auto_corr_filename': auto_corr_filename,
        'lambdas':            lambdas,
        'epochs':             epochs,
        '_format':            _format,
        'save_indiv':         save_indiv,
        'options':            options,
    }

    for env_id, mode, observe, hiddens, new_traj, env_name in _collect_per_env(
        run_info, arguments, visual_only, hidden_name=target,
    ):
        env_save_dir = os.path.join(save_dir, env_name)
        os.makedirs(env_save_dir, exist_ok=True)
        for name in analyses:
            if _should_skip(name, env_save_dir, ctx, force):
                if verbose:
                    print(f'[skip] {name:<24s} {env_save_dir}')
                continue
            if verbose:
                print(f'[run ] {name:<24s} {env_save_dir}')
            runner, _ = ANALYSES[name]
            runner(hiddens, new_traj, env_save_dir, ctx)


def _iter_seed_dirs(dirname, constraints, epoch=None):
    """Yield ``(run_name, run_info_path)`` pairs matching ``constraints``.

    ``constraints`` is a list of substrings that must all appear in the
    per-seed directory name. ``epoch`` picks a specific ``run_info_<epoch>.pth``
    checkpoint (e.g. ``'001999'`` for the end-of-env0 dump); default None
    keeps the legacy behaviour of picking the last checkpoint.
    """
    for run_name in sorted(os.listdir(dirname)):
        if not all(c in run_name for c in (constraints or [])):
            continue
        run_dir_ = os.path.join(dirname, run_name)
        if not os.path.isdir(run_dir_):
            continue
        run_infos = sorted(glob.glob(os.path.join(run_dir_, 'run_info_*.pth')))
        if not run_infos:
            continue
        if epoch is not None:
            candidate = os.path.join(run_dir_, f'run_info_{epoch}.pth')
            if not os.path.exists(candidate):
                continue
            yield run_name, candidate
            continue
        # last checkpoint = last epoch snapshot; matches the [:1] slice
        # in the legacy collect.run_analysis loop (which took the first).
        yield run_name, run_infos[-1]


def run_dir(dirname, save_dir, target='hidden', analyses=None, force=None,
            constraints=None, seeds=None, visual_only=False, save_indiv=False,
            _format='pdf', options=None, use_cuda=True, verbose=True,
            epoch=None):
    """Walk a training directory, dispatching :func:`run_seed` per seed.

    For each per-seed subdirectory of ``dirname`` that matches
    ``constraints`` (substring filter) and whose ``arguments.json:seed``
    is in ``seeds`` (if given), loads the last ``run_info_*.pth`` and
    calls :func:`run_seed` with the requested selection.

    Args:
        dirname: Directory containing per-seed subdirs (each with an
            ``arguments.json`` and one or more ``run_info_*.pth``).
        save_dir: Root output dir. Analyses land under
            ``save_dir/<run_name>/<epoch>/<env>/``.
        target: Which hidden field to analyze.
        analyses, force, visual_only, save_indiv, _format, options:
            Forwarded to :func:`run_seed`.
        constraints: List of substrings; only run names containing all
            of them are analyzed. Empty list matches everything.
        seeds: Optional iterable of ints. If given, seeds not in the
            set are skipped based on ``arguments.json:seed``.
        use_cuda: If True and CUDA is available, load run_info onto GPU.
        verbose: Passed through.
    """
    seeds = set(seeds) if seeds is not None else None
    for run_name, run_info_path in _iter_seed_dirs(dirname, constraints, epoch=epoch):
        argument_path = os.path.join(dirname, run_name, 'arguments.json')
        if os.path.exists(argument_path):
            with open(argument_path) as f:
                arguments = json.load(f)
        else:
            arguments = {}
        if seeds is not None and arguments.get('seed') not in seeds:
            continue

        epoch = os.path.basename(run_info_path).split('run_info_')[1].split('.')[0]
        seed_save_dir = os.path.join(save_dir, run_name, epoch)
        if verbose:
            print(f'\n=== {run_name} (epoch {epoch}) ===')

        if use_cuda and torch.cuda.is_available():
            run_info = torch.load(run_info_path, weights_only=False)
        else:
            run_info = torch.load(run_info_path, map_location='cpu',
                                  weights_only=False)

        run_seed(run_info, seed_save_dir, target=target,
                 analyses=analyses, force=force, arguments=arguments,
                 visual_only=visual_only, save_indiv=save_indiv,
                 _format=_format, options=options, verbose=verbose)


__all__ = ['run_seed', 'run_dir']
