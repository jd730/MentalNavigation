"""Named analyses, each with a runner + a "have we already done this" predicate.

Every entry in :data:`ANALYSES` is a ``(runner, exists)`` tuple where:

* ``runner(hiddens, new_traj, save_dir, ctx)`` performs the analysis.
* ``exists(save_dir, ctx) -> bool`` returns True if the analysis has
  already been done in ``save_dir``. The dispatcher uses this to
  decide whether to skip (unless the analysis is in the caller's
  ``force`` set). Predicates use ``glob.glob`` where the output
  filenames depend on the data (e.g. ``hidden (L={target}).pdf``,
  where the target index isn't known ahead of time).

``ctx`` is a plain dict carrying the shared per-env context:

    image_interval      (int)   input_dim + interval_dim // step_size
    auto_corr_filename  (str)   NPY name for the interval-histogram cache
    lambdas             (list)  grid module periods
    _format             (str)   figure extension (pdf / png / ...)
    save_indiv          (bool)  emit per-d individual figures
    options             (dict)  the caller-provided options bundle

The runners are thin wrappers around the existing primitives in
:mod:`lib.analyze_utils` -- no numerical code is duplicated here.

If you add a new analysis, add its entry here and it becomes available
to :func:`lib.analysis.dispatch.run_seed`, the :mod:`analyze` CLI, and
callers that introspect ``ANALYSES`` for the CLI ``--analyses`` /
``--force`` choices.

The ``ANALYSES`` dict is deliberately declared in dispatch order:
autocorrelation (periodicity) first because both firing-rate variants
read its cached NPY output, then the two firing-rate variants, then
the cheaper single-neuron / regression analyses.
"""
import glob
import os
import re
import shutil

from lib.analyze_utils import autocorr as _autocorr
from lib.analyze_utils import firing_rate as _firing_rate
from lib.analyze_utils import neurons as _neurons
from lib.analyze_utils import regression as _regression


# Paper-named periodicity histogram is a copy of auto_corr/avg_of_avg.pdf,
# renamed to actionRNN_S<seed>.pdf / distanceRNN_S<seed>.pdf / RNN_S<seed>.pdf
# and placed at <sweep-top>/periodicity/<model>/. Layer mapping:
_PAPER_LAYER_LABEL = {
    # ReLU sweep (VHA-ReLU-D is double-CTRNN so has both layers)
    ('VHA-ReLU-D', 'hidden'):      'actionRNN',
    ('VHA-ReLU-D', 'base_hidden'): 'distanceRNN',
    ('VHA-ReLU',   'hidden'):      'RNN',
    # Legacy baselines (single-CTRNN vs double-CTRNN)
    ('VHA-D',      'hidden'):      'actionRNN',
    ('VHA-D',      'base_hidden'): 'distanceRNN',
    ('VHA',        'hidden'):      'RNN',
}


def _emit_paper_periodicity(save_dir, ext):
    """Copy ``<save_dir>/auto_corr/avg_of_avg.<ext>`` to the paper-named
    per-seed histogram location, if the sweep directory layout matches. Silent
    no-op otherwise -- keeps this helper safe to call from every run.

    Expected layout::

        <sweep_top>/dynamics/<model>_..._S<seed>Ep..._<lambdas>/<epoch>/visual__<env>_<target>/

    Emits::

        <sweep_top>/periodicity/<model>/<label>_S<seed>.<ext>
    """
    src = os.path.join(save_dir, 'auto_corr', f'avg_of_avg.{ext}')
    if not os.path.isfile(src):
        return
    # save_dir is 4 levels below the sweep_top: <top>/dynamics/<run>/<epoch>/<env_target>
    parts = os.path.normpath(save_dir).split(os.sep)
    if len(parts) < 5 or parts[-4] != 'dynamics':
        return
    sweep_top = os.sep.join(parts[:-4])
    run_name = parts[-3]
    env_target = parts[-1]                       # e.g. visual__0_base_hidden
    # env_target = visual__<env_id>_<target>; split on the last '_'
    m = re.match(r'(?:visual|mental)__\d+_(.+)$', env_target)
    if not m:
        return
    target = m.group(1)
    m_seed = re.search(r'S(\d+)Ep', run_name)
    if not m_seed:
        return
    seed = int(m_seed.group(1))
    # Model = leading token before "_N<num_images>L..."
    m_model = re.match(r'([A-Za-z\-]+(?:_[A-Z]+)*)_N\d+L', run_name)
    if not m_model:
        return
    model = m_model.group(1)
    label = _PAPER_LAYER_LABEL.get((model, target))
    if label is None:
        return
    dst_dir = os.path.join(sweep_top, 'periodicity', model)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(dst_dir, f'{label}_S{seed}.{ext}'))


# ---------------------------------------------------------------------------
# autocorr (periodicity)
# ---------------------------------------------------------------------------

def _autocorr_run(hiddens, new_traj, save_dir, ctx):
    """Autocorrelation + FFT + interval-histogram; honors separate_direction.

    Also emits the paper-named per-seed periodicity histogram (a rename of
    ``auto_corr/avg_of_avg.pdf``) at
    ``<sweep_top>/periodicity/<model>/<label>_S<seed>.pdf`` if the sweep
    directory layout matches.
    """
    options = ctx['options'].get('autocorr', {})
    if options.get('separate_direction', False):
        direction = [traj[-1] > traj[0] for traj in new_traj]
        left = [h for h, d in zip(hiddens, direction) if not d]
        right = [h for h, d in zip(hiddens, direction) if d]
        opt = dict(options)
        opt['direction'] = 'left'
        if left:
            _autocorr.run_autocorr(left, save_dir, True,
                                   ctx['image_interval'], ctx['auto_corr_filename'],
                                   ctx['lambdas'], _format=ctx['_format'], options=opt)
        opt = dict(options)
        opt['direction'] = 'right'
        if right:
            _autocorr.run_autocorr(right, save_dir, True,
                                   ctx['image_interval'], ctx['auto_corr_filename'],
                                   ctx['lambdas'], _format=ctx['_format'], options=opt)
    else:
        _autocorr.run_autocorr(hiddens, save_dir, True,
                               ctx['image_interval'], ctx['auto_corr_filename'],
                               ctx['lambdas'], _format=ctx['_format'], options=options)
    _emit_paper_periodicity(save_dir, ctx['_format'])


def _autocorr_exists(save_dir, ctx):
    """``auto_corr/avg_of_avg.{ext}`` is the per-seed periodicity histogram
    that downstream aggregators + the paper's periodicity/ figures reuse.
    (auto_corr_all.pdf and period_*.pdf are no longer produced.)"""
    ext = ctx['_format']
    return os.path.exists(os.path.join(save_dir, 'auto_corr', f'avg_of_avg.{ext}'))


# ---------------------------------------------------------------------------
# firing_rate  (aggregate + per-d indiv)
# ---------------------------------------------------------------------------

def _firing_rate_run(hiddens, new_traj, save_dir, ctx):
    """Aggregate firing-rate figure + is_onset variant; honors separate_direction."""
    options = ctx['options'].get('firing_rate', {})
    save_indiv = ctx['save_indiv']
    _format = ctx['_format']
    if options.get('separate_direction', False):
        direction = [traj[-1] > traj[0] for traj in new_traj]
        left = [h for h, d in zip(hiddens, direction) if not d]
        right = [h for h, d in zip(hiddens, direction) if d]
        opt = dict(options)
        opt['direction'] = 'left'
        if left:
            _firing_rate.run_firing_rate(left, save_dir, save_indiv=save_indiv, _format=_format, options=opt)
            _firing_rate.run_firing_rate(left, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=opt)
        opt = dict(options)
        opt['direction'] = 'right'
        if right:
            _firing_rate.run_firing_rate(right, save_dir, save_indiv=save_indiv, _format=_format, options=opt)
            _firing_rate.run_firing_rate(right, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=opt)
    else:
        _firing_rate.run_firing_rate(hiddens, save_dir, is_onset=True, save_indiv=save_indiv, _format=_format, options=options)
        _firing_rate.run_firing_rate(hiddens, save_dir, save_indiv=save_indiv, _format=_format, options=options)


def _firing_rate_exists(save_dir, ctx):
    """Both the aggregate and the onset-aligned figure must exist."""
    ext = ctx['_format']
    return (os.path.exists(os.path.join(save_dir, f'firing_rate.{ext}')) and
            os.path.exists(os.path.join(save_dir, f'firing_rate_onset.{ext}')))


# ---------------------------------------------------------------------------
# firing_rate_each_trial
# ---------------------------------------------------------------------------

def _firing_rate_each_trial_run(hiddens, new_traj, save_dir, ctx):
    """Per-trial firing-rate figures split by target index."""
    _firing_rate.run_firing_rate_each_trial(
        hiddens, new_traj, save_dir, _format=ctx['_format'],
    )


def _firing_rate_each_trial_exists(save_dir, ctx):
    """Any ``each_trial_firing_rate_*.{ext}`` counts -- target IDs are data-dependent."""
    ext = ctx['_format']
    return bool(glob.glob(os.path.join(save_dir, f'each_trial_firing_rate_*.{ext}')))


# ---------------------------------------------------------------------------
# regression (learned-vs-target distance scatter)
# ---------------------------------------------------------------------------

def _regression_run(hiddens, new_traj, save_dir, ctx):
    _regression.run_regression(save_dir, ctx['epochs'], _format=ctx['_format'])


def _regression_exists(save_dir, ctx):
    return os.path.exists(os.path.join(save_dir, 'regression', f'scatter.{ctx["_format"]}'))


# ---------------------------------------------------------------------------
# neurons (single-neuron per-index firing figures + ranking)
# ---------------------------------------------------------------------------

def _neurons_run(hiddens, new_traj, save_dir, ctx):
    _neurons.analyze_neurons(hiddens, save_dir, _format=ctx['_format'])


def _neurons_exists(save_dir, ctx):
    """analyze_neurons writes ``hidden (L={target}).{ext}`` per ranked neuron.

    Target IDs are data-dependent (not always 0), so use glob over the
    file name pattern rather than assuming a specific target index.
    """
    ext = ctx['_format']
    return bool(glob.glob(os.path.join(save_dir, f'hidden (L=*).{ext}')))


ANALYSES = {
    'autocorr':               (_autocorr_run,               _autocorr_exists),
    'firing_rate':            (_firing_rate_run,            _firing_rate_exists),
    'firing_rate_each_trial': (_firing_rate_each_trial_run, _firing_rate_each_trial_exists),
    'regression':             (_regression_run,             _regression_exists),
    'neurons':                (_neurons_run,                _neurons_exists),
}

__all__ = ['ANALYSES']
