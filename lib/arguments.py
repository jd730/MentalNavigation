"""Command-line interface for train.py / collect / analysis scripts.

Defines the shared argparse Namespace via :func:`get_args` and the
canonical on-disk run name via :func:`get_name`.

The CLI is intentionally minimal: every flag still listed below has at
least one consumer in the current codebase. Architectural sub-flags
that used to live here (``-cat_action``, ``-decoder``, ``-gcpc``, ...)
are now derived from ``-model`` inside ``train.py``.
"""
import argparse

import numpy as np


def get_name(args):
    """Build the canonical on-disk run name.

    Format: ``<model>_N..L..I..Step..A..lr..F..S..Ep..[_tag...]``.

    The fixed-position fields (``N`` num_images, ``L`` input_dim,
    ``I`` interval_dim, ``Step`` step_size, ``A`` alpha, ``lr``,
    ``F`` feature_dim, ``S`` seed, ``Ep`` epochs) are always present.
    Optional tags are appended only when the relevant flag is set:

    * ``_scalingfactor``   -- ``-scaling_factor``
    * ``_Res<n>``          -- ``-resolution > 1``
    * ``_gNs<>Np<>_<lambdas>`` -- VHA / VHA-D baselines (grid cells)

    Architectural choices (decoder head, cat_action input bump,
    encoder depth) are encoded in ``args.model`` and don't need
    separate tags.

    Args:
        args: argparse.Namespace returned by :func:`get_args`.

    Returns:
        str: The run name used as the on-disk ``save_dir`` suffix.
    """
    name = (f'{args.model}'
            f'_N{args.num_images}'
            f'L{args.input_dim}'
            f'I{args.interval_dim}'
            f'Step{args.step_size}'
            f'A{args.alpha}'
            f'lr{args.lr}'
            f'F{args.feature_dim}'
            f'S{args.seed}'
            f'Ep{args.epochs}')

    if args.scaling_factor:
        name += '_SF'
    if args.resolution > 1:
        name += f'_Res{args.resolution}'
    # Non-default dist-target variants (default = scale=5, abs on).
    ds = float(getattr(args, 'dist_scale', 5))
    if ds != 5.0:
        # Integer scales get a bare label (e.g. Scale10), fractional keep the dot.
        name += f'_Scale{int(ds)}' if ds == int(ds) else f'_Scale{ds}'
    if getattr(args, 'no_abs', False):
        name += '_Signed'
    if getattr(args, 'no_rev', False):
        name += '_NoRev'

    if args.grid_cells:
        # Ns = sensory dim per landmark; Np capped at the total number
        # of distinct grid codes (prod of the module periods).
        Ns = args.input_dim
        Np = min(args.np, int(np.prod(args.lambdas)))
        name += f'_gNs{Ns}Np{Np}'
        for lamb in args.lambdas:
            name += f'_{lamb}'

    return name


def get_args():
    """Build and return the project-wide argparse Namespace.

    Defines every flag consumed by ``train.py`` / ``analyze.py`` /
    ``rdm_analysis.py``. The groups below mirror the on-disk run name
    layout: dataset, grid cells, architecture, training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description='Mental-navigation training CLI.')

    # ----- Logging / I/O -----
    parser.add_argument('-debug', action='store_true',
                        help='Disable wandb logging (use for local smoke tests).')
    parser.add_argument('-save_dir', type=str, default='results',
                        help='Root directory under which per-run checkpoints '
                             'and run_info_*.pth snapshots are written.')

    # ----- Dataset / environment -----
    parser.add_argument('-num_images', type=int, default=6,
                        help='Number of landmark images per environment (N in the run name).')
    parser.add_argument('-step_size', type=int, default=64,
                        help='Pixels travelled per internal-velocity step. Must satisfy '
                             'lambda / resolution == (input_dim + interval_dim) / step_size '
                             'for VHA / VHA-D (grid cells).')
    parser.add_argument('-input_dim', type=int, default=384,
                        help='Dimensionality of each landmark sensory vector (one half of the '
                             'model input; the other half is the target).')
    parser.add_argument('-trial_length', type=int, default=100,
                        help='Maximum number of action steps per trial before the env is reset.')
    parser.add_argument('-num_envs', type=int, default=3,
                        help='Number of parallel environments used during adaptation/eval.')
    parser.add_argument('-interval_dim', type=int, default=384,
                        help='Empty padding interleaved between landmark slots. Used in the '
                             'grid-wrap consistency check '
                             '`lambda / resolution == (input_dim + interval_dim) / step_size`.')
    parser.add_argument('-eval_only', action='store_true',
                        help='Skip training and only run evaluation on existing checkpoints '
                             'under -save_dir.')

    # ----- Grid cells (auto-on for VHA/VHA-D; derived in train.py) -----
    parser.add_argument('-lambdas', nargs='+', type=int, default=[11, 12, 13],
                        help='Grid-module periods (one per module). The total grid-code '
                             'cardinality is prod(lambdas) ** dimension.')
    parser.add_argument('-np', type=int, default=400,
                        help='Number of place cells. Capped internally at prod(lambdas).')
    parser.add_argument('-resolution', type=int, default=1,
                        help='Internal-velocity multiplier; >1 makes the grid step in chunks '
                             "of `resolution` per environment step and adds an _Res<n> tag to "
                             'the run name.')

    # ----- Architecture -----
    parser.add_argument('-model', type=str, required=True,
                        help='Baseline name (see lib.models.MODELS). Architectural sub-flags '
                             '(encoder depth, decoder head, cat_action input bump) are derived '
                             'from this -- callers only need to set -model.')
    parser.add_argument('-feature_dim', type=int, default=256,
                        help='Hidden dimension of the (CT)RNN core.')
    parser.add_argument('-alpha', type=float, default=0.9,
                        help='CTRNN integration constant: 1 = fully discrete update, '
                             '<1 = leaky integrator with effective time constant 1/alpha.')
    parser.add_argument('-scaling_factor', action='store_true',
                        help='Enable the internal_corr_suc_3 scaling-factor head (the only '
                             'velocity-supervision mode we still use).')

    # ----- Distance-prediction target variants (paper canonical: rev+abs, scale=5) -----
    parser.add_argument('-dist_scale', type=float, default=5.0,
                        help="Weight for the rev'd distance target: "
                             "gt = |initial - current| * (dist_scale / |initial|). "
                             "Paper canonical is 5 ('norm=5'); pass another value "
                             "to test the sensitivity of the dist RNN to this scale.")
    parser.add_argument('-no_abs', action='store_true',
                        help="Turn OFF the final abs() on the rev'd distance target. "
                             "Paper canonical is on (positive scalar 0..scale). "
                             "With -no_abs, target keeps its sign so left- and right-going "
                             "trials get opposite target trajectories.")
    parser.add_argument('-no_rev', action='store_true',
                        help="Turn OFF the reverse step. Paper canonical does "
                             "gt = |initial - current|*ratio (0 -> scale over the trial). "
                             "With -no_rev, gt = |current|*ratio (scale -> 0), i.e. "
                             "'remaining distance to target' rather than 'progress from start'.")

    # ----- Training -----
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('-epochs', type=int, default=2000,
                        help='Total training epochs.')
    parser.add_argument('-optimizer', type=str, default='adam',
                        help="'adam' (paper default) or 'sgd' (momentum 0.9, hardcoded).")
    parser.add_argument('-seed', type=int, default=13,
                        help='Random seed for python / numpy / torch RNGs and env layout.')
    parser.add_argument('-log_interval', type=int, default=100,
                        help='Checkpoint + run_info snapshot every N epochs.')
    parser.add_argument('-project_name', type=str, default='mental_navigation_publish',
                        help='wandb project name (ignored when -debug is set).')

    return parser.parse_args()
