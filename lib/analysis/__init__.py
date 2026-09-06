"""Selective per-seed analysis with granular skip and force control.

This is a new orchestration layer that sits on top of the analysis
primitives in :mod:`lib.analyze_utils` (which are left unchanged). It
lets callers run *individual* analyses without touching the outputs
of the others.

Public API:

    from lib.analysis import ANALYSES, run_seed, run_dir

    # regenerate only periodicity for one seed's run_info dump
    run_seed(run_info, save_dir='out/', target='hidden',
             analyses=['autocorr'], force={'autocorr'},
             arguments=arguments_dict)

    # walk a whole training directory, running two analyses per seed
    run_dir(dirname='decoder_dir_mental/REV/vHMN2_period',
            save_dir='analysis_out/',
            target='hidden',
            analyses=['autocorr', 'neurons'],
            force=set(),                    # skip if outputs exist
            constraints=[])

CLI entry point lives at :mod:`analyze` (repo root):

    python analyze.py -dirname ... -save_dir ... \\
        -analyses autocorr neurons -force autocorr

"""
from .registry import ANALYSES
from .dispatch import run_seed, run_dir

__all__ = ['ANALYSES', 'run_seed', 'run_dir']
