"""Circular Gaussian smoothing of the ``gs`` field (grid-cell one-hots) in
per-seed .mat exports. Matches the ``smooth_gs_circular.py`` referenced by
draw_xcorr.m:184 (fallback path we didn't have before, so the current
mat_files_smoothed/ symlink was serving raw one-hots).

For each module segment [0:11], [11:23], [23:36] (periods 11, 12, 13)
apply scipy.ndimage.gaussian_filter1d along the module dim with
``mode='wrap'`` -- one-hot positions blur into the neighbours (circularly)
so the xcov peaks land at consistent phases across modules.

Other .mat fields (hidden, base_hidden, trajs, suc, ps, ...) are passed
through untouched. NaN padding on time bins is preserved (skip smoothing
over NaN rows).

Usage:
    python scripts/smooth_gs_circular.py \\
        --input-file <raw.mat> --output-dir <out_dir> --sigma 1.0

Batch usage:
    python scripts/smooth_gs_circular.py \\
        --input-dir results/ReLU/mat_exports \\
        --output-dir results/ReLU/mat_exports_smoothed \\
        --sigma 1.0
"""
import argparse
import glob
import os

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d

# Grid-module boundaries in the 36-wide one-hot gs. Matches _export_run_info_to_mat
# in lib/analyze_utils/collect.py:1153-1162 which one-hots each module separately
# into 11 + 12 + 13 columns.
MODULE_BOUNDS = [(0, 11), (11, 23), (23, 36)]


def smooth_gs(gs, sigma):
    """Circular gaussian smoothing per module.

    Args:
        gs: (n_trials, T, 36) float array with NaN in padded time bins.
        sigma: gaussian sigma (in module bins).

    Returns:
        (n_trials, T, 36) same shape/dtype with smoothed values in the
        non-NaN rows; NaN rows preserved.
    """
    out = gs.astype(np.float64).copy()
    n_trials, T, _ = out.shape
    for i in range(n_trials):
        # Which time rows are valid (any non-NaN)? Iterate over those only.
        valid = ~np.any(np.isnan(out[i]), axis=1)
        if not valid.any():
            continue
        for t in np.where(valid)[0]:
            for lo, hi in MODULE_BOUNDS:
                out[i, t, lo:hi] = gaussian_filter1d(
                    out[i, t, lo:hi], sigma=sigma, mode='wrap',
                )
    return out.astype(gs.dtype)


def _smooth_one(input_path, output_dir, sigma):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_path))
    m = scipy.io.loadmat(input_path)
    if 'gs' not in m:
        raise KeyError(f'{input_path} has no gs field')
    m['gs'] = smooth_gs(m['gs'], sigma)
    m.pop('__header__', None)
    m.pop('__version__', None)
    m.pop('__globals__', None)
    scipy.io.savemat(out_path, m)
    return out_path


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input-file')
    g.add_argument('--input-dir')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sigma', type=float, default=1.0)
    args = p.parse_args()

    if args.input_file:
        out = _smooth_one(args.input_file, args.output_dir, args.sigma)
        print(f'wrote {out}')
        return

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.mat')))
    print(f'{len(files)} mats to smooth (sigma={args.sigma})')
    for i, f in enumerate(files, 1):
        try:
            _smooth_one(f, args.output_dir, args.sigma)
        except Exception as e:  # noqa: BLE001
            print(f'  fail {os.path.basename(f)}: {e}')
        if i % 10 == 0:
            print(f'  {i}/{len(files)}')
    print('done')


if __name__ == '__main__':
    main()
