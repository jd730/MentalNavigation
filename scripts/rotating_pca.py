"""Standalone rotating-PCA generator for VHA-ReLU-D dist-variant sweeps.

Deliberately kept out of analyze_relu.py's orchestration: this iterates
raw .mat exports (or run_info dirs) directly, so it can be pointed at any
sweep folder without needing the full analyze_relu.py pipeline (log.csv
aggregation, RDM, etc.) that assumes the canonical decoder_dir_mental/ReLU
layout.

Usage:
    # 1. Export .mat files for a variant's runs (uses collect helper).
    python scripts/rotating_pca.py \\
        --run_dir decoder_dir_mental/ReLU_dist_variants \\
        --mat_out results/dist_variants/mat_exports \\
        --pca_out results/dist_variants/pca_rotation \\
        --n_workers 8

Produces one rotation-grid PDF per (model, layer, seed, variant) at
    <pca_out>/<run_name>/{actionRNN, distanceRNN}_S<seed>_rotation.pdf

The run_name suffix (_Signed, _Scale10, ...) is preserved so variants are
disambiguated on disk without needing a lookup table.
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool


def _find_mat_files(run_dir):
    """Return list of every seed-run subdir under run_dir."""
    return sorted(d for d in glob.glob(os.path.join(run_dir, '*'))
                  if os.path.isdir(d) and 'run_info_001999.pth' in os.listdir(d))


def _export_mats(run_dir, mat_out):
    """Convert every run_info_001999.pth under run_dir to a .mat under mat_out.

    Skips runs whose .mat is already exported.
    """
    from lib.analyze_utils.collect import export_run_info_to_mat
    os.makedirs(mat_out, exist_ok=True)
    for sub in _find_mat_files(run_dir):
        run_name = os.path.basename(sub)
        if os.path.exists(os.path.join(mat_out, run_name + '.mat')):
            continue
        # export_run_info_to_mat takes (dirname, constraint list, save_dir);
        # constrain to the single run_name for isolation.
        print(f'exporting {run_name}')
        try:
            export_run_info_to_mat(run_dir, [run_name], mat_out)
        except Exception as e:  # noqa: BLE001
            print(f'  failed: {e}')


def _worker(args):
    """Multiprocessing worker: generate one rotation PDF pair for one .mat."""
    mat_path, save_root, trim_tail = args
    from lib.analyze_utils.pca import plot_pca_rotation_grid_for_mat
    run_name = os.path.basename(mat_path).replace('.mat', '')
    # Group by full run_name (keeps variant suffixes distinct).
    save_dir = os.path.join(save_root, run_name)
    # Filename suffix disambiguates trim variants (side-by-side compare).
    suffix = f'_trim{trim_tail}' if trim_tail else ''
    try:
        plot_pca_rotation_grid_for_mat(mat_path, save_dir, trim_tail=trim_tail,
                                        filename_suffix=suffix)
        return f'ok  {run_name}{suffix}'
    except Exception as e:  # noqa: BLE001
        return f'FAIL {run_name}{suffix}: {e}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', required=True,
                   help='Directory of trained runs (each subdir has run_info_001999.pth).')
    p.add_argument('--mat_out', required=True,
                   help='Where per-run .mat exports should land.')
    p.add_argument('--pca_out', required=True,
                   help='Where rotation-grid PDFs should land.')
    p.add_argument('--n_workers', type=int, default=8,
                   help='Parallel PCA jobs (multiprocessing.Pool). Default 8.')
    p.add_argument('--only_seeds', nargs='+', type=int, default=None,
                   help='Optional integer allow-list. Filters mat files by seed.')
    p.add_argument('--model_filter', type=str, default=None,
                   help="Restrict to mats whose basename starts with '<model>_' "
                        "(e.g. 'VHA-ReLU-D'). Also drops VHA-ReLU-D when "
                        "'VHA-ReLU' is requested, since the prefix matches both.")
    p.add_argument('--trim_tail', type=int, default=0,
                   help='Drop the last N steps of every trajectory before PCA. '
                        'Filenames are suffixed _trim<N> so multiple values '
                        "coexist under the same seed directory.")
    args = p.parse_args()

    # 1. Export .mat files (fast if already done -- existence check).
    _export_mats(args.run_dir, args.mat_out)

    # 2. Collect .mat files to process.
    mats = sorted(glob.glob(os.path.join(args.mat_out, '*.mat')))
    if args.model_filter:
        prefix = args.model_filter + '_'
        mats = [m for m in mats if os.path.basename(m).startswith(prefix)]
        if args.model_filter == 'VHA-ReLU':
            mats = [m for m in mats if not os.path.basename(m).startswith('VHA-ReLU-D_')]
    if args.only_seeds:
        allow = set(args.only_seeds)
        def _seed_ok(m):
            mm = re.search(r'S(\d+)Ep', os.path.basename(m))
            return mm and int(mm.group(1)) in allow
        mats = [m for m in mats if _seed_ok(m)]
    if not mats:
        print('no .mat files to process; exiting.')
        return
    print(f'{len(mats)} .mat files -> pca_out ({args.n_workers} workers, trim_tail={args.trim_tail})')

    jobs = [(m, args.pca_out, args.trim_tail) for m in mats]
    with Pool(processes=args.n_workers) as pool:
        for msg in pool.imap_unordered(_worker, jobs):
            print(msg, flush=True)
    print('all done')


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
