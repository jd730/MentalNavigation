# Structural generalization and continual learning enabled by factorized entorhinal-hippocampal memory and entorhinal-parietal action circuits

Code accompanying **Hwang, Neupane, Jazayeri & Fiete (2026)**, bioRxiv:
<https://www.biorxiv.org/content/10.64898/2026.08.25.747129v1>.

Trains a recurrent network on a mental navigation task and compares its
internal dynamics against monkey neural recordings.

## Setup

The recommended setup uses conda; a plain-pip alternative works too.

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate mental
```

**Pip:**

```bash
python -m venv mental-env
source mental-env/bin/activate
pip install -r requirements.txt
```

**Hardware / driver notes for exact bit-identical reproduction:**

* The paper-config runs were checked on **NVIDIA A100 80 GB PCIe** with
  driver ≥ 550 and CUDA runtime 12.4 (bundled with the `torch` wheel).
* We rely on `cudnn.deterministic=True` + `cudnn.benchmark=False` +
  fixed `PYTHONHASHSEED` (all set in `lib/utils.py:set_seed`).
* Byte-identical checkpoints were verified across two independent
  32-minute runs of `python train.py -model VHA-D -seed 43` on the
  same A100 (SHA-256 match on `log.csv`, all `*.pth` checkpoints, and
  all `run_info_*.pth` dumps).
* Different GPU architecture / driver / cuDNN version will give
  numerically close but not byte-identical results (a PyTorch reality,
  not a code issue).

## Quick start

Train the canonical paper model (VHA-D, 2000 epochs per environment
over 3 environments = 6000 total epochs, one of the 50 seeds used
for the paper sweep):

```bash
python train.py -model VHA-D -seed 43 -debug
```

`-debug` disables wandb logging; drop it if you want the run logged
to your own wandb project (default project name in `lib/arguments.py`).

That single command trains the full paper config. Every architectural
knob, every default (optimizer, learning rate, feature_dim,
trial_length, input_dim, interval_dim, step_size, num_envs, alpha,
grid-cell wrapper, ...) comes from `lib/arguments.py`. Reviewers can
run the same command and expect the same result (see setup notes
above).

Runs are written to `results/<run_name>/` by default; override with
`-save_dir`.

## Batch training on SLURM

`scripts/` holds unified SLURM launchers. Each script picks its
task-layout via env-var `MODE` and reads other knobs (STEP_SIZE,
RESOLUTION, seeds, ...) from env too; pass partition / wall /
--array at `sbatch` time. Preset command examples live in the
docstring at the top of each script.

| script                          | topic                                                            |
|---------------------------------|------------------------------------------------------------------|
| `train_baseline.sh`             | 9-model baseline comparison sweep (includes canonical VHA-D)     |
| `train_rnn_baseline.sh`         | RNN / RNN-D single-CTRNN vs double-CTRNN baselines               |
| `period_sweep.sh`               | VHA-D grid-period sweep (6 configs × 50 seeds)                   |
| `scaling_factor.sh`             | VHA-D scaling-factor sweep (2 resolution variants × 50 seeds)    |
| `rdm_run.sh`                    | RDM analysis batch (one task per seed 0–49)                      |

Analysis entry points:

| script                          | purpose                                                          |
|---------------------------------|------------------------------------------------------------------|
| `analyze.py`                    | general driver; auto-detects `KNOWN_MODELS` present under `-dirname` |
| `scripts/single_neuron.py`      | per-seed single-neuron dumps (σ=2, ignore_last=True canonical)   |
| `scripts/rotating_pca.py`       | trajectory-PCA rotation grids per seed                           |
| `scripts/matlab_xcorr_all_seeds.m` | MATLAB batch for pairwise xcov heatmaps                       |
| `scripts/smooth_gs_circular.py` | Gaussian smoothing on grid-cell exports for xcorr               |

## Repository layout

```
.
├── train.py                     # trains one model / one seed (CLI)
├── analyze.py                   # general analysis driver (auto-detect models)
├── rdm_analysis.py              # RDM: monkey vs model similarity (per-seed)
├── rdm_aggregate.py             # RDM: cross-seed aggregation of pair scores
├── requirements.txt             # pip-installable dependency pins
├── environment.yml              # conda env for full reproducibility
├── scripts/                     # SLURM launchers + analysis helpers
│   ├── train_baseline.sh        # 9-model x 50-seed baseline sweep (includes VHA-D)
│   ├── train_rnn_baseline.sh    # RNN / RNN-D baselines
│   ├── period_sweep.sh          # VHA-D grid-period sweep
│   ├── scaling_factor.sh        # VHA-D scaling-factor sweep
│   ├── rdm_run.sh               # RDM per-seed batch
│   ├── extract_neurons.sh       # per-figure single-neuron extraction
│   ├── single_neuron.py         # per-seed single-neuron dumps
│   ├── rotating_pca.py          # per-seed PCA rotation grids
│   ├── matlab_xcorr_all_seeds.m # MATLAB xcov batch
│   └── smooth_gs_circular.py    # gaussian smoothing for xcorr
│
├── pca_utils/                   # trajectory-PCA helpers (model + monkey)
│
└── lib/
    ├── arguments.py             # argparse + canonical run-name builder
    ├── exp.py                   # training / eval loop helpers
    ├── utils.py                 # set_seed, GPU picker, misc numpy helpers
    ├── rl_utils.py              # REINFORCE (used by -scaling_factor)
    ├── gridUtils.py             # grid-cell init + one-step dynamics
    │
    ├── models/                  # per-baseline architecture files
    │   ├── _base.py             # SingleCTRNN + DoubleCTRNN base classes
    │   ├── rnn.py, rnn_d.py, rnn_ff.py, rnn_d_ff.py, rnn_action.py,
    │   ├── rnn_autoreg.py, rnn_d_autoreg.py,
    │   ├── vha.py, vha_d.py
    │   └── __init__.py          # MODELS registry -> {name: class}
    │
    ├── envs/
    │   ├── base_wrapper.py      # BaseWrapper for env decorators
    │   ├── random_vector.py     # RandomVecEnv (the paper task)
    │   ├── grid_wrapper.py      # GridWrapper: adds the grid-cell input
    │   └── __init__.py
    │
    ├── modules/
    │   └── ctrnn.py             # the CTRNN cell used by every baseline
    │
    ├── analysis/                # per-seed analysis dispatcher
    │   ├── dispatch.py          # run_seed / run_dir -- registry-driven
    │   └── registry.py          # ANALYSES = {'autocorr', 'firing_rate', ...}
    │
    ├── grid_cells/              # low-level grid-cell math
    │
    └── analyze_utils/           # aggregate analysis library
        ├── collect.py           # draw_generalization, draw_aggregated_periodicity
        ├── rdm.py               # per-condition patterns + RDM helpers
        ├── pca.py               # trajectory PCA + rotation grids
        ├── autocorr.py          # per-seed autocorrelation / periodicity npys
        └── firing_rate.py       # per-seed firing-rate figures
```

## Citation

If you use this code, please cite:

```bibtex
@article{hwang2026structural,
  title={Structural generalization and continual learning enabled by
         factorized entorhinal-hippocampal memory and entorhinal-parietal
         action circuits},
  author={Hwang, Jaedong and Neupane, Sujaya and Jazayeri, Mehrdad and
          Fiete, Ila},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
