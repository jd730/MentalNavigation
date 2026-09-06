# Mental navigation human psychophysics — anonymized data

Trial-level behavioral data for 7 subjects (`sub_01.mat` … `sub_07.mat`), 6 sessions each. All identifying information (names, dates, file paths) has been removed; subject numbering matches the trial-count table in the paper.

## Data format

Each `.mat` file contains one array per field, one entry per trial
(trials in chronological order across sessions):

| Field         | Description                                            |
|---------------|--------------------------------------------------------|
| `talong`      | true (target) vector duration (s), signed by direction |
| `tplong`      | produced vector duration (s)                           |
| `rtlong`      | reaction time (s)                                      |
| `trainlong`   | 1 = visual navigation (vnav), 0 = mental navigation (mnav) |
| `env`         | environment index (1–3)                                |
| `sessionlong` | session index (0–5)                                    |
| `blocklong`   | block index within session                             |
| `curr`, `targ`| current and target landmark indices                    |
| `att`         | attempt number for the trial                           |
| `masklong`    | 1 = intervening landmarks masked                       |
| `seq`         | sequence condition                                     |
| `seenonly`    | 1 = landmark pair only passively seen                  |
| `seenpair`    | 1 = trained (seen) pair, 0 = novel pair                |
| `condlong`    | condition code                                         |
| `subject`     | anonymized subject number (1–7)                        |

Load in Python with `scipy.io.loadmat`, or in MATLAB with `load`.

## Reproducing the figures

```
python3 plot_figures.py
```

requires `numpy`, `scipy`, `matplotlib`. Writes Fig. 2 panels e/f/g (learning curves + example-subject scatters, and the catastrophic-forgetting panel) as PDF/PNG into `figures/`.
