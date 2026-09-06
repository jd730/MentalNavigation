# Monkey single-neuron data — source and paths

Example single-neuron firing-rate traces from the monkey mental-navigation
recordings. This directory holds the conversion script that turns the
per-neuron CSV exports into the `hiddens` list format consumed by
`run_monkey.py`.

## Where the data comes from

The raw electrophysiology, recorded from two NHPs performing the mental
navigation task (Neupane, Fiete & Jazayeri), is available for
[entorhinal cortex (EC)](https://www.dropbox.com/scl/fo/nw6zals6ayf0w7vszysbl/h?rlkey=e4c8ee6rr9iv7k218ybym6e1b&e=1&dl=0)
and [posterior parietal cortex (PPC)](https://www.dropbox.com/scl/fo/4uorhieclbnllpipli9h7/AKytcQB9kSgP_YLCtNuRHEs?rlkey=8ahrvo4gv4jf4nr6onkpx7y3b&e=1&dl=0).

Recording probes were V-probes (32 or 64 channels, Plexon Inc.).

## Files expected in this directory

`convert_monkey.py` reads four per-neuron CSVs (the four example units shown in
the paper), exported from the sessions linked above:

```
monkey_EC_neur13_sess1.csv
monkey_EC_neur2_sess5.csv
monkey_PPC_neur04_sess1.csv
monkey_PPC_neur19_sess1.csv
```

Each CSV is one row per (distance condition x time bin), with columns:

| Column                   | Description                                      |
|--------------------------|--------------------------------------------------|
| `nhp_neuron_id`          | session + unit identifier                        |
| `dist`                   | signed integer distance condition                |
| `time_JS_onset`          | time relative to joystick onset (s, 0.04 s bins) |
| `firing_rate`            | onset-aligned trial-averaged firing rate (Hz)    |
| `firing_rate_SEM`        | SEM of the onset-aligned rate                    |
| `time_JS_offset`         | time relative to joystick offset (s)             |
| `firing_rate_offset`     | offset-aligned trial-averaged firing rate (Hz)   |
| `firing_rate_SEM_offset` | SEM of the offset-aligned rate                   |

These CSVs are not tracked in git — place them here after downloading.

## Conversion

```bash
python monkey_single_neurons/convert_monkey.py
```

Writes `hiddens_onset.pkl` and `hiddens_offset.pkl` into this directory. Each
pickle is `{'hiddens': [...], 'neuron_names': [...], 'dists': [...]}`, where
`hiddens[i]` has shape `[T_i, 1, D]` (time x batch x neurons), matching the RNN
hidden-state convention used by the analysis code.

Then run the shift / stretch firing-rate analysis:

```bash
python run_monkey.py          # reads hiddens_onset.pkl -> results/monkey/
```

## Related: session-level tensors

`rdm_analysis.py` and `pca_utils/` use the same recordings in a different,
session-level form — `7a_amadeus06242019_a_neur_tensor_joyon.mat` (PPC) and
`ec_amadeus08292019_a_neur_tensor_joyon.mat` (EC), i.e. `(Neuron, Time, Trial)`
spike tensors with trial metadata. `rdm_analysis.py` resolves those filenames
relative to the working directory (see `MONKEY_FILES` in
`lib/analyze_utils/rdm.py`); the `pca_utils/` scripts look for them under
`monkey_raw_data/`. Both come from the same recordings linked above.

## Citation

If you use this data, please cite the accompanying paper (see the repository
root `README.md`) and the original recordings by Neupane, Fiete & Jazayeri.
