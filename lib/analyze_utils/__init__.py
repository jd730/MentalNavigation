"""Analysis utilities for Mental Navigation RNN internal dynamics.

Submodules group analyses by theme (tdr, firing_rate, autocorr, regression,
neurons, pca, orchestration, collect). Re-exports below let callers do
`from lib.analyze_utils import run_firing_rate` without knowing which
submodule it lives in.
"""

from .tdr import (
    TDR,
    visualize_TDR,
    get_z_score,
    visualize_neuronal_coeff,
    get_xy,
    visualize_figure3d,
    _run_tdr,
    run_tdr,
)
from .firing_rate import (
    run_firing_rate,
    run_firing_rate_each_trial,
    run_firing_rate_shifted,
    run_firing_rate_stretched,
)
from .autocorr import run_autocorr, analyze_periodicity
from .regression import run_regression
from .dim_reduction import run_non_linear_reduction
from .neurons import analyze_var, analyze_neurons, analyze_speed
from .pca import (
    split_by_signed_distance,
    smooth_and_aggregate,
    fit_pca,
    visualize_specific_view,
    plot_pca_for_mat,
    plot_pca_for_sweep,
)
from .orchestration import (
    run_visualization,
    extract_hiddens,
    run_shifted_visualization,
    run_stretched_visualization,
)
from . import collect             # collect.read_results, etc.
from . import rdm                 # rdm.<helper>, e.g. variant_subdir
from . import rdm_aggregate       # collect, add_specificity_columns, wilcoxon_table, plot_*
