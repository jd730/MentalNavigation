"""RNN_action baseline (was 'cat_action'): single CTRNN that takes the previous action concatenated to its input.

The architectural delta is just `input_dim += 1` upstream (train.py
already does this when args.cat_action is set); this class is
otherwise identical to RNN. Exists so reviewers can refer to the
baseline by a single canonical name.
"""
from ._base import SingleCTRNN


class RNNAction(SingleCTRNN):
    """Single CTRNN; train.py is expected to pass `input_dim` already incremented by 1."""

    def __init__(self, input_dim, hidden_dim, num_class, alpha=None,
                 grid_cells: bool = False, periods=None,
                 scaling_factor: bool = False):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class=num_class,
            recon=False,
            alpha=alpha,
            encoder_dim=[hidden_dim],
            scaling_factor=scaling_factor,
            grid_cells=grid_cells,
            periods=periods,
        )
