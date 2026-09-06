"""RNN_FF baseline (was 'deeper'): RNN with a 3-layer feed-forward encoder."""
from ._base import SingleCTRNN


class RNNFF(SingleCTRNN):
    """Single CTRNN with a [400, 36, hidden_dim] linear encoder."""

    def __init__(self, input_dim, hidden_dim, num_class, alpha=None,
                 grid_cells: bool = False, periods=None,
                 scaling_factor: bool = False):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class=num_class,
            recon=False,
            alpha=alpha,
            encoder_dim=[400, 36, hidden_dim],
            scaling_factor=scaling_factor,
            grid_cells=grid_cells,
            periods=periods,
        )
