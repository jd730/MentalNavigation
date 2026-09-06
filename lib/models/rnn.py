"""RNN baseline (was 'vRNN'): single CTRNN with a one-layer linear encoder."""
from ._base import SingleCTRNN


class RNN(SingleCTRNN):
    """Single CTRNN, simple Linear encoder (hidden_dim wide)."""

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
