"""RNN_autoreg baseline: single CTRNN + autoregressive reconstruction head.

The "future" decoder name controls reconstruction-loss behavior
inside lib.exp; the architectural piece is just that a decoder head
exists (recon=True).
"""
from ._base import SingleCTRNN


class RNNAutoreg(SingleCTRNN):
    """Single CTRNN with a reconstruction head (recon_dim defaults to input_dim // 2)."""

    def __init__(self, input_dim, hidden_dim, num_class, alpha=None,
                 grid_cells: bool = False, periods=None,
                 scaling_factor: bool = False, recon_dim=None):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class=num_class,
            recon=True,
            alpha=alpha,
            encoder_dim=[hidden_dim],
            recon_dim=recon_dim,
            scaling_factor=scaling_factor,
            grid_cells=grid_cells,
            periods=periods,
        )
