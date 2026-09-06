"""RNN-D baseline (was 'vRNN2'): double CTRNN with a distance-regression head.

This is the canonical model for the current paper (rev+abs norm=5 defaults).
"""
from ._base import DoubleCTRNN


class RNND(DoubleCTRNN):
    """Double CTRNN (action RNN + distance RNN) with reg_dist head."""

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
