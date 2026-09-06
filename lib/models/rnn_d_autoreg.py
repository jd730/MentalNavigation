"""RNN-D_autoreg baseline: double CTRNN + reg_dist + autoregressive reconstruction head."""
from ._base import DoubleCTRNN


class RNNDAutoreg(DoubleCTRNN):
    """Double CTRNN + reg_dist + reconstruction head."""

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
