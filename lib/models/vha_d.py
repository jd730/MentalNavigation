"""VHA-D baseline: double CTRNN with grid-cell wrapper enabled.

**This is the canonical paper model** (rev+abs norm=5 defaults +
grid-cell configuration). Architecturally the same as ``RNN-D``;
the name marks the variant that runs with the grid-cell input
wrapper turned on. ``train.py`` sees ``-grid_cells`` and builds the
env via ``GridWrapper`` and passes ``grid_cells=True`` to the model
(which turns on the periodic-subtract input transform).
"""
from ._base import DoubleCTRNN


class VHAD(DoubleCTRNN):
    """Double CTRNN + reg_dist head; designed to run with ``-gcpc g`` (paper default)."""

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
