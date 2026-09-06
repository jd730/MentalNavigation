"""VHA-ReLU-D: double-CTRNN VHA-D using the Vector-HaSH ReLU
sensory->place nonlinearity instead of sign.

Architecturally identical to VHA-D; the only difference is the flag
``use_relu=True`` on the grid-cell wrapper. train.py detects the name
and threads it through arg_gcpc.
"""
from .vha_d import VHAD


class VHAReLUD(VHAD):
    pass
