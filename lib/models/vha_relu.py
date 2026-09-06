"""VHA-ReLU: single-CTRNN VHA using the Vector-HaSH ReLU sensory->place
nonlinearity instead of sign.

Architecturally identical to VHA; the only difference is the flag
``use_relu=True`` on the grid-cell wrapper. train.py detects the name
and threads it through arg_gcpc.
"""
from .vha import VHA


class VHAReLU(VHA):
    pass
