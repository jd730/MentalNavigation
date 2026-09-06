"""Per-baseline model classes.

Each baseline lives in its own file and exposes one class. ``MODELS``
below maps the CLI ``-model <name>`` value to the class to instantiate;
train.py uses it for dispatch.

``Mental`` (defined in :mod:`lib.models._base`) is the shared
implementation -- a union of all the per-baseline knobs. It is
deliberately not re-exported here: callers should pick a concrete
baseline class instead.
"""
from .rnn import RNN
from .rnn_d import RNND
from .rnn_ff import RNNFF
from .rnn_d_ff import RNNDFF
from .rnn_action import RNNAction
from .rnn_autoreg import RNNAutoreg
from .rnn_d_autoreg import RNNDAutoreg
from .vha import VHA
from .vha_d import VHAD
from .vha_relu import VHAReLU
from .vha_relu_d import VHAReLUD

MODELS = {
    # no grid cells (basic baselines)
    'RNN':           RNN,
    'RNN-D':         RNND,
    'RNN_FF':        RNNFF,
    'RNN-D_FF':      RNNDFF,
    'RNN_action':    RNNAction,
    'RNN_autoreg':   RNNAutoreg,
    'RNN-D_autoreg': RNNDAutoreg,
    # grid cells (gcpc='g') -- VHA-D is the canonical paper model
    'VHA':           VHA,
    'VHA-D':         VHAD,
    # ReLU variants (Vector-HaSH nonlin(x)=relu(x) on place cells)
    'VHA-ReLU':      VHAReLU,
    'VHA-ReLU-D':    VHAReLUD,
}

__all__ = ['MODELS',
           'RNN', 'RNND', 'RNNFF', 'RNNDFF', 'RNNAction',
           'RNNAutoreg', 'RNNDAutoreg',
           'VHA', 'VHAD',
           'VHAReLU', 'VHAReLUD']
