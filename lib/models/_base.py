"""Shared base classes for the per-baseline model files.

Two architectures live here:

* ``SingleCTRNN`` -- one CTRNN core. Hidden state is detached before
  the recurrent step (matching the legacy ``ctrnn`` mode).
  Used by RNN, RNN_FF, RNN_action, RNN_autoreg, VHA.
* ``DoubleCTRNN`` -- two CTRNN cores stacked, with a distance head on
  the base RNN output (``first_reg_dist``) and gradients allowed to
  flow into the base (``nodetach``).
  Used by RNN-D, RNN-D_FF, RNN-D_autoreg, VHA-D.

Both classes accept the same outer-knob set (encoder_dim / recon /
recon_dim / scaling_factor / grid_cells / periods). The legacy
``Mental`` class that fused both architectures behind an
``is_double`` flag is gone; each baseline now subclasses exactly one
of these two bases.

**Module-creation order** is load-bearing for seed reproducibility:
each ``__init__`` follows the original ``Mental.__init__`` order
(encoder -> recurrent core(s) -> [dist_classifier in Double] ->
classifier -> decoder -> velocity_estimator) so that per-module
parameter init consumes the global ``torch`` RNG in the same
sequence as before this refactor.
"""
import torch.nn as nn

from ..modules import CTRNN


def _build_encoder(input_dim, encoder_dim):
    """Linear+ReLU stack matching the legacy Mental encoder.

    Returns ``(module_or_None, out_dim)``. ``out_dim`` equals
    ``input_dim`` when ``encoder_dim`` is empty (no encoder).
    """
    if len(encoder_dim) == 0:
        return None, input_dim
    dims = [input_dim] + [int(e) for e in encoder_dim]
    modules = []
    for i in range(len(encoder_dim)):
        modules.append(nn.Linear(dims[i], dims[i+1]))
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules), dims[-1]


def _build_decoder(hidden_dim, input_dim, recon_dim):
    """Reconstruction head (matches the legacy Mental.decoder)."""
    recon_out = recon_dim if recon_dim is not None else input_dim // 2
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, recon_out))


def _build_velocity_estimator(hidden_dim):
    """Scaling-factor (``internal_corr_suc_3``) head -> 3 outputs."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 3))


def _grid_subtract(x, len_g, periods):
    """Periodic-mod current-vs-target grid-code subtraction (in-place).

    Was ``subtract_mode='positive'`` + ``conv_int=True`` in the legacy
    Mental.forward; we only call this when ``grid_cells=True``.
    """
    C = x.shape[-1]
    x[:, :len_g] -= x[:, C//2:C//2+len_g]
    x[:, C//2:C//2+len_g] = 0
    x[:, :len_g] = (x[:, :len_g] + periods) % periods


class SingleCTRNN(nn.Module):
    """One CTRNN; detach hidden before the recurrent step."""

    def __init__(self, input_dim, hidden_dim, num_class=3,
                 alpha=None, encoder_dim=[],
                 recon=False, recon_dim=None,
                 scaling_factor: bool = False,
                 grid_cells: bool = False, periods=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.alpha = alpha
        self.use_velocity = scaling_factor
        self.grid_cells = grid_cells
        self.periods = periods if grid_cells else None
        self.len_g = periods.shape[1] if grid_cells else 0

        # Module creation order matches legacy Mental.__init__ exactly
        # (encoder -> rnn -> classifier -> decoder -> velocity_estimator)
        # so the seeded RNG draws line up with pre-refactor runs.
        self.encoder, rnn_input_dim = _build_encoder(input_dim, encoder_dim)
        self.rnn = CTRNN(input_size=rnn_input_dim, hidden_size=hidden_dim,
                         num_layers=1, alpha=alpha)
        self.classifier = nn.Linear(hidden_dim, num_class)

        self.recon = recon
        if recon:
            self.decoder = _build_decoder(hidden_dim, input_dim, recon_dim)
        if scaling_factor:
            self.velocity_estimator = _build_velocity_estimator(hidden_dim)

        self.hidden = None

    def init_hidden(self):
        """Return a fresh zero hidden state from the CTRNN core."""
        return self.rnn.init_hidden()

    def cuda(self):
        """Move the model + the ``periods`` tensor (if any) to GPU."""
        if self.periods is not None:
            self.periods = self.periods.cuda()
        return super().cuda()

    def reset(self) -> None:
        """Re-initialise the hidden state."""
        self.hidden = self.init_hidden()

    def forward(self, x, is_log: bool = False):
        """Single-CTRNN forward: encoder -> detach -> CTRNN -> classifier.

        Returns ``(logits, None)`` when ``is_log=False``;
        ``(logits, output_dict)`` otherwise.
        """
        if self.grid_cells:
            _grid_subtract(x, self.len_g, self.periods)

        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        if self.hidden is None:
            self.reset()

        if self.encoder is not None:
            x = self.encoder(x)
        fc_hidden = x

        x = x.detach()
        x, hidden = self.rnn(x, self.hidden)
        self.hidden = hidden

        if len(x.shape) == 3:
            x = x[:, 0]
        x = x.relu()
        out = self.classifier(x)

        if not is_log:
            return out, None

        output = {'logit': out, 'hidden': hidden, 'fc_hidden': fc_hidden}
        if self.use_velocity:
            output['velocity'] = self.velocity_estimator(x)
        if self.recon:
            output['recon'] = self.decoder(x)
        return out, output


class DoubleCTRNN(nn.Module):
    """Two stacked CTRNNs + first_reg_dist head; no detach (nodetach)."""

    def __init__(self, input_dim, hidden_dim, num_class=3,
                 alpha=None, encoder_dim=[],
                 recon=False, recon_dim=None,
                 scaling_factor: bool = False,
                 grid_cells: bool = False, periods=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.alpha = alpha
        self.use_velocity = scaling_factor
        self.grid_cells = grid_cells
        self.periods = periods if grid_cells else None
        self.len_g = periods.shape[1] if grid_cells else 0

        # Module creation order matches legacy Mental.__init__ exactly
        # (encoder -> rnn_base -> rnn -> dist_classifier -> classifier
        # -> decoder -> velocity_estimator) so the seeded RNG draws line
        # up with pre-refactor runs.
        self.encoder, rnn_input_dim = _build_encoder(input_dim, encoder_dim)
        self.rnn_base = CTRNN(input_size=rnn_input_dim, hidden_size=hidden_dim,
                              num_layers=1, alpha=alpha)
        self.rnn = CTRNN(input_size=hidden_dim, hidden_size=hidden_dim,
                         num_layers=1, alpha=alpha)
        # first_reg_dist: distance head on the base RNN output.
        self.dist_classifier = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_class)

        self.recon = recon
        if recon:
            self.decoder = _build_decoder(hidden_dim, input_dim, recon_dim)
        if scaling_factor:
            self.velocity_estimator = _build_velocity_estimator(hidden_dim)

        self.hidden = None
        self.hidden_base = None

    def init_hidden(self):
        """Return a fresh zero hidden state from the action CTRNN core."""
        return self.rnn.init_hidden()

    def cuda(self):
        """Move the model + the ``periods`` tensor (if any) to GPU."""
        if self.periods is not None:
            self.periods = self.periods.cuda()
        return super().cuda()

    def reset(self) -> None:
        """Re-initialise both hidden states."""
        self.hidden = self.init_hidden()
        self.hidden_base = self.rnn_base.init_hidden()

    def forward(self, x, is_log: bool = False):
        """Double-CTRNN forward: encoder -> rnn_base -> dist_head -> rnn -> classifier.

        Returns ``((logits, st_tg_dist), None)`` when ``is_log=False``;
        ``((logits, st_tg_dist), output_dict)`` otherwise.
        """
        if self.grid_cells:
            _grid_subtract(x, self.len_g, self.periods)

        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        if self.hidden is None:
            self.reset()

        if self.encoder is not None:
            x = self.encoder(x)
        fc_hidden = x

        x, hidden_base = self.rnn_base(x, self.hidden_base)
        self.hidden_base = hidden_base
        x = x.relu()
        st_tg_dist = self.dist_classifier(x)
        # 'nodetach': gradients flow into the base RNN.

        x, hidden = self.rnn(x, self.hidden)
        self.hidden = hidden

        if len(x.shape) == 3:
            x = x[:, 0]
        x = x.relu()
        out = self.classifier(x)

        if not is_log:
            return (out, st_tg_dist), None

        output = {'logit': out, 'hidden': hidden, 'fc_hidden': fc_hidden,
                  'hidden_base': hidden_base, 'st_tg_dist': st_tg_dist}
        if self.use_velocity:
            output['velocity'] = self.velocity_estimator(x)
        if self.recon:
            output['recon'] = self.decoder(x)
        return out, output


__all__ = ['SingleCTRNN', 'DoubleCTRNN']
