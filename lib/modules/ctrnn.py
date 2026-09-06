import torch
import torch.nn as nn


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, tau=100, num_layers=1, alpha=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        if alpha is not None:
            self.alpha = alpha
        else:
            if dt is None:
                alpha = 1
            else:
                alpha = dt / self.tau
            self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)

        self.num_layers = num_layers
        module_list = []
        for _ in range(num_layers):
            module_list.append(nn.Linear(hidden_size, hidden_size))
        self.rnn_cells = nn.ModuleList(module_list)

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.rnn_cells[0].weight.device)

    def forward(self, input, hidden):
        """Propogate input through the network."""
        hiddens_out = self.init_hidden()
        x = self.input2h(input)
        for i, cell in enumerate(self.rnn_cells):
            x = torch.tanh(x + cell(hidden[i:i+1]))
            x = hidden[i:i+1] * (1 - self.alpha) + x * self.alpha
            # RNG-state preservation (for reproducibility): keep one
            # `torch.randn_like` draw per cell per step so the global torch
            # RNG advances identically to the pre-refactor code that had
            # `+ torch.randn_like(x) * self.noise_var`. noise_var was
            # always 0, so the value contribution was always 0; only the
            # side effect of advancing the RNG matters downstream
            # (env reconstruction, env_cl shuffle).
            _ = torch.randn_like(x)
            hiddens_out[i, 0] = x[0, 0]
        return x, hiddens_out
