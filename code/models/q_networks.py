import torch
import torch.nn as nn
from models.ltc_cell import LTCCell
from models.wiring import NCP, FullyConnected


class QNetworkBase(nn.Module):
    """Base class for all Q-networks. Unified interface for DQN agent."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs, hidden=None):
        """
        Args:
            obs: (batch, obs_dim) single step or (batch, seq_len, obs_dim) sequence
            hidden: architecture-specific hidden state, or None
        Returns:
            q_values: (batch, act_dim)
            new_hidden: updated hidden state
        """
        raise NotImplementedError

    def init_hidden(self, batch_size: int):
        return None

    @property
    def is_recurrent(self) -> bool:
        raise NotImplementedError


class MLPQNetwork(QNetworkBase):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=None):
        super().__init__(obs_dim, act_dim)
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    @property
    def is_recurrent(self):
        return False

    def forward(self, obs, hidden=None):
        if obs.dim() == 3:
            obs = obs[:, -1, :]
        return self.net(obs), None


class LSTMQNetwork(QNetworkBase):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__(obs_dim, act_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, act_dim)

    @property
    def is_recurrent(self):
        return True

    def init_hidden(self, batch_size: int):
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(self, obs, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if hidden is None:
            hidden = self.init_hidden(obs.size(0))
        out, new_hidden = self.lstm(obs, hidden)
        q = self.head(out[:, -1, :])
        return q, new_hidden


class GRUQNetwork(QNetworkBase):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__(obs_dim, act_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(obs_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, act_dim)

    @property
    def is_recurrent(self):
        return True

    def init_hidden(self, batch_size: int):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, obs, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if hidden is None:
            hidden = self.init_hidden(obs.size(0))
        out, new_hidden = self.gru(obs, hidden)
        q = self.head(out[:, -1, :])
        return q, new_hidden


class NCPQNetwork(QNetworkBase):
    """NCP-based Q-network using LTCCell + dedicated Q-head.

    Uses the full hidden state (all neuron activations) as input to a linear
    Q-value head, rather than relying solely on the 5 motor neuron outputs.
    This gives NCP comparable expressivity to LSTM/GRU while preserving the
    interpretable liquid dynamics.
    """

    def __init__(self, obs_dim: int, act_dim: int, wiring=None,
                 inter_neurons=12, command_neurons=8,
                 sensory_fanout=4, inter_fanout=4,
                 recurrent_command_synapses=4, motor_fanin=4):
        super().__init__(obs_dim, act_dim)
        if wiring is None:
            wiring = NCP(
                inter_neurons=inter_neurons,
                command_neurons=command_neurons,
                motor_neurons=act_dim,
                sensory_fanout=sensory_fanout,
                inter_fanout=inter_fanout,
                recurrent_command_synapses=recurrent_command_synapses,
                motor_fanin=motor_fanin,
            )
        self.wiring = wiring
        self.ltc = LTCCell(wiring, in_features=obs_dim)
        # Two-layer Q-head: full state → hidden → Q-values
        state_size = wiring.units  # inter + command + motor
        q_hidden = max(32, state_size)
        self.q_head = nn.Sequential(
            nn.Linear(state_size, q_hidden),
            nn.ReLU(),
            nn.Linear(q_hidden, act_dim),
        )

    @property
    def is_recurrent(self):
        return True

    def init_hidden(self, batch_size: int):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.ltc.state_size, device=device)

    def forward(self, obs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(obs.size(0))
        if obs.dim() == 3:
            for t in range(obs.size(1)):
                _, hidden = self.ltc(obs[:, t, :], hidden)
        else:
            _, hidden = self.ltc(obs, hidden)
        # Use full hidden state (all neurons) for Q-value computation
        q_values = self.q_head(hidden)
        return q_values, hidden

    def get_activations(self, obs, hidden=None):
        """Return per-layer neuron activations for interpretability."""
        if hidden is None:
            hidden = self.init_hidden(obs.size(0))
        if obs.dim() == 3:
            for t in range(obs.size(1)):
                _, hidden = self.ltc(obs[:, t, :], hidden)
        else:
            _, hidden = self.ltc(obs, hidden)

        q_values = self.q_head(hidden)
        n_motor = self.wiring._num_motor_neurons
        n_cmd = self.wiring._num_command_neurons
        return {
            "output": q_values,
            "motor": hidden[:, :n_motor],
            "command": hidden[:, n_motor:n_motor + n_cmd],
            "inter": hidden[:, n_motor + n_cmd:],
            "full_state": hidden,
        }


class RandomQNetwork(QNetworkBase):
    """Random baseline: outputs uniform random Q-values. No learning."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__(obs_dim, act_dim)
        # Dummy parameter so .parameters() is not empty (avoids optimizer errors)
        self._dummy = nn.Parameter(torch.zeros(1))

    @property
    def is_recurrent(self):
        return False

    def forward(self, obs, hidden=None):
        if obs.dim() == 3:
            batch = obs.size(0)
        else:
            batch = obs.size(0)
        return torch.randn(batch, self.act_dim, device=obs.device), None


class FCLTCQNetwork(QNetworkBase):
    """Fully-Connected LTC Q-network. Same ODE dynamics as NCP but with
    all-to-all connectivity instead of sparse 4-layer wiring.
    Serves as a direct ablation to show the value of NCP's structured sparsity.
    """

    def __init__(self, obs_dim: int, act_dim: int, units: int = 25):
        super().__init__(obs_dim, act_dim)
        wiring = FullyConnected(units, output_dim=act_dim)
        self.wiring = wiring
        self.ltc = LTCCell(wiring, in_features=obs_dim)
        q_hidden = max(32, units)
        self.q_head = nn.Sequential(
            nn.Linear(units, q_hidden),
            nn.ReLU(),
            nn.Linear(q_hidden, act_dim),
        )

    @property
    def is_recurrent(self):
        return True

    def init_hidden(self, batch_size: int):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.ltc.state_size, device=device)

    def forward(self, obs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(obs.size(0))
        if obs.dim() == 3:
            for t in range(obs.size(1)):
                _, hidden = self.ltc(obs[:, t, :], hidden)
        else:
            _, hidden = self.ltc(obs, hidden)
        q_values = self.q_head(hidden)
        return q_values, hidden


def build_q_network(arch: str, obs_dim: int, act_dim: int, config: dict,
                    wiring=None) -> QNetworkBase:
    """Factory function to create Q-networks."""
    if arch == "mlp":
        return MLPQNetwork(obs_dim, act_dim, config["mlp"]["hidden_sizes"])
    elif arch == "lstm":
        return LSTMQNetwork(obs_dim, act_dim, config["rnn"]["hidden_size"],
                            config["rnn"]["num_layers"])
    elif arch == "gru":
        return GRUQNetwork(obs_dim, act_dim, config["rnn"]["hidden_size"],
                           config["rnn"]["num_layers"])
    elif arch == "ncp":
        if wiring is not None:
            return NCPQNetwork(obs_dim, act_dim, wiring=wiring)
        return NCPQNetwork(obs_dim, act_dim, **config["ncp"])
    elif arch == "fc_ltc":
        return FCLTCQNetwork(obs_dim, act_dim,
                             units=config.get("fc_ltc", {}).get("units", 25))
    elif arch == "random":
        return RandomQNetwork(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
