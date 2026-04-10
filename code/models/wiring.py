# Wiring classes for Neural Circuit Policies
# Based on keras-ncp by Mathias Lechner (Apache 2.0 License)

import numpy as np


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.input_dim = None
        self.output_dim = None

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_shape):
        input_dim = int(input_shape[1])
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Conflicting input dimensions: set_input_dim() was called with "
                f"{self.input_dim} but actual input has dimension {input_dim}"
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self):
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self):
        return np.copy(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(f"Invalid src neuron {src} for {self.units} units")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid dest neuron {dest} for {self.units} units")
        if polarity not in [-1, 1]:
            raise ValueError(f"Polarity must be -1 or +1, got {polarity}")
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError("Cannot add sensory synapses before build()")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"Invalid sensory src {src} for input_dim {self.input_dim}")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"Invalid dest neuron {dest} for {self.units} units")
        if polarity not in [-1, 1]:
            raise ValueError(f"Polarity must be -1 or +1, got {polarity}")
        self.sensory_adjacency_matrix[src, dest] = polarity

    def get_config(self):
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        wiring = Wiring(config["units"])
        wiring.adjacency_matrix = config["adjacency_matrix"]
        wiring.sensory_adjacency_matrix = config["sensory_adjacency_matrix"]
        wiring.input_dim = config["input_dim"]
        wiring.output_dim = config["output_dim"]
        return wiring


class FullyConnected(Wiring):
    def __init__(self, units, output_dim=None, erev_init_seed=1111, self_connections=True):
        super().__init__(units)
        if output_dim is None:
            output_dim = units
        self.self_connections = self_connections
        self.set_output_dim(output_dim)
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)


class NCP(Wiring):
    def __init__(
        self,
        inter_neurons,
        command_neurons,
        motor_neurons,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin,
        seed=22222,
    ):
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

        # Neuron IDs: [0..motor | command | inter]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(range(
            self._num_motor_neurons,
            self._num_motor_neurons + self._num_command_neurons,
        ))
        self._inter_neurons = list(range(
            self._num_motor_neurons + self._num_command_neurons,
            self._num_motor_neurons + self._num_command_neurons + self._num_inter_neurons,
        ))

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(
                f"Motor fanin {self._motor_fanin} > command neurons {self._num_command_neurons}"
            )
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(
                f"Sensory fanout {self._sensory_fanout} > inter neurons {self._num_inter_neurons}"
            )
        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(
                f"Inter fanout {self._inter_fanout} > command neurons {self._num_command_neurons}"
            )

    def get_type_of_neuron(self, neuron_id):
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def _build_sensory_to_inter_layer(self):
        unreachable = list(self._inter_neurons)
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                self._inter_neurons, size=self._sensory_fanout, replace=False
            ):
                if dest in unreachable:
                    unreachable.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        mean_fanin = int(
            self._num_sensory_neurons * self._sensory_fanout / self._num_inter_neurons
        )
        mean_fanin = np.clip(mean_fanin, 1, self._num_sensory_neurons)
        for dest in unreachable:
            for src in self._rng.choice(
                self._sensory_neurons, size=mean_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def _build_inter_to_command_layer(self):
        unreachable = list(self._command_neurons)
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                self._command_neurons, size=self._inter_fanout, replace=False
            ):
                if dest in unreachable:
                    unreachable.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        mean_fanin = int(
            self._num_inter_neurons * self._inter_fanout / self._num_command_neurons
        )
        mean_fanin = np.clip(mean_fanin, 1, self._num_command_neurons)
        for dest in unreachable:
            for src in self._rng.choice(
                self._inter_neurons, size=mean_fanin, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def _build_recurrent_command_layer(self):
        for _ in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def _build_command_to_motor_layer(self):
        unreachable = list(self._command_neurons)
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                self._command_neurons, size=self._motor_fanin, replace=False
            ):
                if src in unreachable:
                    unreachable.remove(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        mean_fanout = int(
            self._num_motor_neurons * self._motor_fanin / self._num_command_neurons
        )
        mean_fanout = np.clip(mean_fanout, 1, self._num_motor_neurons)
        for src in unreachable:
            for dest in self._rng.choice(
                self._motor_neurons, size=mean_fanout, replace=False
            ):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))

        self._build_sensory_to_inter_layer()
        self._build_inter_to_command_layer()
        self._build_recurrent_command_layer()
        self._build_command_to_motor_layer()
