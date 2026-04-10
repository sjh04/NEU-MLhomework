# Liquid Time-Constant (LTC) Cell - PyTorch Implementation
# Based on keras-ncp by Mathias Lechner (Apache 2.0 License)

import torch
import torch.nn as nn
import numpy as np


class LTCCell(nn.Module):
    def __init__(
        self,
        wiring,
        in_features=None,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
    ):
        super().__init__()
        if in_features is not None:
            wiring.build((None, in_features))
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. "
                "Please pass 'in_features' or call 'wiring.build()'."
            )
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._allocate_parameters()

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix))

    def _add_weight(self, name, init_value):
        param = nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self._add_weight(
            "gleak", self._get_init_value((self.state_size,), "gleak")
        )
        self._params["vleak"] = self._add_weight(
            "vleak", self._get_init_value((self.state_size,), "vleak")
        )
        self._params["cm"] = self._add_weight(
            "cm", self._get_init_value((self.state_size,), "cm")
        )
        self._params["sigma"] = self._add_weight(
            "sigma", self._get_init_value((self.state_size, self.state_size), "sigma")
        )
        self._params["mu"] = self._add_weight(
            "mu", self._get_init_value((self.state_size, self.state_size), "mu")
        )
        self._params["w"] = self._add_weight(
            "w", self._get_init_value((self.state_size, self.state_size), "w")
        )
        self._params["erev"] = self._add_weight(
            "erev", torch.Tensor(self._wiring.erev_initializer())
        )
        self._params["sensory_sigma"] = self._add_weight(
            "sensory_sigma",
            self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma"),
        )
        self._params["sensory_mu"] = self._add_weight(
            "sensory_mu",
            self._get_init_value((self.sensory_size, self.state_size), "sensory_mu"),
        )
        self._params["sensory_w"] = self._add_weight(
            "sensory_w",
            self._get_init_value((self.sensory_size, self.state_size), "sensory_w"),
        )
        self._params["sensory_erev"] = self._add_weight(
            "sensory_erev",
            torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        # Sparsity masks (non-learnable) - registered as buffers for device transfer
        self.register_buffer(
            "sparsity_mask", torch.Tensor(np.abs(self._wiring.adjacency_matrix))
        )
        self.register_buffer(
            "sensory_sparsity_mask",
            torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)),
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self._add_weight(
                "input_w", torch.ones((self.sensory_size,))
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self._add_weight(
                "input_b", torch.zeros((self.sensory_size,))
            )
        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self._add_weight(
                "output_w", torch.ones((self.motor_size,))
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self._add_weight(
                "output_b", torch.zeros((self.motor_size,))
            )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        sensory_w_activation = self._params["sensory_w"] * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = sensory_w_activation * self.sensory_sparsity_mask

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        cm_t = self._params["cm"] / (elapsed_time / self._ode_unfolds)

        for t in range(self._ode_unfolds):
            w_activation = self._params["w"] * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )
            w_activation = w_activation * self.sparsity_mask

            rev_activation = w_activation * self._params["erev"]

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = (
                cm_t * v_pre
                + self._params["gleak"] * self._params["vleak"]
                + w_numerator
            )
            denominator = cm_t + self._params["gleak"] + w_denominator

            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        self._params["w"].data = torch.relu(self._params["w"].data)
        self._params["sensory_w"].data = torch.relu(self._params["sensory_w"].data)
        self._params["cm"].data = torch.relu(self._params["cm"].data)
        self._params["gleak"].data = torch.relu(self._params["gleak"].data)

    def forward(self, inputs, states):
        elapsed_time = 1.0
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, elapsed_time)
        outputs = self._map_outputs(next_state)
        return outputs, next_state
