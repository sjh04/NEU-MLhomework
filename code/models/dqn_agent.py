import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.q_networks import build_q_network
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, arch: str, config: dict,
                 wiring=None, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.arch = arch
        self.device = torch.device(device)
        self.cfg = config["dqn"]

        self.q_net = build_q_network(arch, obs_dim, act_dim, config, wiring).to(self.device)
        self.target_net = build_q_network(arch, obs_dim, act_dim, config, wiring).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg["lr"])
        self.buffer = ReplayBuffer(
            self.cfg["buffer_size"], obs_dim, self.cfg["sequence_length"]
        )

        self.epsilon = self.cfg["eps_start"]
        self.step_count = 0
        self._hidden = None

    def reset_hidden(self):
        if self.q_net.is_recurrent:
            self._hidden = self.q_net.init_hidden(1)
        else:
            self._hidden = None

    def select_action(self, obs: np.ndarray) -> int:
        if self.arch == "random" or np.random.random() < self.epsilon:
            return np.random.randint(self.act_dim)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values, new_hidden = self.q_net(obs_t, self._hidden)
            self._hidden = new_hidden
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        frac = min(1.0, self.step_count / self.cfg["eps_decay_steps"])
        self.epsilon = self.cfg["eps_start"] + frac * (
            self.cfg["eps_end"] - self.cfg["eps_start"]
        )

    def train_step(self) -> float:
        if self.arch == "random":
            return 0.0
        if self.buffer.size < self.cfg["batch_size"]:
            return 0.0

        if self.q_net.is_recurrent:
            batch = self.buffer.sample_sequences(
                self.cfg["batch_size"], device=self.device
            )
            q_vals, _ = self.q_net(batch["obs_seq"])
            with torch.no_grad():
                next_q, _ = self.target_net(batch["next_obs_seq"])
            actions = batch["actions_seq"][:, -1]
            rewards = batch["rewards_seq"][:, -1]
            dones = batch["dones_seq"][:, -1]
        else:
            batch = self.buffer.sample_single(
                self.cfg["batch_size"], device=self.device
            )
            q_vals, _ = self.q_net(batch["obs"])
            with torch.no_grad():
                next_q, _ = self.target_net(batch["next_obs"])
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]

        q_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            if self.q_net.is_recurrent:
                next_q_online, _ = self.q_net(batch["next_obs_seq"])
            else:
                next_q_online, _ = self.q_net(batch["next_obs"])
        best_actions = next_q_online.argmax(dim=1)
        next_q_value = next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target = rewards + self.cfg["gamma"] * (1 - dones) * next_q_value

        loss = nn.functional.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()

        # Enforce NCP weight constraints
        if hasattr(self.q_net, "ltc"):
            self.q_net.ltc.apply_weight_constraints()

        self.step_count += 1
        self.update_epsilon()

        if self.step_count % self.cfg["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())

    def measure_inference_latency(self, n_runs: int = 100) -> float:
        """Returns mean inference time in milliseconds."""
        obs = torch.randn(1, self.obs_dim).to(self.device)
        hidden = self.q_net.init_hidden(1)
        self.q_net.eval()
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                self.q_net(obs, hidden)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                self.q_net(obs, hidden)
            times.append((time.perf_counter() - start) * 1000)
        self.q_net.train()
        return np.mean(times)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
        self.epsilon = ckpt["epsilon"]
