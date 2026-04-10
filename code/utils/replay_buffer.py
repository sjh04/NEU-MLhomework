import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer supporting single-step and sequence sampling."""

    def __init__(self, capacity: int, obs_dim: int, sequence_length: int = 8):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.sequence_length = sequence_length

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # Track which timestep within an episode each transition belongs to
        self.episode_ids = np.zeros(capacity, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self._episode_counter = 0

    def mark_episode_start(self):
        self._episode_counter += 1

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.episode_ids[self.pos] = self._episode_counter
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_single(self, batch_size: int, device="cpu") -> dict:
        """Sample individual transitions (for MLP)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[indices]).to(device),
            "actions": torch.LongTensor(self.actions[indices]).to(device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(device),
            "next_obs": torch.FloatTensor(self.next_obs[indices]).to(device),
            "dones": torch.FloatTensor(self.dones[indices]).to(device),
        }

    def sample_sequences(self, batch_size: int, seq_len: int = None,
                         device="cpu") -> dict:
        """Sample contiguous sequences within episodes (for RNN/NCP)."""
        seq_len = seq_len or self.sequence_length

        # Build valid start indices: sequences that don't cross episode boundaries
        valid_starts = []
        for i in range(self.size - seq_len + 1):
            indices = [(i + t) % self.capacity for t in range(seq_len)]
            ep_ids = self.episode_ids[indices]
            if np.all(ep_ids == ep_ids[0]):
                valid_starts.append(i)

        if len(valid_starts) < batch_size:
            # Fall back to single-step sampling reshaped as length-1 sequences
            batch = self.sample_single(batch_size, device)
            return {
                "obs_seq": batch["obs"].unsqueeze(1),
                "actions_seq": batch["actions"].unsqueeze(1),
                "rewards_seq": batch["rewards"].unsqueeze(1),
                "next_obs_seq": batch["next_obs"].unsqueeze(1),
                "dones_seq": batch["dones"].unsqueeze(1),
            }

        chosen = np.random.choice(valid_starts, size=batch_size, replace=True)

        obs_seq = np.zeros((batch_size, seq_len, self.obs_dim), dtype=np.float32)
        act_seq = np.zeros((batch_size, seq_len), dtype=np.int64)
        rew_seq = np.zeros((batch_size, seq_len), dtype=np.float32)
        next_obs_seq = np.zeros((batch_size, seq_len, self.obs_dim), dtype=np.float32)
        done_seq = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b, start in enumerate(chosen):
            for t in range(seq_len):
                idx = (start + t) % self.capacity
                obs_seq[b, t] = self.obs[idx]
                act_seq[b, t] = self.actions[idx]
                rew_seq[b, t] = self.rewards[idx]
                next_obs_seq[b, t] = self.next_obs[idx]
                done_seq[b, t] = self.dones[idx]

        return {
            "obs_seq": torch.FloatTensor(obs_seq).to(device),
            "actions_seq": torch.LongTensor(act_seq).to(device),
            "rewards_seq": torch.FloatTensor(rew_seq).to(device),
            "next_obs_seq": torch.FloatTensor(next_obs_seq).to(device),
            "dones_seq": torch.FloatTensor(done_seq).to(device),
        }
