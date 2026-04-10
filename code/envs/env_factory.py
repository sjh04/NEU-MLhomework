import gymnasium as gym
import numpy as np
import highway_env  # registers highway-env environments


# Unified observation config for all environments
_FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
_VEHICLES_COUNT = 5

_OBS_CONFIG = {
    "type": "Kinematics",
    "vehicles_count": _VEHICLES_COUNT,
    "features": _FEATURES,
    "normalize": True,
    "absolute": False,
    "order": "sorted",
}

# Safety reward config
_COLLISION_REWARD = -5.0  # heavy penalty for collision

# Per-environment overrides
ENV_CONFIGS = {
    "highway-v0": {
        "observation": _OBS_CONFIG,
        "action": {"type": "DiscreteMetaAction"},
        "duration": 40,
        "lanes_count": 3,
        "vehicles_count": 20,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "collision_reward": _COLLISION_REWARD,
    },
    "merge-v0": {
        "observation": _OBS_CONFIG,
        "action": {"type": "DiscreteMetaAction"},
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "collision_reward": _COLLISION_REWARD,
    },
    "intersection-v0": {
        "observation": _OBS_CONFIG,
        "action": {"type": "DiscreteMetaAction"},
        "duration": 13,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "initial_vehicle_count": 5,
        "collision_reward": _COLLISION_REWARD,
    },
    "roundabout-v0": {
        "observation": _OBS_CONFIG,
        "action": {"type": "DiscreteMetaAction"},
        "duration": 11,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "collision_reward": _COLLISION_REWARD,
    },
}

OBS_DIM = _VEHICLES_COUNT * len(_FEATURES)  # 5 * 7 = 35


def make_env(env_name: str, seed: int = 0) -> gym.Env:
    """Create a highway-env environment with unified observation format."""
    env = gym.make(env_name)
    config = ENV_CONFIGS.get(env_name, {})
    env.unwrapped.configure(config)
    env.reset(seed=seed)
    return env


def get_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten the (vehicles_count, features) observation to a 1D vector."""
    return obs.flatten().astype(np.float32)


def get_obs_dim() -> int:
    return OBS_DIM


def get_action_dim(env_name: str) -> int:
    """All environments use 5 discrete meta-actions after config override."""
    return 5
