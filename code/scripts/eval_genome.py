"""
Standalone genome evaluation script. Called by parallel search.
Takes genome params from CLI, trains NCP on specified envs, prints fitness.
"""
import sys, os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.wiring import NCP
from models.dqn_agent import DQNAgent
from envs.env_factory import get_obs_dim, get_action_dim
from scripts.train import train as train_agent
from utils.common import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str, required=True, help="JSON dict")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--envs", type=str, default="highway-v0,roundabout-v0")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cuda")
    config.setdefault("save_dir", "results/search_tmp")
    config.setdefault("log_interval", 100000)  # suppress progress output

    genome = json.loads(args.genome)
    env_names = args.envs.split(",")
    steps = args.steps or config["search"]["fitness_train_steps"]

    set_seed(args.seed)
    obs_dim = get_obs_dim()

    total_reward = 0.0
    total_collision = 0.0
    for env_name in env_names:
        act_dim = get_action_dim(env_name)
        wiring = NCP(
            inter_neurons=genome["inter_neurons"],
            command_neurons=genome["command_neurons"],
            motor_neurons=act_dim,
            sensory_fanout=genome["sensory_fanout"],
            inter_fanout=genome["inter_fanout"],
            recurrent_command_synapses=genome["recurrent_command_synapses"],
            motor_fanin=genome["motor_fanin"],
        )
        agent = DQNAgent(obs_dim, act_dim, "ncp", config,
                         wiring=wiring, device=config["device"])
        result = train_agent("ncp", env_name, config, seed=args.seed,
                             train_steps=steps, agent=agent)
        ep_rewards = result["episode_rewards"]
        if len(ep_rewards) >= 10:
            total_reward += float(np.mean(ep_rewards[-10:]))
        elif ep_rewards:
            total_reward += float(np.mean(ep_rewards))
        # Use eval collision rate (from log.csv last row)
        import pandas as pd
        log_path = os.path.join(config["save_dir"],
                                f"ncp_{env_name}_s{args.seed}", "log.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            total_collision += float(df.iloc[-1].get("collision_rate", 1.0))

    fitness_reward = total_reward / len(env_names)
    fitness_collision = total_collision / len(env_names)

    # Safety-aware multi-objective fitness
    alpha = 10.0
    fitness = fitness_reward - alpha * fitness_collision

    # Print result as JSON to stdout
    print(json.dumps({
        "reward": fitness_reward,
        "collision_rate": fitness_collision,
        "fitness": fitness,
    }))


if __name__ == "__main__":
    main()
