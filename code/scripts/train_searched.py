"""
Train NCP with the best wiring genome found by evolutionary search.
Results saved as arch "ncp" — will be renamed by the batch script.
"""
import sys, os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.wiring import NCP
from models.dqn_agent import DQNAgent
from envs.env_factory import get_obs_dim, get_action_dim
from scripts.train import train as train_agent
from utils.common import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--genome", default="results/search/best_genome.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cuda")
    config.setdefault("save_dir", "results")
    config.setdefault("log_interval", 1000)

    with open(args.genome) as f:
        genome = json.load(f)
    print(f"Loaded genome: {genome}")

    set_seed(args.seed)
    obs_dim = get_obs_dim()
    act_dim = get_action_dim(args.env)

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

    train_agent("ncp", args.env, config, seed=args.seed,
                train_steps=args.steps, agent=agent)


if __name__ == "__main__":
    main()
