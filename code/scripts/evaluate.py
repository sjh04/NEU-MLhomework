"""
Evaluation script: compare architectures across environments.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --arch ncp --env highway-v0 --checkpoint results/ncp_highway-v0_s42/model.pt
"""
import sys
import os
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.env_factory import make_env, get_obs, get_obs_dim, get_action_dim
from models.dqn_agent import DQNAgent
from utils.common import load_config, set_seed


def evaluate(agent: DQNAgent, env_name: str, n_episodes: int = 50) -> dict:
    """Full evaluation with detailed metrics."""
    env = make_env(env_name, seed=9999)
    rewards_list = []
    collisions = 0
    episode_lengths = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy evaluation

    for ep in range(n_episodes):
        obs_raw, _ = env.reset(seed=9999 + ep)
        obs = get_obs(obs_raw)
        agent.reset_hidden()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs = get_obs(obs_raw)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if info.get("crashed", False):
                collisions += 1

        rewards_list.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    agent.epsilon = old_epsilon

    return {
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "collision_rate": collisions / n_episodes,
        "mean_episode_length": np.mean(episode_lengths),
        "param_count": agent.get_param_count(),
        "inference_latency_ms": agent.measure_inference_latency(),
    }


def compare_all(config: dict, env_names: list = None, seeds: list = None):
    """Train and evaluate all architectures on all environments."""
    from scripts.train import train as train_agent

    if env_names is None:
        env_names = config["env"]["names"]
    if seeds is None:
        seeds = [0, 1, 2]

    archs = ["mlp", "lstm", "gru", "ncp"]
    all_results = []

    for env_name in env_names:
        for arch in archs:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Training {arch} on {env_name} (seed={seed})")
                print(f"{'='*60}")

                result = train_agent(arch, env_name, config, seed=seed)
                agent = result["agent"]
                metrics = evaluate(agent, env_name)
                metrics["arch"] = arch
                metrics["env"] = env_name
                metrics["seed"] = seed
                all_results.append(metrics)

                print(f"  Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
                print(f"  Collision rate: {metrics['collision_rate']:.2%}")
                print(f"  Params: {metrics['param_count']}")
                print(f"  Latency: {metrics['inference_latency_ms']:.2f}ms")

    # Save results
    import pandas as pd
    df = pd.DataFrame(all_results)
    save_path = os.path.join(config["save_dir"], "comparison_results.csv")
    os.makedirs(config["save_dir"], exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

    # Print summary table
    summary = df.groupby(["env", "arch"]).agg({
        "mean_reward": ["mean", "std"],
        "collision_rate": "mean",
        "param_count": "first",
        "inference_latency_ms": "mean",
    }).round(3)
    print(f"\n{summary}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agents")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--arch", default=None, help="Single arch to evaluate")
    parser.add_argument("--env", default=None, help="Single env to evaluate")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run full comparison across all archs/envs")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cpu")
    config.setdefault("save_dir", "results")
    config.setdefault("log_interval", 1000)

    if args.compare_all:
        compare_all(config)
    elif args.checkpoint and args.arch and args.env:
        obs_dim = get_obs_dim()
        act_dim = get_action_dim(args.env)
        agent = DQNAgent(obs_dim, act_dim, args.arch, config, device=config["device"])
        agent.load(args.checkpoint)
        metrics = evaluate(agent, args.env)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("Use --compare-all or specify --arch, --env, --checkpoint")


if __name__ == "__main__":
    main()
