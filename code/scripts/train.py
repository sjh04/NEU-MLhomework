"""
Training script for DQN agents with different Q-network architectures.

Usage:
    python scripts/train.py --arch ncp --env highway-v0 --seed 42
    python scripts/train.py --arch mlp --env highway-v0 --seed 42
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
from utils.common import set_seed, load_config, Logger


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def train(arch: str, env_name: str, config: dict, seed: int = 0,
          train_steps: int = None, agent: DQNAgent = None) -> dict:
    """Main training loop."""
    set_seed(seed)
    cfg = config["dqn"]
    total_steps = train_steps or cfg["train_steps"]
    log_interval = config.get("log_interval", 1000)

    obs_dim = get_obs_dim()
    act_dim = get_action_dim(env_name)

    if agent is None:
        agent = DQNAgent(obs_dim, act_dim, arch, config, device=config.get("device", "cpu"))

    param_count = agent.get_param_count()
    print(f"\n{'='*60}")
    print(f"  Training: {arch.upper()} on {env_name}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Device: {agent.device}")
    print(f"{'='*60}\n")

    log_dir = os.path.join(config.get("save_dir", "results"), f"{arch}_{env_name}_s{seed}")
    logger = Logger(log_dir)

    env = make_env(env_name, seed=seed)

    obs_raw, _ = env.reset(seed=seed)
    obs = get_obs(obs_raw)
    agent.reset_hidden()
    agent.buffer.mark_episode_start()

    episode_reward = 0.0
    episode_rewards = []
    losses = []
    episode_count = 0
    collisions = 0
    train_start = time.time()
    last_log_time = train_start

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_obs = get_obs(next_obs_raw)
        done = terminated or truncated

        agent.buffer.add(obs, action, reward, next_obs, done)
        episode_reward += reward

        loss = agent.train_step()
        if loss > 0:
            losses.append(loss)

        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            if info.get("crashed", False):
                collisions += 1

            obs_raw, _ = env.reset()
            obs = get_obs(obs_raw)
            agent.reset_hidden()
            agent.buffer.mark_episode_start()
            episode_reward = 0.0

        if step % log_interval == 0:
            elapsed = time.time() - train_start
            steps_per_sec = step / elapsed
            eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            progress = step / total_steps * 100

            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            recent_loss = np.mean(losses[-100:]) if losses else 0
            collision_rate = collisions / max(episode_count, 1)

            logger.log(step, {
                "mean_reward": np.mean(recent_rewards),
                "loss": recent_loss,
                "epsilon": agent.epsilon,
                "episodes": episode_count,
                "collision_rate": collision_rate,
            })

            # Progress bar
            bar_len = 30
            filled = int(bar_len * step / total_steps)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"  [{bar}] {progress:5.1f}% | "
                  f"Step {step:>6d}/{total_steps} | "
                  f"Ep {episode_count:>4d} | "
                  f"R={np.mean(recent_rewards):>7.2f} | "
                  f"Loss={recent_loss:.4f} | "
                  f"Eps={agent.epsilon:.3f} | "
                  f"Col={collision_rate:.1%} | "
                  f"Speed={steps_per_sec:.0f}it/s | "
                  f"ETA={format_time(eta)}", flush=True)

    env.close()
    logger.save()

    total_time = time.time() - train_start
    model_path = os.path.join(log_dir, "model.pt")
    agent.save(model_path)

    # Evaluation
    print(f"\n  Evaluating ({cfg['eval_episodes']} episodes)...", end=" ", flush=True)
    eval_start = time.time()
    eval_rewards = evaluate_quick(agent, env_name, n_episodes=cfg["eval_episodes"])
    eval_time = time.time() - eval_start

    latency = agent.measure_inference_latency()

    print(f"done ({format_time(eval_time)})")
    print(f"\n  {'─'*50}")
    print(f"  Results for {arch.upper()} on {env_name}:")
    print(f"    Eval reward:  {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"    Collision rate: {collisions/max(episode_count,1):.1%}")
    print(f"    Parameters:   {param_count:,}")
    print(f"    Latency:      {latency:.2f}ms")
    print(f"    Train time:   {format_time(total_time)}")
    print(f"    Model saved:  {model_path}")
    print(f"  {'─'*50}\n")

    return {
        "episode_rewards": episode_rewards,
        "losses": losses,
        "eval_rewards": eval_rewards,
        "param_count": param_count,
        "agent": agent,
    }


def evaluate_quick(agent: DQNAgent, env_name: str, n_episodes: int = 50) -> list:
    """Run evaluation episodes with greedy policy."""
    env = make_env(env_name, seed=9999)
    rewards = []
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    for ep in range(n_episodes):
        obs_raw, _ = env.reset(seed=9999 + ep)
        obs = get_obs(obs_raw)
        agent.reset_hidden()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                action = agent.select_action(obs)
            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs = get_obs(obs_raw)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    agent.epsilon = old_eps
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent")
    parser.add_argument("--arch", choices=["mlp", "lstm", "gru", "ncp", "fc_ltc", "random"], required=True)
    parser.add_argument("--env", default="highway-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cpu")
    config.setdefault("save_dir", "results")
    config.setdefault("log_interval", 1000)

    results = train(args.arch, args.env, config, seed=args.seed,
                    train_steps=args.steps)

    print(f"Final eval reward: {np.mean(results['eval_rewards']):.2f}")


if __name__ == "__main__":
    main()
