"""
ODE unfolds ablation - clean version without monkey-patching.
Usage: python scripts/run_ablation_ode.py --unfolds 3 --seed 42
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np

from models.wiring import NCP
from models.ltc_cell import LTCCell
from models.q_networks import QNetworkBase
from envs.env_factory import make_env, get_obs, get_obs_dim, get_action_dim
from utils.replay_buffer import ReplayBuffer
from utils.common import load_config, set_seed, Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfolds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", default="highway-v0")
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    obs_dim = get_obs_dim()
    act_dim = get_action_dim(args.env)
    cfg = config["dqn"]

    # Build NCP with specific ode_unfolds - NO monkey-patching
    def make_ncp():
        wiring = NCP(
            inter_neurons=12, command_neurons=8, motor_neurons=act_dim,
            sensory_fanout=4, inter_fanout=4,
            recurrent_command_synapses=4, motor_fanin=4,
        )
        ltc = LTCCell(wiring, in_features=obs_dim, ode_unfolds=args.unfolds)
        print(f"  Created LTCCell with ode_unfolds={ltc._ode_unfolds}")

        class NCPAblation(nn.Module):
            def __init__(self):
                super().__init__()
                self.wiring = wiring
                self.ltc = ltc
                s = wiring.units
                self.q_head = nn.Sequential(
                    nn.Linear(s, max(32, s)), nn.ReLU(), nn.Linear(max(32, s), act_dim)
                )

            def forward(self, obs, hidden=None):
                if hidden is None:
                    hidden = torch.zeros(obs.size(0), self.ltc.state_size, device=obs.device)
                if obs.dim() == 3:
                    for t in range(obs.size(1)):
                        _, hidden = self.ltc(obs[:, t, :], hidden)
                else:
                    _, hidden = self.ltc(obs, hidden)
                return self.q_head(hidden), hidden

        return NCPAblation()

    q_net = make_ncp().to(device)
    target_net = make_ncp().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, cfg["sequence_length"])

    # Output directory
    out_dir = f"results/ablation_ode{args.unfolds}/ncp_{args.env}_s{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(out_dir)

    # Training loop
    env = make_env(args.env, seed=args.seed)
    obs_raw, _ = env.reset(seed=args.seed)
    obs = get_obs(obs_raw)
    hidden = None
    buffer.mark_episode_start()

    epsilon = cfg["eps_start"]
    episode_reward = 0.0
    episode_rewards = []
    episode_count = 0
    collisions = 0

    print(f"\nTraining: ode_unfolds={args.unfolds}, env={args.env}, seed={args.seed}")
    print(f"  Params: {sum(p.numel() for p in q_net.parameters())}")

    for step in range(1, args.steps + 1):
        # Action selection
        if np.random.random() < epsilon:
            action = np.random.randint(act_dim)
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q, hidden = q_net(obs_t, hidden)
                action = q.argmax(dim=1).item()

        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        next_obs = get_obs(next_obs_raw)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1
            if info.get("crashed", False):
                collisions += 1
            obs_raw, _ = env.reset()
            obs = get_obs(obs_raw)
            hidden = None
            buffer.mark_episode_start()
            episode_reward = 0.0

        # Train
        if buffer.size >= cfg["batch_size"]:
            batch = buffer.sample_sequences(cfg["batch_size"], device=device)
            q_vals, _ = q_net(batch["obs_seq"])
            with torch.no_grad():
                next_q_target, _ = target_net(batch["next_obs_seq"])
                next_q_online, _ = q_net(batch["next_obs_seq"])
            best_actions = next_q_online.argmax(dim=1)
            next_q_value = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            actions = batch["actions_seq"][:, -1]
            rewards = batch["rewards_seq"][:, -1]
            dones = batch["dones_seq"][:, -1]

            q_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
            target = rewards + cfg["gamma"] * (1 - dones) * next_q_value
            loss = nn.functional.smooth_l1_loss(q_taken, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), cfg["grad_clip"])
            optimizer.step()
            q_net.ltc.apply_weight_constraints()

        # Epsilon decay
        frac = min(1.0, step / cfg["eps_decay_steps"])
        epsilon = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        # Target update
        if step % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Log
        if step % 5000 == 0:
            recent = episode_rewards[-10:] if episode_rewards else [0]
            col_rate = collisions / max(episode_count, 1)
            logger.log(step, {
                "mean_reward": np.mean(recent),
                "collision_rate": col_rate,
                "epsilon": epsilon,
                "episodes": episode_count,
            })
            print(f"  Step {step}/{args.steps} | R={np.mean(recent):.1f} | Col={col_rate:.1%} | Eps={epsilon:.3f}")

    env.close()
    logger.save()
    torch.save({"q_net": q_net.state_dict()}, os.path.join(out_dir, "model.pt"))
    print(f"\nDone: ode_unfolds={args.unfolds}, R={np.mean(episode_rewards[-10:]):.2f}")


if __name__ == "__main__":
    main()
