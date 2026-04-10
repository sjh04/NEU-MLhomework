"""Visualization utilities for NCP experiments."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# ===================== Training Curves =====================

def plot_training_curves(results: dict, env_name: str, save_path: str):
    """Plot reward curves for multiple architectures on a single env.

    Args:
        results: {arch_name: {"episode_rewards": [...]}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"mlp": "#1f77b4", "lstm": "#ff7f0e", "gru": "#2ca02c", "ncp": "#d62728"}

    for arch, data in results.items():
        rewards = data["episode_rewards"]
        if len(rewards) < 2:
            continue
        # Smooth with running mean
        window = min(20, len(rewards) // 3 + 1)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=arch.upper(), color=colors.get(arch, None), linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(f"Training Curves - {env_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_bars(df, save_path: str):
    """Bar charts comparing metrics across architectures.

    Args:
        df: DataFrame with columns [arch, env, mean_reward, collision_rate, param_count, inference_latency_ms]
    """
    metrics = ["mean_reward", "collision_rate", "param_count", "inference_latency_ms"]
    labels = ["Mean Reward", "Collision Rate", "Parameter Count", "Latency (ms)"]

    envs = df["env"].unique()
    archs = df["arch"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        x = np.arange(len(envs))
        width = 0.8 / len(archs)

        for i, arch in enumerate(archs):
            vals = []
            for env in envs:
                subset = df[(df["arch"] == arch) & (df["env"] == env)]
                vals.append(subset[metric].mean() if len(subset) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=arch.upper(),
                   color=colors[i % len(colors)])

        ax.set_xlabel("Environment")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x + width * (len(archs) - 1) / 2)
        ax.set_xticklabels([e.replace("-v0", "") for e in envs], rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Architecture Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===================== NCP Topology =====================

def plot_ncp_topology(wiring, save_path: str, title: str = "NCP Topology"):
    """Visualize NCP wiring as a layered graph."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n_motor = wiring._num_motor_neurons
    n_cmd = wiring._num_command_neurons
    n_inter = wiring._num_inter_neurons
    n_sensory = wiring.input_dim

    # Compute positions for each layer
    def layer_positions(n, y, x_center=0.5):
        if n == 0:
            return []
        xs = np.linspace(0.1, 0.9, n) if n > 1 else [x_center]
        return [(x, y) for x in xs]

    pos_sensory = layer_positions(n_sensory, 4.0)
    pos_inter = layer_positions(n_inter, 3.0)
    pos_cmd = layer_positions(n_cmd, 2.0)
    pos_motor = layer_positions(n_motor, 1.0)

    # Draw nodes
    node_size = 200
    for i, (x, y) in enumerate(pos_sensory):
        ax.scatter(x, y, s=node_size, c="#4CAF50", zorder=5, edgecolors="black")
    for i, (x, y) in enumerate(pos_inter):
        ax.scatter(x, y, s=node_size, c="#2196F3", zorder=5, edgecolors="black")
    for i, (x, y) in enumerate(pos_cmd):
        ax.scatter(x, y, s=node_size, c="#FF9800", zorder=5, edgecolors="black")
    for i, (x, y) in enumerate(pos_motor):
        ax.scatter(x, y, s=node_size, c="#F44336", zorder=5, edgecolors="black")

    # Draw sensory connections
    adj_s = wiring.sensory_adjacency_matrix
    for src in range(n_sensory):
        for dest in range(wiring.units):
            if adj_s[src, dest] != 0:
                if dest < n_motor:
                    dst_pos = pos_motor[dest]
                elif dest < n_motor + n_cmd:
                    dst_pos = pos_cmd[dest - n_motor]
                else:
                    dst_pos = pos_inter[dest - n_motor - n_cmd]
                color = "#4CAF50" if adj_s[src, dest] > 0 else "#9C27B0"
                style = "-" if adj_s[src, dest] > 0 else "--"
                ax.annotate("", xy=dst_pos, xytext=pos_sensory[src],
                           arrowprops=dict(arrowstyle="->", color=color,
                                          linestyle=style, alpha=0.4, lw=0.8))

    # Draw internal connections
    adj = wiring.adjacency_matrix
    all_pos = list(pos_motor) + list(pos_cmd) + list(pos_inter)
    for src in range(wiring.units):
        for dest in range(wiring.units):
            if adj[src, dest] != 0:
                color = "#333333" if adj[src, dest] > 0 else "#9C27B0"
                style = "-" if adj[src, dest] > 0 else "--"
                ax.annotate("", xy=all_pos[dest], xytext=all_pos[src],
                           arrowprops=dict(arrowstyle="->", color=color,
                                          linestyle=style, alpha=0.5, lw=1.0))

    # Legend
    legend_elements = [
        mpatches.Patch(color="#4CAF50", label=f"Sensory ({n_sensory})"),
        mpatches.Patch(color="#2196F3", label=f"Inter ({n_inter})"),
        mpatches.Patch(color="#FF9800", label=f"Command ({n_cmd})"),
        mpatches.Patch(color="#F44336", label=f"Motor ({n_motor})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax.set_yticklabels(["Motor", "Command", "Inter", "Sensory"])
    ax.set_xticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===================== Interpretability =====================

def collect_episode_activations(agent, env_name: str):
    """Run one episode and record NCP activations at each step."""
    from envs.env_factory import make_env, get_obs

    env = make_env(env_name, seed=12345)
    obs_raw, _ = env.reset(seed=12345)
    obs = get_obs(obs_raw)
    agent.reset_hidden()

    data = {"command": [], "inter": [], "motor": [],
            "actions": [], "rewards": [], "obs": []}
    done = False

    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            act_info = agent.q_net.get_activations(obs_t, agent._hidden)
            _, agent._hidden = agent.q_net(obs_t, agent._hidden)

        action = act_info["output"].argmax(dim=1).item()
        data["command"].append(act_info["command"].squeeze(0).cpu().numpy())
        data["inter"].append(act_info["inter"].squeeze(0).cpu().numpy())
        data["motor"].append(act_info["motor"].squeeze(0).cpu().numpy())
        data["actions"].append(action)
        data["obs"].append(obs)

        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs = get_obs(obs_raw)
        data["rewards"].append(reward)
        done = terminated or truncated

    env.close()
    return {k: np.array(v) for k, v in data.items()}


def plot_command_activations(activations: dict, save_path: str,
                            title: str = "Command Neuron Activations"):
    """Heatmap of command neuron activations with action annotations."""
    action_names = ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 2]})

    # Command neuron heatmap
    cmd = activations["command"]  # (T, n_cmd)
    im = axes[0].imshow(cmd.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0].set_ylabel("Command\nNeuron")
    axes[0].set_title(title)
    plt.colorbar(im, ax=axes[0], fraction=0.02)

    # Action taken
    actions = activations["actions"]
    axes[1].step(range(len(actions)), actions, where="mid", color="black", linewidth=1.5)
    axes[1].set_ylabel("Action")
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(action_names, fontsize=8)
    axes[1].set_ylim(-0.5, 4.5)

    # Reward
    rewards = activations["rewards"]
    axes[2].bar(range(len(rewards)), rewards, color="#4CAF50", alpha=0.7)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Timestep")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===================== Search Convergence =====================

def plot_search_convergence(fitness_history: list, save_path: str):
    """Plot best/mean/worst fitness per generation."""
    gens = [h["gen"] for h in fitness_history]
    bests = [h["best"] for h in fitness_history]
    means = [h["mean"] for h in fitness_history]
    worsts = [h["worst"] for h in fitness_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, bests, "g-o", label="Best", markersize=4, linewidth=2)
    ax.plot(gens, means, "b-s", label="Mean", markersize=3, linewidth=1.5)
    ax.fill_between(gens, worsts, bests, alpha=0.15, color="blue")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (Mean Reward)", fontsize=12)
    ax.set_title("Evolutionary Search Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_topology_comparison(wirings: dict, save_path: str):
    """Side-by-side topology plots for multiple wirings.

    Args:
        wirings: {"Hand-designed": wiring, "Searched": wiring}
    """
    n = len(wirings)
    fig_width = 6 * n
    for i, (name, wiring) in enumerate(wirings.items()):
        path = save_path.replace(".png", f"_{name.lower().replace(' ', '_')}.png")
        plot_ncp_topology(wiring, path, title=name)
