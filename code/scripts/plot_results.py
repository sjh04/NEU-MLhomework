"""
Generate all Part A & D visualization figures from experiment results.

Usage:
    python scripts/plot_results.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# Config
# ============================================================
RESULTS_DIR = "results"
FIG_DIR = "results/figures"
IMG_DIR = "../img"  # for paper
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

ENVS = ["highway-v0", "merge-v0", "intersection-v0", "roundabout-v0"]
ENV_SHORT = {"highway-v0": "Highway", "merge-v0": "Merge",
             "intersection-v0": "Intersection", "roundabout-v0": "Roundabout"}
ARCHS = ["random", "mlp", "lstm", "gru", "fc_ltc", "ncp"]
ARCH_LABELS = {"random": "Random", "mlp": "MLP", "lstm": "LSTM", "gru": "GRU",
               "fc_ltc": "FC-LTC", "ncp": "NCP (Ours)"}
ARCH_COLORS = {"random": "#999999", "mlp": "#1f77b4", "lstm": "#ff7f0e",
               "gru": "#2ca02c", "fc_ltc": "#9467bd", "ncp": "#d62728"}
SEEDS = [42, 0, 123]
PARAMS = {"random": 1, "mlp": 13189, "lstm": 26181, "gru": 19717,
          "fc_ltc": 7152, "ncp": 7152}


def load_all_logs():
    """Load all log.csv files into a single DataFrame."""
    rows = []
    for env in ENVS:
        for seed in SEEDS:
            for arch in ARCHS:
                path = os.path.join(RESULTS_DIR, f"{arch}_{env}_s{seed}", "log.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                df["arch"] = arch
                df["env"] = env
                df["seed"] = seed
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def savefig(fig, name):
    for d in [FIG_DIR, IMG_DIR]:
        fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Figure 1: Training curves (reward vs step) per env
# ============================================================
def plot_training_curves(data):
    print("Plotting training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        for arch in ["mlp", "lstm", "gru", "fc_ltc", "ncp"]:
            sub = data[(data["env"] == env) & (data["arch"] == arch)]
            if sub.empty:
                continue
            # Group by step, compute mean/std across seeds
            grouped = sub.groupby("step")["mean_reward"].agg(["mean", "std"]).reset_index()
            ax.plot(grouped["step"], grouped["mean"], label=ARCH_LABELS[arch],
                    color=ARCH_COLORS[arch], linewidth=2)
            ax.fill_between(grouped["step"],
                            grouped["mean"] - grouped["std"],
                            grouped["mean"] + grouped["std"],
                            color=ARCH_COLORS[arch], alpha=0.15)
        ax.set_title(ENV_SHORT[env], fontsize=13)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(5))

    fig.suptitle("Training Curves (3 seeds, mean ± std)", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig(fig, "fig_training_curves.png")


# ============================================================
# Figure 2: Bar chart - final reward comparison
# ============================================================
def plot_reward_bars(data):
    print("Plotting reward comparison bars...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    archs_plot = ["mlp", "lstm", "gru", "fc_ltc", "ncp"]

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        means, stds = [], []
        for arch in archs_plot:
            sub = data[(data["env"] == env) & (data["arch"] == arch)]
            last_per_seed = sub.groupby("seed")["mean_reward"].last()
            means.append(last_per_seed.mean())
            stds.append(last_per_seed.std())

        x = np.arange(len(archs_plot))
        colors = [ARCH_COLORS[a] for a in archs_plot]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(ENV_SHORT[env], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in archs_plot], rotation=25, fontsize=9)
        ax.set_ylabel("Eval Reward" if idx == 0 else "")
        ax.grid(True, alpha=0.3, axis="y")

        # Highlight NCP bar
        bars[-1].set_edgecolor("red")
        bars[-1].set_linewidth(2)

    fig.suptitle("Final Reward Comparison (3 seeds, mean ± std)", fontsize=14, y=1.02)
    plt.tight_layout()
    savefig(fig, "fig_reward_comparison.png")


# ============================================================
# Figure 3: Bar chart - collision rate comparison
# ============================================================
def plot_collision_bars(data):
    print("Plotting collision rate bars...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    archs_plot = ["mlp", "lstm", "gru", "fc_ltc", "ncp"]

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        means, stds = [], []
        for arch in archs_plot:
            sub = data[(data["env"] == env) & (data["arch"] == arch)]
            last_per_seed = sub.groupby("seed")["collision_rate"].last()
            means.append(last_per_seed.mean() * 100)
            stds.append(last_per_seed.std() * 100)

        x = np.arange(len(archs_plot))
        colors = [ARCH_COLORS[a] for a in archs_plot]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(ENV_SHORT[env], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in archs_plot], rotation=25, fontsize=9)
        ax.set_ylabel("Collision Rate (%)" if idx == 0 else "")
        ax.grid(True, alpha=0.3, axis="y")

        bars[-1].set_edgecolor("red")
        bars[-1].set_linewidth(2)

    fig.suptitle("Collision Rate Comparison (3 seeds, mean ± std)", fontsize=14, y=1.02)
    plt.tight_layout()
    savefig(fig, "fig_collision_rate.png")


# ============================================================
# Figure 4: Pareto front - params vs reward
# ============================================================
def plot_pareto(data):
    print("Plotting Pareto front...")
    fig, ax = plt.subplots(figsize=(9, 6))
    archs_plot = ["mlp", "lstm", "gru", "fc_ltc", "ncp"]

    for arch in archs_plot:
        sub = data[(data["env"] == "highway-v0") & (data["arch"] == arch)]
        last_per_seed = sub.groupby("seed")["mean_reward"].last()
        reward_mean = last_per_seed.mean()
        reward_std = last_per_seed.std()
        params = PARAMS[arch]

        ax.errorbar(params, reward_mean, yerr=reward_std, fmt="o", markersize=12,
                     color=ARCH_COLORS[arch], capsize=5, linewidth=2,
                     label=f"{ARCH_LABELS[arch]} ({params:,})")

    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Mean Reward (highway-v0)", fontsize=12)
    ax.set_title("Parameter Efficiency: Pareto Front", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Draw Pareto-optimal line for NCP
    ax.annotate("Best efficiency", xy=(PARAMS["ncp"], 34.94),
                xytext=(PARAMS["ncp"] * 2.5, 30),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=11, color="red", fontweight="bold")

    plt.tight_layout()
    savefig(fig, "fig_pareto_front.png")


# ============================================================
# Figure 5: NCP topology visualization
# ============================================================
def plot_ncp_topology():
    print("Plotting NCP topology...")
    from models.wiring import NCP
    from utils.visualize import plot_ncp_topology as _plot_topo

    wiring = NCP(inter_neurons=12, command_neurons=8, motor_neurons=5,
                 sensory_fanout=4, inter_fanout=4,
                 recurrent_command_synapses=4, motor_fanin=4)
    wiring.build((None, 35))

    for d in [FIG_DIR, IMG_DIR]:
        _plot_topo(wiring, os.path.join(d, "fig_ncp_topology.png"),
                   title="NCP Wiring Topology (Hand-designed)")
    print("  Saved fig_ncp_topology.png")


# ============================================================
# Figure 6: Parameter efficiency table (for paper)
# ============================================================
def plot_param_table(data):
    print("Plotting parameter efficiency table...")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    archs_plot = ["random", "mlp", "lstm", "gru", "fc_ltc", "ncp"]
    headers = ["Model", "Params", "Highway\nReward", "Merge\nReward",
               "Highway\nCol.Rate", "Reward/\n1K Params"]
    rows = []
    for arch in archs_plot:
        row = [ARCH_LABELS[arch], f"{PARAMS[arch]:,}"]
        for env in ["highway-v0", "merge-v0"]:
            sub = data[(data["env"] == env) & (data["arch"] == arch)]
            last = sub.groupby("seed")["mean_reward"].last()
            row.append(f"{last.mean():.1f}±{last.std():.1f}")
        # Highway collision rate
        sub = data[(data["env"] == "highway-v0") & (data["arch"] == arch)]
        last_col = sub.groupby("seed")["collision_rate"].last()
        row.append(f"{last_col.mean()*100:.1f}%")
        # Efficiency
        sub = data[(data["env"] == "highway-v0") & (data["arch"] == arch)]
        last_r = sub.groupby("seed")["mean_reward"].last()
        eff = last_r.mean() / (PARAMS[arch] / 1000) if PARAMS[arch] > 1 else "-"
        row.append(f"{eff:.2f}" if isinstance(eff, float) else eff)
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#f0f0f0"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Bold NCP row
    for j in range(len(headers)):
        table[len(rows), j].set_text_props(fontweight="bold")

    plt.tight_layout()
    savefig(fig, "fig_summary_table.png")


# ============================================================
# Figure 7: Search convergence
# ============================================================
def plot_search_convergence():
    print("Plotting search convergence...")
    history_path = os.path.join(RESULTS_DIR, "search", "fitness_history.json")
    if not os.path.exists(history_path):
        print("  [SKIP] No search history found")
        return

    import json
    with open(history_path) as f:
        history = json.load(f)

    gens = [h["gen"] + 1 for h in history]
    bests = [h["best_fitness"] for h in history]
    means = [h["mean_fitness"] for h in history]
    worsts = [h["worst_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, bests, "g-o", label="Best", markersize=5, linewidth=2)
    ax.plot(gens, means, "b-s", label="Mean", markersize=4, linewidth=1.5)
    ax.fill_between(gens, worsts, bests, alpha=0.12, color="blue")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (reward - 10×collision)", fontsize=12)
    ax.set_title("Evolutionary Search Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    savefig(fig, "fig_search_convergence.png")


# ============================================================
# Figure 8: Searched NCP topology
# ============================================================
def plot_searched_topology():
    print("Plotting searched NCP topology...")
    import json
    genome_path = os.path.join(RESULTS_DIR, "search", "best_genome.json")
    if not os.path.exists(genome_path):
        print("  [SKIP] No best genome found")
        return

    from models.wiring import NCP
    from utils.visualize import plot_ncp_topology as _plot_topo

    with open(genome_path) as f:
        genome = json.load(f)

    wiring = NCP(
        inter_neurons=genome["inter_neurons"],
        command_neurons=genome["command_neurons"],
        motor_neurons=5,
        sensory_fanout=genome["sensory_fanout"],
        inter_fanout=genome["inter_fanout"],
        recurrent_command_synapses=genome["recurrent_command_synapses"],
        motor_fanin=genome["motor_fanin"],
    )
    wiring.build((None, 35))

    title = (f"Searched NCP Topology\n"
             f"(inter={genome['inter_neurons']}, cmd={genome['command_neurons']}, "
             f"sf={genome['sensory_fanout']}, if={genome['inter_fanout']}, "
             f"rcs={genome['recurrent_command_synapses']}, mf={genome['motor_fanin']})")

    for d in [FIG_DIR, IMG_DIR]:
        _plot_topo(wiring, os.path.join(d, "fig_searched_topology.png"), title=title)
    print("  Saved fig_searched_topology.png")


# ============================================================
# Figure 9: NCP-Hand vs NCP-Searched comparison
# ============================================================
def plot_hand_vs_searched():
    print("Plotting hand vs searched comparison...")
    envs = ["highway-v0", "merge-v0", "intersection-v0", "roundabout-v0"]
    seeds = [42, 0, 123]

    hand_rewards, searched_rewards = {}, {}
    hand_cols, searched_cols = {}, {}

    for env in envs:
        hr, sr, hc, sc = [], [], [], []
        for seed in seeds:
            for prefix, rlist, clist in [("ncp", hr, hc), ("ncp_searched", sr, sc)]:
                path = os.path.join(RESULTS_DIR, f"{prefix}_{env}_s{seed}", "log.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    last = df.iloc[-1]
                    rlist.append(last["mean_reward"])
                    clist.append(last.get("collision_rate", 1.0))
        hand_rewards[env] = hr
        searched_rewards[env] = sr
        hand_cols[env] = hc
        searched_cols[env] = sc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward comparison
    ax = axes[0]
    x = np.arange(len(envs))
    w = 0.35
    h_means = [np.mean(hand_rewards[e]) for e in envs]
    h_stds = [np.std(hand_rewards[e]) for e in envs]
    s_means = [np.mean(searched_rewards[e]) for e in envs]
    s_stds = [np.std(searched_rewards[e]) for e in envs]

    ax.bar(x - w/2, h_means, w, yerr=h_stds, capsize=4, label="NCP-Hand (25 neurons)",
           color="#d62728", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, s_means, w, yerr=s_stds, capsize=4, label="NCP-Searched (15 neurons)",
           color="#ff9896", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT[e] for e in envs])
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward: Hand-designed vs Searched")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Collision rate comparison
    ax = axes[1]
    h_col_means = [np.mean(hand_cols[e]) * 100 for e in envs]
    h_col_stds = [np.std(hand_cols[e]) * 100 for e in envs]
    s_col_means = [np.mean(searched_cols[e]) * 100 for e in envs]
    s_col_stds = [np.std(searched_cols[e]) * 100 for e in envs]

    ax.bar(x - w/2, h_col_means, w, yerr=h_col_stds, capsize=4, label="NCP-Hand",
           color="#d62728", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, s_col_means, w, yerr=s_col_stds, capsize=4, label="NCP-Searched",
           color="#ff9896", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT[e] for e in envs])
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collision Rate: Hand-designed vs Searched")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savefig(fig, "fig_hand_vs_searched.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    os.chdir("/mnt/aisdata/sjh04/LFM/NEU-MLhomework/code")
    data = load_all_logs()
    print(f"Loaded {len(data)} log entries from {data[['arch','env','seed']].drop_duplicates().shape[0]} experiments\n")

    plot_training_curves(data)
    plot_reward_bars(data)
    plot_collision_bars(data)
    plot_pareto(data)
    plot_ncp_topology()
    plot_param_table(data)
    plot_search_convergence()
    plot_searched_topology()
    plot_hand_vs_searched()

    print(f"\nAll figures saved to {FIG_DIR}/ and {IMG_DIR}/")
