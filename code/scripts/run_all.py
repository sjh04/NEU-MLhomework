"""
Master script: runs the entire experiment pipeline.

Usage:
    python scripts/run_all.py --config configs/default.yaml
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.common import load_config, set_seed
from scripts.train import train
from scripts.evaluate import evaluate
from models.wiring import NCP
from models.dqn_agent import DQNAgent
from envs.env_factory import get_obs_dim, get_action_dim
from utils.visualize import (
    plot_training_curves, plot_comparison_bars,
    plot_ncp_topology, collect_episode_activations,
    plot_command_activations, plot_search_convergence,
    plot_topology_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip evolutionary search (Part B)")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cpu")
    config.setdefault("save_dir", "results")
    config.setdefault("log_interval", 1000)
    set_seed(config.get("seed", 42))

    fig_dir = os.path.join(config["save_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ==================== Part A: Train & Compare ====================
    print("\n" + "=" * 70)
    print("PART A: Training all architectures")
    print("=" * 70)

    archs = ["mlp", "lstm", "gru", "ncp"]
    env_names = config["env"]["names"]
    all_metrics = []

    for env_name in env_names:
        env_results = {}
        for arch in archs:
            print(f"\n--- Training {arch} on {env_name} ---")
            result = train(arch, env_name, config, seed=config.get("seed", 42))
            env_results[arch] = result

            metrics = evaluate(result["agent"], env_name,
                              n_episodes=config["dqn"]["eval_episodes"])
            metrics["arch"] = arch
            metrics["env"] = env_name
            all_metrics.append(metrics)

        # Plot training curves per env
        plot_training_curves(
            env_results, env_name,
            os.path.join(fig_dir, f"training_{env_name.replace('-', '_')}.png")
        )

    # Plot comparison
    import pandas as pd
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(config["save_dir"], "comparison_results.csv"), index=False)
    plot_comparison_bars(df, os.path.join(fig_dir, "comparison_bars.png"))

    print("\nPart A complete. Results saved.")

    # ==================== Part B: Evolutionary Search ====================
    if not args.skip_search:
        print("\n" + "=" * 70)
        print("PART B: Evolutionary Search")
        print("=" * 70)

        from search.evolution import EvolutionarySearch
        search = EvolutionarySearch(config)
        best_genome = search.run()

        # Plot search convergence
        plot_search_convergence(
            search.fitness_history,
            os.path.join(fig_dir, "search_convergence.png")
        )

        # Visualize searched topology
        act_dim = get_action_dim("highway-v0")
        searched_wiring = NCP(
            inter_neurons=best_genome.inter_neurons,
            command_neurons=best_genome.command_neurons,
            motor_neurons=act_dim,
            sensory_fanout=best_genome.sensory_fanout,
            inter_fanout=best_genome.inter_fanout,
            recurrent_command_synapses=best_genome.recurrent_command_synapses,
            motor_fanin=best_genome.motor_fanin,
        )
        searched_wiring.build((None, get_obs_dim()))

        hand_wiring = NCP(
            inter_neurons=config["ncp"]["inter_neurons"],
            command_neurons=config["ncp"]["command_neurons"],
            motor_neurons=act_dim,
            sensory_fanout=config["ncp"]["sensory_fanout"],
            inter_fanout=config["ncp"]["inter_fanout"],
            recurrent_command_synapses=config["ncp"]["recurrent_command_synapses"],
            motor_fanin=config["ncp"]["motor_fanin"],
        )
        hand_wiring.build((None, get_obs_dim()))

        plot_topology_comparison(
            {"Hand-designed": hand_wiring, "Searched": searched_wiring},
            os.path.join(fig_dir, "topology.png")
        )

        print("\nPart B complete.")

    # ==================== Part C: Interpretability ====================
    print("\n" + "=" * 70)
    print("PART C: Interpretability Analysis")
    print("=" * 70)

    # Use the NCP agent from Part A (highway-v0)
    ncp_result = train("ncp", "highway-v0", config, seed=config.get("seed", 42))
    ncp_agent = ncp_result["agent"]

    # Visualize default NCP topology
    wiring = ncp_agent.q_net.wiring
    plot_ncp_topology(wiring, os.path.join(fig_dir, "ncp_topology_default.png"),
                     title="Default NCP Topology")

    # Collect and plot activations for each env
    for env_name in env_names:
        print(f"  Collecting activations for {env_name}...")
        activations = collect_episode_activations(ncp_agent, env_name)
        plot_command_activations(
            activations,
            os.path.join(fig_dir, f"activations_{env_name.replace('-', '_')}.png"),
            title=f"Command Neuron Activations - {env_name}"
        )

    print("\nPart C complete.")
    print(f"\nAll figures saved to {fig_dir}/")
    print("Experiment pipeline finished!")


if __name__ == "__main__":
    main()
