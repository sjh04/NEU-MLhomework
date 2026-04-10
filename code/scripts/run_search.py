"""
Run evolutionary search for optimal NCP wiring topology.

Usage:
    python scripts/run_search.py --config configs/default.yaml
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search.evolution import EvolutionarySearch
from utils.common import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="NCP Wiring Topology Search")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("device", "cpu")
    config.setdefault("save_dir", "results")
    config.setdefault("log_interval", 1000)

    set_seed(config.get("seed", 42))

    search = EvolutionarySearch(config)
    best_genome = search.run()

    print(f"\nBest genome found: {best_genome}")
    print(f"Genome dict: {best_genome.to_dict()}")


if __name__ == "__main__":
    main()
