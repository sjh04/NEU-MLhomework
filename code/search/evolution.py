"""Evolutionary search for optimal NCP wiring topology."""
import os
import json
import numpy as np

from search.genome import WiringGenome
from models.wiring import NCP
from models.dqn_agent import DQNAgent
from envs.env_factory import get_obs_dim, get_action_dim
from scripts.train import train as train_agent


class EvolutionarySearch:
    def __init__(self, config: dict):
        self.config = config
        self.scfg = config["search"]
        self.env_names = self.scfg["env_names"]
        self.rng = np.random.RandomState(config.get("seed", 42))

        self.population = []
        self.fitness_history = []
        self.best_genome = None
        self.best_fitness = -float("inf")

    def initialize_population(self):
        self.population = [
            WiringGenome.random(self.rng)
            for _ in range(self.scfg["population_size"])
        ]

    def evaluate_fitness(self, genome: WiringGenome) -> float:
        """Train a quick DQN with this genome and evaluate across environments."""
        total_reward = 0.0
        obs_dim = get_obs_dim()

        for env_name in self.env_names:
            act_dim = get_action_dim(env_name)
            wiring = NCP(
                inter_neurons=genome.inter_neurons,
                command_neurons=genome.command_neurons,
                motor_neurons=act_dim,
                sensory_fanout=genome.sensory_fanout,
                inter_fanout=genome.inter_fanout,
                recurrent_command_synapses=genome.recurrent_command_synapses,
                motor_fanin=genome.motor_fanin,
            )
            agent = DQNAgent(
                obs_dim, act_dim, "ncp", self.config,
                wiring=wiring, device=self.config.get("device", "cpu")
            )
            result = train_agent(
                "ncp", env_name, self.config, seed=self.rng.randint(10000),
                train_steps=self.scfg["fitness_train_steps"], agent=agent,
            )
            # Use mean of last 10 episode rewards as fitness
            ep_rewards = result["episode_rewards"]
            if len(ep_rewards) >= 10:
                total_reward += np.mean(ep_rewards[-10:])
            elif ep_rewards:
                total_reward += np.mean(ep_rewards)

        return total_reward / len(self.env_names)

    def tournament_select(self, fitnesses: list) -> int:
        candidates = self.rng.choice(
            len(fitnesses), size=self.scfg["tournament_size"], replace=False
        )
        best = candidates[np.argmax([fitnesses[c] for c in candidates])]
        return best

    def run(self) -> WiringGenome:
        """Run the full evolutionary search."""
        self.initialize_population()
        save_dir = os.path.join(self.config.get("save_dir", "results"), "search")
        os.makedirs(save_dir, exist_ok=True)

        for gen in range(self.scfg["generations"]):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.scfg['generations']}")
            print(f"{'='*60}")

            # Evaluate fitness
            fitnesses = []
            for i, genome in enumerate(self.population):
                print(f"  Evaluating individual {i + 1}/{len(self.population)}: {genome}")
                fitness = self.evaluate_fitness(genome)
                fitnesses.append(fitness)
                print(f"    Fitness: {fitness:.3f}")

            self.fitness_history.append({
                "gen": gen,
                "best": max(fitnesses),
                "mean": np.mean(fitnesses),
                "worst": min(fitnesses),
                "std": np.std(fitnesses),
            })

            # Track best
            sorted_idx = np.argsort(fitnesses)[::-1]
            if fitnesses[sorted_idx[0]] > self.best_fitness:
                self.best_fitness = fitnesses[sorted_idx[0]]
                self.best_genome = self.population[sorted_idx[0]]

            print(f"\n  Gen {gen + 1}: best={max(fitnesses):.3f}, "
                  f"mean={np.mean(fitnesses):.3f}, "
                  f"overall_best={self.best_fitness:.3f}")
            print(f"  Best genome: {self.best_genome}")

            # Elitism + reproduction
            new_pop = [self.population[i] for i in sorted_idx[:self.scfg["elite_count"]]]

            while len(new_pop) < self.scfg["population_size"]:
                p1 = self.population[self.tournament_select(fitnesses)]
                p2 = self.population[self.tournament_select(fitnesses)]
                if self.rng.random() < self.scfg["crossover_rate"]:
                    child = WiringGenome.crossover(p1, p2, self.rng)
                else:
                    child = WiringGenome(**p1.to_dict())
                child = child.mutate(self.rng, self.scfg["mutation_rate"])
                new_pop.append(child)

            self.population = new_pop

        # Save results
        with open(os.path.join(save_dir, "best_genome.json"), "w") as f:
            json.dump(self.best_genome.to_dict(), f, indent=2)
        with open(os.path.join(save_dir, "fitness_history.json"), "w") as f:
            json.dump(self.fitness_history, f, indent=2)

        print(f"\nSearch complete! Best genome: {self.best_genome}")
        print(f"Best fitness: {self.best_fitness:.3f}")

        return self.best_genome
