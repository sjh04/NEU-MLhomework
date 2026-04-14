"""Parallel evolutionary search for optimal NCP wiring topology."""
import os
import json
import subprocess
import time
import numpy as np

from search.genome import WiringGenome


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

        # Parallel evaluation config
        self.max_parallel = self.scfg.get("max_parallel", 10)
        self.gpu_ids = self.scfg.get("gpu_ids", [0])  # list of GPUs to use

    def initialize_population(self):
        self.population = [
            WiringGenome.random(self.rng)
            for _ in range(self.scfg["population_size"])
        ]

    def _launch_eval(self, genome: WiringGenome, gpu_id: int, seed: int) -> subprocess.Popen:
        """Launch a subprocess to evaluate one genome."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            "python", "scripts/eval_genome.py",
            "--genome", json.dumps(genome.to_dict()),
            "--seed", str(seed),
            "--steps", str(self.scfg["fitness_train_steps"]),
            "--envs", ",".join(self.env_names),
        ]
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        return proc

    def _parse_result(self, proc: subprocess.Popen) -> dict:
        """Parse fitness from subprocess stdout."""
        stdout, stderr = proc.communicate()
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        print(f"[WARN] Failed to parse result. Stderr: {stderr[-500:]}")
        return {"reward": 0.0, "collision_rate": 1.0, "fitness": -10.0}

    def evaluate_population_parallel(self, population: list) -> list:
        """Evaluate all individuals in parallel across available GPUs."""
        results = [None] * len(population)
        running = {}  # proc -> (idx, start_time)

        i = 0
        while i < len(population) or running:
            # Launch new jobs up to max_parallel
            while i < len(population) and len(running) < self.max_parallel:
                gpu_id = self.gpu_ids[len(running) % len(self.gpu_ids)]
                seed = self.rng.randint(10000)
                proc = self._launch_eval(population[i], gpu_id, seed)
                running[proc] = (i, time.time(), population[i])
                print(f"  [Launch] Individual {i+1}/{len(population)} on GPU{gpu_id}: {population[i]}")
                i += 1

            # Wait for any to complete
            done_procs = [p for p in running if p.poll() is not None]
            if not done_procs:
                time.sleep(2)
                continue

            for proc in done_procs:
                idx, start_time, genome = running.pop(proc)
                elapsed = time.time() - start_time
                result = self._parse_result(proc)
                results[idx] = result
                print(f"  [Done  ] {idx+1}: reward={result['reward']:.2f}, "
                      f"col={result['collision_rate']*100:.0f}%, "
                      f"fit={result['fitness']:.2f} "
                      f"({elapsed:.0f}s)")

        return results

    def tournament_select(self, fitnesses: list) -> int:
        candidates = self.rng.choice(
            len(fitnesses), size=self.scfg["tournament_size"], replace=False
        )
        best = candidates[np.argmax([fitnesses[c] for c in candidates])]
        return best

    def run(self) -> WiringGenome:
        """Run the full evolutionary search with parallel evaluation."""
        self.initialize_population()
        save_dir = os.path.join(self.config.get("save_dir", "results"), "search")
        os.makedirs(save_dir, exist_ok=True)

        for gen in range(self.scfg["generations"]):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.scfg['generations']}")
            print(f"{'='*60}")

            gen_start = time.time()
            results = self.evaluate_population_parallel(self.population)
            fitnesses = [r["fitness"] for r in results]
            rewards = [r["reward"] for r in results]
            collisions = [r["collision_rate"] for r in results]

            self.fitness_history.append({
                "gen": gen,
                "best_fitness": float(max(fitnesses)),
                "mean_fitness": float(np.mean(fitnesses)),
                "worst_fitness": float(min(fitnesses)),
                "best_reward": float(max(rewards)),
                "mean_reward": float(np.mean(rewards)),
                "mean_collision": float(np.mean(collisions)),
                "population": [g.to_dict() for g in self.population],
                "fitnesses": fitnesses,
            })

            sorted_idx = np.argsort(fitnesses)[::-1]
            if fitnesses[sorted_idx[0]] > self.best_fitness:
                self.best_fitness = fitnesses[sorted_idx[0]]
                self.best_genome = self.population[sorted_idx[0]]

            gen_time = time.time() - gen_start
            print(f"\n  Gen {gen + 1} done ({gen_time:.0f}s): "
                  f"best={max(fitnesses):.2f}, "
                  f"mean={np.mean(fitnesses):.2f}, "
                  f"overall_best={self.best_fitness:.2f}")
            print(f"  Best genome so far: {self.best_genome}")

            # Save intermediate state after each generation
            with open(os.path.join(save_dir, "fitness_history.json"), "w") as f:
                json.dump(self.fitness_history, f, indent=2)
            with open(os.path.join(save_dir, "best_genome.json"), "w") as f:
                json.dump(self.best_genome.to_dict(), f, indent=2)

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

        print(f"\n{'='*60}")
        print(f"Search complete! Best genome: {self.best_genome}")
        print(f"Best fitness: {self.best_fitness:.3f}")
        print(f"{'='*60}")
        return self.best_genome
