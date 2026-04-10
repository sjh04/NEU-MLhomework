"""NCP Wiring Genome for evolutionary search."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, ClassVar


@dataclass
class WiringGenome:
    """Searchable NCP wiring configuration."""
    inter_neurons: int = 12
    command_neurons: int = 8
    sensory_fanout: int = 4
    inter_fanout: int = 4
    recurrent_command_synapses: int = 4
    motor_fanin: int = 4

    BOUNDS: ClassVar[Dict[str, Tuple[int, int]]] = {
        "inter_neurons": (2, 20),
        "command_neurons": (2, 16),
        "sensory_fanout": (1, 8),
        "inter_fanout": (1, 8),
        "recurrent_command_synapses": (0, 8),
        "motor_fanin": (1, 8),
    }

    @staticmethod
    def random(rng: np.random.RandomState) -> "WiringGenome":
        params = {}
        for name, (lo, hi) in WiringGenome.BOUNDS.items():
            params[name] = rng.randint(lo, hi + 1)
        genome = WiringGenome(**params)
        genome.repair()
        return genome

    def mutate(self, rng: np.random.RandomState, mutation_rate: float = 0.3) -> "WiringGenome":
        params = self.to_dict()
        for name, (lo, hi) in self.BOUNDS.items():
            if rng.random() < mutation_rate:
                # Gaussian perturbation, rounded to int
                delta = int(round(rng.normal(0, max(1, (hi - lo) / 4))))
                params[name] = int(np.clip(params[name] + delta, lo, hi))
        child = WiringGenome(**params)
        child.repair()
        return child

    @staticmethod
    def crossover(p1: "WiringGenome", p2: "WiringGenome",
                  rng: np.random.RandomState) -> "WiringGenome":
        params = {}
        d1, d2 = p1.to_dict(), p2.to_dict()
        for name in WiringGenome.BOUNDS:
            params[name] = d1[name] if rng.random() < 0.5 else d2[name]
        child = WiringGenome(**params)
        child.repair()
        return child

    def repair(self):
        """Clamp fanouts to satisfy structural constraints."""
        for name, (lo, hi) in self.BOUNDS.items():
            val = getattr(self, name)
            setattr(self, name, int(np.clip(val, lo, hi)))
        self.sensory_fanout = min(self.sensory_fanout, self.inter_neurons)
        self.inter_fanout = min(self.inter_fanout, self.command_neurons)
        self.motor_fanin = min(self.motor_fanin, self.command_neurons)

    def is_valid(self) -> bool:
        return (self.sensory_fanout <= self.inter_neurons and
                self.inter_fanout <= self.command_neurons and
                self.motor_fanin <= self.command_neurons)

    def total_neurons(self) -> int:
        return self.inter_neurons + self.command_neurons

    def to_dict(self) -> dict:
        return {
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons,
            "sensory_fanout": self.sensory_fanout,
            "inter_fanout": self.inter_fanout,
            "recurrent_command_synapses": self.recurrent_command_synapses,
            "motor_fanin": self.motor_fanin,
        }

    def __repr__(self):
        return (f"WiringGenome(inter={self.inter_neurons}, cmd={self.command_neurons}, "
                f"sf={self.sensory_fanout}, if={self.inter_fanout}, "
                f"rcs={self.recurrent_command_synapses}, mf={self.motor_fanin})")
