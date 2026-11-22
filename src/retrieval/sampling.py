"""Deterministic streaming reservoir sampling for evaluation strata.

Stratum seeds are derived with BLAKE2b so results do not depend on
PYTHONHASHSEED or process-local ``hash()``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar

import numpy as np

T = TypeVar("T")

# Inclusive upper bound for numpy Generator seeds (signed 32-bit positive).
_SEED_MOD = 2**31 - 1


def derive_stratum_seed(master_seed: int, stratum: str) -> int:
    """Stable per-stratum seed in ``[0, 2**31 - 2]`` from master seed + name."""
    payload = f"{int(master_seed)}\0{stratum}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MOD


@dataclass
class ReservoirSample(Generic[T]):
    """Algorithm R (Vitter): uniform sample of size k from a stream of unknown N."""

    k: int
    seed: int
    rng: np.random.Generator = field(init=False)
    reservoir: list[T] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed) % _SEED_MOD)
        if self.k < 0:
            raise ValueError("k must be >= 0")

    def consider(self, item: T) -> None:
        self.seen += 1
        if self.k == 0:
            return
        if len(self.reservoir) < self.k:
            self.reservoir.append(item)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.k:
            self.reservoir[j] = item

    def fill_from(self, items: Iterable[T]) -> list[T]:
        for item in items:
            self.consider(item)
        return list(self.reservoir)

    @property
    def population_size(self) -> int:
        return self.seen

    @property
    def inclusion_probability(self) -> float | None:
        if self.seen == 0 or self.k == 0:
            return None
        return min(1.0, self.k / self.seen)

    @property
    def sampling_weight(self) -> float | None:
        pi = self.inclusion_probability
        if pi is None or pi == 0:
            return None
        return 1.0 / pi


@dataclass
class StratifiedReservoirs(Generic[T]):
    """Independent reservoirs keyed by stratum (e.g. lang::primary_group)."""

    k_per_stratum: int
    seed: int
    _reservoirs: dict[str, ReservoirSample[T]] = field(default_factory=dict)

    def _get(self, stratum: str) -> ReservoirSample[T]:
        if stratum not in self._reservoirs:
            h = derive_stratum_seed(self.seed, stratum)
            self._reservoirs[stratum] = ReservoirSample(k=self.k_per_stratum, seed=h)
        return self._reservoirs[stratum]

    def consider(self, stratum: str, item: T) -> None:
        self._get(stratum).consider(item)

    def all_items(self) -> list[T]:
        out: list[T] = []
        # Deterministic traversal order of strata.
        for s in sorted(self._reservoirs.keys()):
            out.extend(self._reservoirs[s].reservoir)
        return out

    def meta(self) -> dict[str, dict]:
        return {
            s: {
                "population_size": r.population_size,
                "N_h": r.population_size,
                "k": r.k,
                "n_h": len(r.reservoir),
                "n_sampled": len(r.reservoir),
                "inclusion_probability": r.inclusion_probability,
                "sampling_weight": r.sampling_weight,
                "seed": r.seed,
            }
            for s, r in sorted(self._reservoirs.items())
        }

    @property
    def master_seed(self) -> int:
        return self.seed
