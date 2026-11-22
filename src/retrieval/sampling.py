"""Deterministic streaming reservoir sampling for evaluation strata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class ReservoirSample(Generic[T]):
    """Algorithm R (Vitter): uniform sample of size k from a stream of unknown N."""

    k: int
    seed: int
    rng: np.random.Generator = field(init=False)
    reservoir: list[T] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
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


@dataclass
class StratifiedReservoirs(Generic[T]):
    """Independent reservoirs keyed by stratum (e.g. lang::primary_group)."""

    k_per_stratum: int
    seed: int
    _reservoirs: dict[str, ReservoirSample[T]] = field(default_factory=dict)

    def _get(self, stratum: str) -> ReservoirSample[T]:
        if stratum not in self._reservoirs:
            # Derive a stable per-stratum seed from the master seed + stratum name.
            h = abs(hash((self.seed, stratum))) % (2**31 - 1)
            self._reservoirs[stratum] = ReservoirSample(k=self.k_per_stratum, seed=h)
        return self._reservoirs[stratum]

    def consider(self, stratum: str, item: T) -> None:
        self._get(stratum).consider(item)

    def all_items(self) -> list[T]:
        out: list[T] = []
        for r in self._reservoirs.values():
            out.extend(r.reservoir)
        return out

    def meta(self) -> dict[str, dict]:
        return {
            s: {
                "population_size": r.population_size,
                "k": r.k,
                "n_sampled": len(r.reservoir),
                "inclusion_probability": r.inclusion_probability,
                "seed": r.seed,
            }
            for s, r in self._reservoirs.items()
        }
