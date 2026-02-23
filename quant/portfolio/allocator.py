from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class Allocation:
    """
    Portfolio allocation in weights (sum <= 1.0).
    """
    weights: Dict[str, float]


def equal_weight(symbols: List[str]) -> Allocation:
    if not symbols:
        return Allocation(weights={})
    w = 1.0 / len(symbols)
    return Allocation(weights={s: w for s in symbols})