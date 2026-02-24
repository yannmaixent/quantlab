from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class RobustnessReport:
    meta: Dict
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    robustness_score: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_robustness_score(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> float:
    """
    V1 robustness score in [0..1].

    Ingredients:
    - Test Sharpe (higher is better)
    - Test MaxDD (closer to 0 is better, it's negative)
    - Divergence between train/test CAGR
    - Stability score (already from rolling metrics) on test
    """
    # safe pulls
    sharpe_t = float(test_metrics.get("sharpe", 0.0))
    maxdd_t = float(test_metrics.get("max_drawdown", 0.0))  # negative
    cagr_tr = float(train_metrics.get("cagr", 0.0))
    cagr_te = float(test_metrics.get("cagr", 0.0))
    stab = float(test_metrics.get("stability_score", 0.0))

    # normalize
    sharpe_part = _sigmoid(sharpe_t)               # 0..1
    dd_part = 1.0 / (1.0 + abs(maxdd_t))           # 1 when dd=0, lower when large dd
    drift = abs(cagr_tr - cagr_te)                 # divergence penalty
    drift_part = 1.0 / (1.0 + drift)

    # stability already 0..1
    stab_part = max(0.0, min(1.0, stab))

    score = 0.35 * sharpe_part + 0.25 * dd_part + 0.20 * drift_part + 0.20 * stab_part
    return float(max(0.0, min(1.0, score)))


def build_report(meta: Dict, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> RobustnessReport:
    score = compute_robustness_score(train_metrics, test_metrics)
    return RobustnessReport(
        meta=meta,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        robustness_score=score,
    )