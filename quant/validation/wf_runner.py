from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.validation.walkforward import walk_forward_splits
from quant.validation.robustness import compute_robustness_score


@dataclass(frozen=True)
class WalkForwardReport:
    meta: Dict
    per_window: pd.DataFrame
    summary: Dict[str, float]


def run_walk_forward(
    prices: pd.DataFrame,
    strategy,
    config: BacktestConfig,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
) -> WalkForwardReport:
    splits = walk_forward_splits(
        prices=prices,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )

    rows: List[Dict] = []
    for i, sp in enumerate(splits, start=1):
        res_train = run_backtest(sp.train, strategy, config)
        res_test = run_backtest(sp.test, strategy, config)

        score = compute_robustness_score(res_train.metrics, res_test.metrics)

        rows.append(
            {
                "window": i,
                "train_start": sp.train_start,
                "train_end": sp.train_end,
                "test_start": sp.test_start,
                "test_end": sp.test_end,
                "train_sharpe": res_train.metrics.get("sharpe"),
                "test_sharpe": res_test.metrics.get("sharpe"),
                "train_cagr": res_train.metrics.get("cagr"),
                "test_cagr": res_test.metrics.get("cagr"),
                "train_maxdd": res_train.metrics.get("max_drawdown"),
                "test_maxdd": res_test.metrics.get("max_drawdown"),
                "test_stability": res_test.metrics.get("stability_score"),
                "robustness_score": score,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No walk-forward windows produced. Increase data range or reduce bars.")

    # Summary stats (monetisable)
    mean_score = float(df["robustness_score"].mean())
    std_score = float(df["robustness_score"].std(ddof=0))
    min_score = float(df["robustness_score"].min())
    pass_rate = float((df["test_sharpe"] > 0.0).mean())

    summary = {
        "wf_windows": float(len(df)),
        "robustness_mean": mean_score,
        "robustness_std": std_score,
        "robustness_min": min_score,
        "test_sharpe_pass_rate": pass_rate,
    }

    return WalkForwardReport(
        meta={
            "symbol": config.symbol,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "step_bars": step_bars if step_bars is not None else test_bars,
        },
        per_window=df,
        summary=summary,
    )