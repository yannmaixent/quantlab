from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def time_train_test_split(prices: pd.DataFrame, train_ratio: float = 0.7, min_bars: int = 50) -> TimeSplit:
    """
    Time-based split (no shuffling).
    train_ratio in (0,1)
    min_bars ensures both splits have enough data.
    """
    if prices is None or prices.empty:
        raise ValueError("Empty prices dataframe")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    n = len(prices)
    split_idx = int(n * train_ratio)

    train = prices.iloc[:split_idx].copy()
    test = prices.iloc[split_idx:].copy()

    if len(train) < min_bars or len(test) < min_bars:
        raise ValueError(f"Not enough data for split. train={len(train)}, test={len(test)}, min_bars={min_bars}")

    return TimeSplit(train=train, test=test)