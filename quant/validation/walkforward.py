from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterator, Tuple
import pandas as pd


@dataclass(frozen=True)
class WFSplit:
    """
    One walk-forward split: train slice then test slice (time-ordered).
    """
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    prices: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    min_total_bars: int | None = None,
) -> List[WFSplit]:
    """
    Generate rolling time splits.

    Example:
      train_bars=252, test_bars=63, step_bars=63
      -> train 1y, test 3m, step forward by 3m each iteration.

    - step_bars defaults to test_bars.
    - min_total_bars (optional) ensures there is enough history.
    """
    if prices is None or prices.empty:
        raise ValueError("Empty prices dataframe")

    n = len(prices)
    if min_total_bars is None:
        min_total_bars = train_bars + test_bars

    if n < min_total_bars:
        raise ValueError(f"Not enough data for walk-forward. n={n} min={min_total_bars}")

    if step_bars is None:
        step_bars = test_bars

    splits: List[WFSplit] = []

    start_train = 0
    while True:
        end_train = start_train + train_bars
        end_test = end_train + test_bars

        if end_test > n:
            break

        train = prices.iloc[start_train:end_train].copy()
        test = prices.iloc[end_train:end_test].copy()

        splits.append(
            WFSplit(
                train=train,
                test=test,
                train_start=train.index.min(),
                train_end=train.index.max(),
                test_start=test.index.min(),
                test_end=test.index.max(),
            )
        )

        start_train += step_bars

    return splits