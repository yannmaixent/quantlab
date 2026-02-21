import pandas as pd
import numpy as np
import pytest

from quant.metrics.performance import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown
)

def test_total_return():
    equity = pd.Series([100, 110])
    assert compute_total_return(equity) == pytest.approx(0.10)

def test_max_drawdown():
    equity = pd.Series([100, 120, 80, 130])
    # drawdown from 120 -> 80 is -33.333...%
    assert compute_max_drawdown(equity) == pytest.approx(-1/3)


def test_share_zero_vol():
    equity = pd.Series([100, 100, 100, 100])
    assert compute_sharpe_ratio(equity) == 0.0