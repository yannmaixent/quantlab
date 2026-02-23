import numpy as np
import pandas as pd

from quant.metrics.rolling import rolling_sharpe, rolling_volatility, rolling_max_drawdown, stability_score


def test_rolling_metrics_shapes_and_no_crash():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # geometric growth => constant returns
    equity = pd.Series(100.0 * (1.01 ** np.arange(len(idx))), index=idx)

    w = 10
    rs = rolling_sharpe(equity, window=w, risk_free_rate=0.0)
    rv = rolling_volatility(equity, window=w)
    rdd = rolling_max_drawdown(equity, window=w)

    assert len(rs) == len(equity)
    assert len(rv) == len(equity)
    assert len(rdd) == len(equity)

    assert rs.dropna().size > 0
    assert rv.dropna().size > 0
    assert rdd.dropna().size > 0

    sc = stability_score(rs)
    assert 0.0 <= sc <= 1.0


def test_rolling_max_drawdown_basic_behavior():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    equity = pd.Series([100, 90, 95, 80, 120], index=idx)

    rdd = rolling_max_drawdown(equity, window=5)
    worst = float(rdd.dropna().min())

    # Peak 100 -> trough 80 => -20%
    assert worst <= 0.0
    assert abs(worst - (-0.20)) < 1e-6