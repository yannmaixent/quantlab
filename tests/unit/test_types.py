import pandas as pd

from quant.backtest.types import BacktestConfig, BacktestResult

def test_types_instantiation():
    cfg = BacktestConfig(symbol="SPY", initial_cash=10_000, fees_bps=1.0, slippage_bps=1.0)

    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    res = BacktestResult(
        meta={"symbol": "SPY"},
        equity_curve=pd.Series([1, 2, 3], index=idx, dtype=float),
        positions=pd.Series([0, 1, 1], index=idx, dtype=float),
        trades=pd.DataFrame(),
        metrics={"TotalReturn": 0.0},
        artifacts={},  # ✅ AJOUT
    )

    assert cfg.symbol == "SPY"
    assert "symbol" in res.meta
    assert isinstance(res.equity_curve, pd.Series)
    assert isinstance(res.positions, pd.Series)