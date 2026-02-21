def test_fees_reduce_equity():
    import pandas as pd
    from quant.backtest.engine_vector import run_backtest
    from quant.backtest.types import BacktestConfig
    from quant.strategies.buy_and_hold import BuyAndHold

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "open": [1,2,3,4,5],
            "high": [1,2,3,4,5],
            "low": [1,2,3,4,5],
            "close": [1,2,3,4,5],
            "volume": [10,10,10,10,10],
        },
        index=idx,
    )

    cfg_no_fee = BacktestConfig(symbol="SPY", initial_cash=100.0, fees_bps=0.0)
    cfg_fee = BacktestConfig(symbol="SPY", initial_cash=100.0, fees_bps=50.0)

    res_no_fee = run_backtest(prices, BuyAndHold(), cfg_no_fee)
    res_fee = run_backtest(prices, BuyAndHold(), cfg_fee)

    assert res_fee.equity_curve.iloc[-1] < res_no_fee.equity_curve.iloc[-1]
