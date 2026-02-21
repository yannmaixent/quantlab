from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from quant.reporting.report import BacktestReport
from quant.backtest.types import BacktestResult


def test_report_summary_and_export(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    res = BacktestResult(
        meta={"symbol": "SPY", "strategy": "buy_and_hold", "engine": "vector_v2_execution"},
        equity_curve=pd.Series([100.0, 101.0, 102.0], index=idx, name="equity"),
        positions=pd.Series([1.0, 1.0, 1.0], index=idx, name="shares"),
        trades=pd.DataFrame(),
        metrics={"total_return": 0.02, "cagr": 0.02, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        artifacts={},
    )

    rep = BacktestReport(res)
    s = rep.summary_str()
    assert "Symbol: SPY" in s
    out_dir = rep.export(tmp_path)

    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "positions.csv").exists()
    assert (out_dir / "metrics.json").exists()

    d = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert d["symbol"] == "SPY"
    assert "total_return" in d
