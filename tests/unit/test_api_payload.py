from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from quant.reporting.report import BacktestReport
from quant.backtest.types import BacktestResult


def test_report_payload_is_json_serializable(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")

    res = BacktestResult(
        meta={"symbol": "SPY", "strategy": "buy_and_hold", "engine": "vector_v2_execution"},
        equity_curve=pd.Series([100, 101, 102, 103, 104], index=idx, name="equity"),
        positions=pd.Series([1, 1, 1, 1, 1], index=idx, name="shares"),
        trades=pd.DataFrame(),
        metrics={"total_return": 0.04, "cagr": 0.04, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        artifacts={},
    )

    rep = BacktestReport(res)
    payload = rep.to_payload(tail=3)

    # Must be JSON serializable
    s = json.dumps(payload)
    assert "meta" in payload
    assert "metrics" in payload
    assert "series" in payload
    assert len(payload["series"]["equity_curve"]) == 3
    assert len(payload["series"]["positions"]) == 3

    # Export should write payload.json too
    out_dir = rep.export(tmp_path)
    assert (out_dir / "payload.json").exists()
