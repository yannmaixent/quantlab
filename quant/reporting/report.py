from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from quant.backtest.types import BacktestResult


@dataclass(frozen=True)
class BacktestReport:
    result: BacktestResult

    def summary_dict(self) -> dict:
        # metrics already computed in result.metrics (day 5)
        m = self.result.metrics.copy()
        return {
            "symbol": self.result.meta.get("symbol"),
            "strategy": self.result.meta.get("strategy"),
            "engine": self.result.meta.get("engine"),
            **m,
        }


    def to_payload(self, tail: int = 200) -> dict:
        """
        Stable payload for API/UI.
        - tail: send only last N points to keep responses light.
        """
        equity_tail = self.result.equity_curve.tail(tail)
        positions_tail = self.result.positions.tail(tail)

        payload = {
            "meta": self.result.meta,
            "metrics": self.result.metrics,
            "series": {
                "equity_curve": [
                    {"ts": str(ts), "value": float(v)}
                    for ts, v in equity_tail.items()
                ],
                "positions": [
                    {"ts": str(ts), "value": float(v)}
                    for ts, v in positions_tail.items()
                ],
            },
        }

        return payload

    

    def summary_str(self) -> str:
        d = self.summary_dict()
        # Format nice + stable (product-ready)
        return (
            f"Symbol: {d.get('symbol')}\n"
            f"Strategy: {d.get('strategy')}\n"
            f"Engine: {d.get('engine')}\n"
            f"Total Return: {d.get('total_return', 0.0):.2%}\n"
            f"CAGR: {d.get('cagr', 0.0):.2%}\n"
            f"Volatility: {d.get('volatility', 0.0):.2%}\n"
            f"Sharpe: {d.get('sharpe', 0.0):.3f}\n"
            f"Max Drawdown: {d.get('max_drawdown', 0.0):.2%}\n"
        )

    def export(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSVs
        self.result.equity_curve.to_csv(out_dir / "equity_curve.csv", header=True)
        self.result.positions.to_csv(out_dir / "positions.csv", header=True)

        # metrics.json
        metrics_path = out_dir / "metrics.json"
        payload_path = out_dir / "payload.json"
        with payload_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_payload(), f, indent=2)

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, indent=2)

        return out_dir
