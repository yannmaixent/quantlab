from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

@dataclass(frozen=True)
class DataSpec:
    symbol: str
    start: Optional[str] = None # "YYYY-MM-DD"
    end: Optional[str] = None
    interval: str = "1d"        # V1: daily only

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to OHLCV with:
    - DatetimeIndex (tz-naive)
    - sorted, unique index
    - columns: open, high, low, close, volume (lowercase)
    """

    if df is None or df.empty:
        raise ValueError("Empty price dataframe received")
    
    # Normalize column names
    df2 = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    # validate required columns on normalized df
    missing = [c for c in OHLCV_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}. Got: {list(df2.columns)}")
    
    out = df2[OHLCV_COLS].copy()

    # Index checks
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Prices index must be a pandas DatetimeIndex")
    
    # Make tz-naive if tz-aware
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Basic numeric coercion
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)

    # validate OHLC not null

    if out[["open", "high", "low", "close"]].isna().any().any():
        # Drop rows with NaN OHLC (can happen on non-trading days in same feeds)
        out = out.dropna(subset=["open", "high", "low", "close"])

    if out.empty:
        raise ValueError("No valid OHLC data after cleaning.")
    
    return out

def load_prices_yfinance(spec: DatSpec) -> pd.DataFrame:
    """
    Load daily OHLCV data from yfinance and standardize.
    """
    if spec.interval != "1d":
        raise ValueError("V1 supports daily data only (interval='1d')")
    
    import yfinance as yf # local import to keep core clean

    ticker = yf.Ticker(spec.symbol)
    df = ticker.history(start=spec.start, end=spec.end, interval=spec.interval, auto_adjust=False)

    # yfinance returns colums like Open High Low Close Volume Dividens Stocks Splits
    df = _standardize_ohlcv(df)

    return df

