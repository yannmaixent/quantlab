import pandas as pd
import pytest

from quant.data.loader import _standardize_ohlcv

def test_standardize_ohlcv_basic():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1.1, 2.1, 3.1],
            "Low": [0.9, 1.9, 2.9],
            "Close": [1.05, 2.05, 3.05],
            "Volume": [100, 200, 300],
            "Dividends": [0, 0, 0],
        },
        index=idx,
    )
    out = _standardize_ohlcv(raw)

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert out[["open", "high", "low", "close"]].isna().sum().sum() == 0


def test_standardize_ohlcv_missing_cols():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    raw = pd.DataFrame({"Open": [1, 2]}, index=idx)
    with pytest.raises(ValueError):
        _standardize_ohlcv(raw)


def test_standardize_handles_spaces_and_duplicates():
    import pandas as pd
    from quant.data.loader import _standardize_ohlcv

    idx = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])
    raw = pd.DataFrame(
        {
            " Open ": [1, 1.1, 2],
            "HiGH": [1.2, 1.3, 2.2],
            "low": [0.9, 1.0, 1.8],
            "Close": [1.09, 1.06, 2.05],
            " Volume": [100, 110, 200],
        },
        index=idx,
    )

    out = _standardize_ohlcv(raw)

    assert out.index.is_unique
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]