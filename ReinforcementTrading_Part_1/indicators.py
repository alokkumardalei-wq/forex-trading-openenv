from pathlib import Path

import numpy as np
import pandas as pd


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Keep the scale stable when the series is flat or one-sided.
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    rsi = rsi.mask((avg_gain > 0.0) & (avg_loss == 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def generate_synthetic_fx_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic EURUSD-like OHLCV series for local runs."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2022-01-03", periods=n_rows, freq="1h")
    t = np.arange(n_rows, dtype=np.float64)

    daily_cycle = 0.00016 * np.sin(2.0 * np.pi * t / 24.0)
    weekly_cycle = 0.00010 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))
    regime_shift = np.where((t // 500) % 2 == 0, 0.00003, -0.00003)
    noise = rng.normal(0.0, 0.00035, n_rows)

    log_returns = 0.00001 + daily_cycle + weekly_cycle + regime_shift + noise
    close = 1.08 * np.exp(np.cumsum(log_returns))

    open_prices = np.roll(close, 1)
    open_prices[0] = close[0] * (1.0 - 0.00015)

    spread = np.abs(rng.normal(0.00025, 0.00008, n_rows)) + 0.00005
    high = np.maximum(open_prices, close) + spread
    low = np.minimum(open_prices, close) - spread
    low = np.clip(low, 0.5, None)
    high = np.maximum(high, low + 1e-6)

    volume = rng.integers(900, 2600, size=n_rows).astype(float)
    volume += 250.0 * np.sin(2.0 * np.pi * t / 24.0)
    volume = np.clip(volume, 100.0, None)

    return pd.DataFrame(
        {
            "Time": dates,
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    path = Path(csv_path)

    # Read CSV without specifying time column first to identify it.
    # If the file is missing, fall back to a deterministic synthetic market
    # series so the repo remains self-contained for local validation.
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = generate_synthetic_fx_data()

    # Strip any trailing spaces in headers (e.g. 'Volume ')
    df.columns = df.columns.str.strip()

    # Identify time column dynamically
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    
    if not time_col:
        # Fallback to first column if no 'time' found
        time_col = df.columns[0]

    # Re-read with parse_dates
    df[time_col] = pd.to_datetime(df[time_col])

    # Datetime index
    df = df.set_index(time_col)
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technicals ----
    # RSI and ATR (already scale-invariant-ish)
    df["rsi_14"] = _rsi(df["Close"], length=14)
    df["atr_14"] = _atr(df["High"], df["Low"], df["Close"], length=14)

    # Moving averages
    df["ma_20"] = _sma(df["Close"], length=20)
    df["ma_50"] = _sma(df["Close"], length=50)

    # Slopes of the MAs
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Distance of price from each MA (relative level)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA divergence: MA20 vs MA50
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
    ]

    return df, feature_cols
