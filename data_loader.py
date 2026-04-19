"""
data_loader.py
Responsible for:
  1. Download stock data from yfinance
  2. Compute Bollinger Band primary technical indicators (BB + EMA + VOL)
  3. Generate training samples based on BB buy signal (mean reversion)

BB Signal Logic (mean reversion):
  Buy  signal : close <= BB_lower  (oversold, expect upward reversion)
  Sell signal : close >= BB_upper  (overbought, expect downward reversion)

Model predicts: when BB buy signal triggers, will holding N days be profitable?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from datetime import datetime

# Feature columns used by LSTM (must match between train and predict)
FEATURE_COLS = [
    "EMA_20", "EMA_60", "EMA_trend",
    "BB_upper", "BB_mid", "BB_lower", "BB_width", "BB_pct",
    "BB_above_upper", "BB_below_lower", "BB_squeeze",
    "VOL_ratio", "VOL_log",
    "Return_1d", "Return_5d", "Return_10d",
    "High_Low_range",
    "Signal_buy", "Signal_sell",
]


def download_data(ticker: str,
                  start: str,
                  end: str = None,
                  interval: str = '1d') -> pd.DataFrame:
    """
    Download OHLCV data from yfinance.

    Args:
        ticker   : e.g. "2330.TW"
        start    : start date string "YYYY-MM-DD"
        end      : end date string (None = today)
        interval : '1d' for daily, '1wk' for weekly, '1mo' for monthly
    """
    end_str  = end if end else datetime.today().strftime("%Y-%m-%d")
    bar_name = {"1d": "days", "1wk": "weeks", "1mo": "months"}.get(interval, interval)

    print(f"\n{'='*55}")
    print(f"  [DataLoader] {ticker}  {start} ~ {end_str}  [{interval}]")
    print(f"{'='*55}")

    df = yf.download(ticker, start=start, end=end_str,
                     interval=interval, progress=False)
    df.dropna(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Downloaded {len(df)} {bar_name}")
    print(f"  Range: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Strategy 4 technical indicators.
    Used by both train.py and predict.py.

    Does NOT compute Label here (Label depends on hold_days).
    """
    df = df.copy()
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # EMA
    df["EMA_20"]    = EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA_60"]    = EMAIndicator(close=close, window=60).ema_indicator()
    df["EMA_trend"] = (df["EMA_20"] > df["EMA_60"]).astype(int)

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_upper"]       = bb.bollinger_hband()
    df["BB_mid"]         = bb.bollinger_mavg()
    df["BB_lower"]       = bb.bollinger_lband()
    df["BB_width"]       = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"]         = bb.bollinger_pband()
    df["BB_above_upper"] = (close >= df["BB_upper"]).astype(int)
    df["BB_below_lower"] = (close <= df["BB_lower"]).astype(int)

    # BB Squeeze: BB_width is narrower than its 20-day average * 0.85
    # → low volatility period, often precedes a breakout
    bb_width_ma          = df["BB_width"].rolling(20).mean()
    df["BB_squeeze"]     = (df["BB_width"] < bb_width_ma * 0.85).astype(int)

    # Volume
    df["VOL_MA20"]  = volume.rolling(20).mean()
    df["VOL_ratio"] = volume / df["VOL_MA20"]
    df["VOL_log"]   = np.log1p(volume)

    # ── BB Primary Signals (mean reversion) ──────────────────
    # Buy  signal: close <= BB_lower → price is oversold, expect upward reversion
    # Sell signal: close >= BB_upper → price is overbought, expect downward reversion
    # EMA trend remains as a FEATURE for the model (context), not as a gate
    df["Signal_buy"]  = df["BB_below_lower"]   # 接觸下軌 → 超賣買訊
    df["Signal_sell"] = df["BB_above_upper"]   # 接觸上軌 → 超買賣訊

    # Price features
    df["Return_1d"]      = close.pct_change(1)
    df["Return_5d"]      = close.pct_change(5)
    df["Return_10d"]     = close.pct_change(10)
    df["High_Low_range"] = (df["High"] - df["Low"]) / close

    df.dropna(inplace=True)
    print(f"  Features computed: {len(FEATURE_COLS)} cols  {len(df)} rows")
    return df


def compute_signal_labels(df: pd.DataFrame,
                          hold_days: int = 5) -> pd.DataFrame:
    """
    Generate training samples based on BB buy signal (mean reversion).

    LABEL LOGIC:
      Only on days where Signal_buy=1 (close <= BB_lower):
        Label = 1  if close[t + hold_days] > close[t]  (profitable, reversion worked)
        Label = 0  if close[t + hold_days] <= close[t] (unprofitable, reversion failed)

    This means:
      - Training data is filtered to BB lower-band touch days only
      - LSTM learns: "when BB buy signal triggers, will holding N days be profitable?"
      - Model captures context: EMA trend, BB squeeze, volume, recent returns, etc.

    Args:
        df        : DataFrame from compute_features()
        hold_days : number of days to hold after buy signal

    Returns:
        DataFrame with only Signal_buy=1 rows and their Labels
    """
    df = df.copy()
    close = df["Close"].squeeze()

    # Compute forward return after hold_days
    df["Forward_return"] = close.shift(-hold_days) / close - 1
    df["Label"]          = (df["Forward_return"] > 0).astype(int)

    # Keep only rows where buy signal triggered
    # AND we have enough future data (not in last hold_days rows)
    signal_df = df[df["Signal_buy"] == 1].copy()
    signal_df = signal_df.iloc[:-hold_days] if len(signal_df) > hold_days else signal_df
    signal_df.dropna(subset=["Label"], inplace=True)

    print(f"\n  [BB Signal Labels] hold_days = {hold_days}")
    print(f"  BB Signal type        : Mean Reversion (close <= BB_lower)")
    print(f"  Total trading days    : {len(df)}")
    print(f"  BB buy signal days    : {len(signal_df)} days  (close touched lower band)")
    print(f"  Signal rate           : {len(signal_df)/len(df):.2%}")
    if len(signal_df) > 0:
        up_pct = signal_df["Label"].mean()
        print(f"  Profitable signals    : {up_pct:.2%}  (reversion succeeded)")
        print(f"  Sample dates (first 5):")
        for d in signal_df.index[:5]:
            print(f"    {d.date()}  close={float(close[d]):.0f}  "
                  f"label={int(signal_df.loc[d,'Label'])}")

    return signal_df, df   # signal_df for training, df for full features