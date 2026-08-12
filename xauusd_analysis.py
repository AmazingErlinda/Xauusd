#!/usr/bin/env python3
"""XAUUSD (gold) technical analysis skeleton.

Plug in a real data source in fetch_price_data(), then run:
    python xauusd_analysis.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd


def fetch_price_data(symbol: str = "XAUUSD", interval: str = "1h", lookback: int = 500) -> pd.DataFrame:
    """Return an OHLCV DataFrame indexed by timestamp, columns: open, high, low, close, volume.

    TODO: wire this up to your data provider (broker API, yfinance, Alpha Vantage,
    Twelve Data, MetaTrader, etc.) and return real historical/intraday candles.
    """
    raise NotImplementedError(
        "fetch_price_data() is a stub - connect it to a real XAUUSD data source"
    )


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
    })


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period).std()
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + num_std * std,
        "lower": mid - num_std * std,
    })


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = df.copy()
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["ema_20"] = ema(close, 20)
    out["rsi_14"] = rsi(close, 14)

    macd_df = macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["histogram"]

    bb_df = bollinger_bands(close)
    out["bb_upper"] = bb_df["upper"]
    out["bb_mid"] = bb_df["mid"]
    out["bb_lower"] = bb_df["lower"]
    return out


@dataclass
class Signal:
    name: str
    verdict: str  # "bullish" | "bearish" | "neutral"
    detail: str


def generate_signals(df: pd.DataFrame) -> list[Signal]:
    last = df.iloc[-1]
    signals: list[Signal] = []

    if last["sma_20"] > last["sma_50"] > last["sma_200"]:
        signals.append(Signal("trend", "bullish", "SMA20 > SMA50 > SMA200"))
    elif last["sma_20"] < last["sma_50"] < last["sma_200"]:
        signals.append(Signal("trend", "bearish", "SMA20 < SMA50 < SMA200"))
    else:
        signals.append(Signal("trend", "neutral", "moving averages mixed"))

    if last["rsi_14"] >= 70:
        signals.append(Signal("rsi", "bearish", f"RSI {last['rsi_14']:.1f} overbought"))
    elif last["rsi_14"] <= 30:
        signals.append(Signal("rsi", "bullish", f"RSI {last['rsi_14']:.1f} oversold"))
    else:
        signals.append(Signal("rsi", "neutral", f"RSI {last['rsi_14']:.1f}"))

    if last["macd"] > last["macd_signal"]:
        signals.append(Signal("macd", "bullish", "MACD above signal line"))
    else:
        signals.append(Signal("macd", "bearish", "MACD below signal line"))

    if last["close"] >= last["bb_upper"]:
        signals.append(Signal("bollinger", "bearish", "price at/above upper band"))
    elif last["close"] <= last["bb_lower"]:
        signals.append(Signal("bollinger", "bullish", "price at/below lower band"))
    else:
        signals.append(Signal("bollinger", "neutral", "price within bands"))

    return signals


def print_summary(df: pd.DataFrame, signals: list[Signal]) -> None:
    last = df.iloc[-1]
    print(f"XAUUSD close: {last['close']:.2f}  (as of {df.index[-1]})")
    print("-" * 50)
    for s in signals:
        print(f"{s.name:>10}: {s.verdict:<8} - {s.detail}")

    bullish = sum(1 for s in signals if s.verdict == "bullish")
    bearish = sum(1 for s in signals if s.verdict == "bearish")
    print("-" * 50)
    if bullish > bearish:
        print("Overall bias: BULLISH")
    elif bearish > bullish:
        print("Overall bias: BEARISH")
    else:
        print("Overall bias: NEUTRAL")


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD technical analysis")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback", type=int, default=500)
    args = parser.parse_args()

    df = fetch_price_data(args.symbol, args.interval, args.lookback)
    df = compute_indicators(df)
    signals = generate_signals(df)
    print_summary(df, signals)


if __name__ == "__main__":
    main()
