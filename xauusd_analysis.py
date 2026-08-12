#!/usr/bin/env python3
"""XAUUSD (gold) technical analysis.

Data source: Twelve Data (https://twelvedata.com), free tier.
Get a free API key at https://twelvedata.com/pricing and either:
    export TWELVEDATA_API_KEY=your_key_here
or pass --api-key your_key_here on the command line.

Usage:
    python xauusd_analysis.py --interval 1h --lookback 300
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


def fetch_price_data(
    symbol: str = "XAU/USD",
    interval: str = "1h",
    lookback: int = 300,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Return an OHLCV DataFrame indexed by timestamp, columns: open, high, low, close, volume.

    Pulls candles from the Twelve Data time_series endpoint. Requires an API
    key: pass api_key explicitly or set the TWELVEDATA_API_KEY env var.
    """
    api_key = api_key or os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Twelve Data API key found. Set the TWELVEDATA_API_KEY env var "
            "or pass --api-key. Get a free key at https://twelvedata.com/pricing"
        )

    response = requests.get(
        TWELVEDATA_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": lookback,
            "apikey": api_key,
            "format": "JSON",
        },
        timeout=15,
    )
    response.raise_for_status()
    return _parse_twelvedata_response(response.json())


def _parse_twelvedata_response(payload: dict) -> pd.DataFrame:
    if payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data API error: {payload.get('message')}")

    values = payload.get("values")
    if not values:
        raise RuntimeError("Twelve Data API returned no candle data")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    return df[["open", "high", "low", "close", "volume"]]


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


def plot_chart(df: pd.DataFrame, output_path: str = "xauusd_chart.png") -> None:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1.2)
    ax_price.plot(df.index, df["sma_20"], label="SMA 20", linewidth=1)
    ax_price.plot(df.index, df["sma_50"], label="SMA 50", linewidth=1)
    ax_price.fill_between(
        df.index, df["bb_lower"], df["bb_upper"], color="gray", alpha=0.15, label="Bollinger Bands"
    )
    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title("XAUUSD Price & Moving Averages")
    ax_price.legend(loc="upper left")

    ax_rsi.plot(df.index, df["rsi_14"], color="purple", linewidth=1)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI 14")
    ax_rsi.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD technical analysis")
    parser.add_argument("--symbol", default="XAU/USD")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback", type=int, default=300)
    parser.add_argument("--api-key", default=None, help="Twelve Data API key (or set TWELVEDATA_API_KEY)")
    parser.add_argument("--chart-output", default="xauusd_chart.png")
    parser.add_argument("--data-output", default="xauusd_data.csv")
    args = parser.parse_args()

    df = fetch_price_data(args.symbol, args.interval, args.lookback, api_key=args.api_key)
    df = compute_indicators(df)
    signals = generate_signals(df)
    print_summary(df, signals)

    df.to_csv(args.data_output)
    print(f"Data saved to {args.data_output}")
    plot_chart(df, args.chart_output)


if __name__ == "__main__":
    main()
