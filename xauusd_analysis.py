#!/usr/bin/env python3
"""XAUUSD (gold) and BTCUSD (bitcoin) technical analysis.

Data sources:
  mt5 (default) - pulls real candles straight from a MetaTrader 5 terminal
      logged into your Vantage account. Requires `pip install MetaTrader5`
      (Windows only) and must run on the same machine/VPS as the terminal.
      If the terminal is already open and logged in, no credentials are
      needed; otherwise set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER env vars
      (or --mt5-login/--mt5-password/--mt5-server) to auto-launch and log in.
      BTCUSD requires your Vantage account to offer crypto CFDs - check
      Market Watch for the exact symbol name (e.g. BTCUSD, BTCUSD.a).

  twelvedata - fallback REST API source, free tier at https://twelvedata.com.
      Set TWELVEDATA_API_KEY or pass --api-key. Supports crypto pairs
      natively, so this is the simplest route for BTCUSD if you don't run
      MT5 locally.

Usage:
    python xauusd_analysis.py --source mt5 --interval H1 --lookback 300
    python xauusd_analysis.py --source twelvedata --interval 1h --lookback 300
    python xauusd_analysis.py --symbol BTCUSD --source twelvedata --interval 1h
    python xauusd_analysis.py --symbol BTCUSD --watch --poll-interval 300
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

MT5_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]

# Maps a friendly --symbol preset to the source-specific symbol string.
# Any other --symbol value is passed through to the source as-is.
SYMBOL_PRESETS = {
    "XAUUSD": {"mt5": "XAUUSD", "twelvedata": "XAU/USD"},
    "BTCUSD": {"mt5": "BTCUSD", "twelvedata": "BTC/USD"},
}


def fetch_price_data_mt5(
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    lookback: int = 300,
    login: str | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    """Return an OHLCV DataFrame pulled live from a MetaTrader 5 terminal logged
    into your Vantage account.

    If the terminal is already running and logged in, just call this with no
    credentials. Otherwise supply login/password/server (e.g. server=
    "VantageInternational-Live 1") to have MT5 auto-launch and log in - get
    these from your Vantage account details, not from this script.

    Vantage sometimes suffixes the gold symbol per account type (e.g.
    XAUUSD.a, XAUUSD_i, GOLD) - check your terminal's Market Watch if the
    default "XAUUSD" isn't found.
    """
    import MetaTrader5 as mt5  # local import: package only installs/imports on Windows

    login = login or os.environ.get("MT5_LOGIN")
    password = password or os.environ.get("MT5_PASSWORD")
    server = server or os.environ.get("MT5_SERVER")
    path = path or os.environ.get("MT5_PATH")

    init_kwargs = {}
    if path:
        init_kwargs["path"] = path
    if login and password and server:
        init_kwargs.update(login=int(login), password=password, server=server)

    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol '{symbol}' not available in Market Watch: {mt5.last_error()}. "
                "Vantage sometimes suffixes gold symbols (e.g. XAUUSD.a, XAUUSD_i, GOLD) "
                "- check your terminal's Market Watch for the exact name."
            )

        tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}", None)
        if tf_const is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {MT5_TIMEFRAMES}")

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, lookback)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {symbol}: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    return _parse_mt5_rates(rates)


def _parse_mt5_rates(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def fetch_price_data_twelvedata(
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


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Prior N-bar high/low (current bar excluded), used to spot breakouts."""
    high = df["high"].shift(1).rolling(period).max()
    low = df["low"].shift(1).rolling(period).min()
    return pd.DataFrame({"high": high, "low": low})


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

    dc_df = donchian_channel(df, 20)
    out["donchian_high"] = dc_df["high"]
    out["donchian_low"] = dc_df["low"]
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

    if pd.notna(last["donchian_high"]) and last["close"] > last["donchian_high"]:
        signals.append(Signal("breakout", "bullish", f"close broke above 20-bar high {last['donchian_high']:.2f}"))
    elif pd.notna(last["donchian_low"]) and last["close"] < last["donchian_low"]:
        signals.append(Signal("breakout", "bearish", f"close broke below 20-bar low {last['donchian_low']:.2f}"))
    else:
        signals.append(Signal("breakout", "neutral", "price within 20-bar range"))

    return signals


def send_webhook_alert(webhook_url: str, text: str) -> None:
    """POST a simple {"text": ...} / {"content": ...} payload, compatible
    with Slack and Discord incoming webhooks."""
    try:
        requests.post(webhook_url, json={"text": text, "content": text}, timeout=10)
    except requests.RequestException as exc:
        print(f"Warning: webhook alert failed: {exc}")


def watch_for_breakout(fetch_fn, symbol_label: str, poll_interval: int, webhook_url: str | None) -> None:
    """Poll the data source on a fixed interval and alert the moment a new
    Donchian breakout signal appears on the latest closed bar, so you catch
    the next takeoff instead of finding out after the fact.
    """
    print(f"Watching {symbol_label} for breakouts every {poll_interval}s. Ctrl+C to stop.")
    last_alerted_bar = None
    while True:
        try:
            df = compute_indicators(fetch_fn())
            last = df.iloc[-1]
            bar_time = df.index[-1]

            breakout = None
            if pd.notna(last["donchian_high"]) and last["close"] > last["donchian_high"]:
                breakout = f"BULLISH BREAKOUT: {symbol_label} closed {last['close']:.2f}, above 20-bar high {last['donchian_high']:.2f} (bar {bar_time})"
            elif pd.notna(last["donchian_low"]) and last["close"] < last["donchian_low"]:
                breakout = f"BEARISH BREAKDOWN: {symbol_label} closed {last['close']:.2f}, below 20-bar low {last['donchian_low']:.2f} (bar {bar_time})"

            if breakout and last_alerted_bar != bar_time:
                print(f"\a{breakout}")
                if webhook_url:
                    send_webhook_alert(webhook_url, breakout)
                last_alerted_bar = bar_time
            else:
                print(f"{bar_time}  {symbol_label} {last['close']:.2f}  no new breakout")
        except Exception as exc:
            print(f"Warning: poll failed: {exc}")

        time.sleep(poll_interval)


def print_summary(df: pd.DataFrame, signals: list[Signal], symbol_label: str = "XAUUSD") -> None:
    last = df.iloc[-1]
    print(f"{symbol_label} close: {last['close']:.2f}  (as of {df.index[-1]})")
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


def plot_chart(df: pd.DataFrame, output_path: str = "xauusd_chart.png", symbol_label: str = "XAUUSD") -> None:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1.2)
    ax_price.plot(df.index, df["sma_20"], label="SMA 20", linewidth=1)
    ax_price.plot(df.index, df["sma_50"], label="SMA 50", linewidth=1)
    ax_price.fill_between(
        df.index, df["bb_lower"], df["bb_upper"], color="gray", alpha=0.15, label="Bollinger Bands"
    )
    ax_price.plot(df.index, df["donchian_high"], label="20-bar high", color="green", linewidth=0.8, linestyle="--")
    ax_price.plot(df.index, df["donchian_low"], label="20-bar low", color="red", linewidth=0.8, linestyle="--")
    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title(f"{symbol_label} Price & Moving Averages")
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


def resolve_symbol(symbol_arg: str | None, source: str) -> tuple[str, str]:
    """Returns (source-specific symbol, friendly label) for a --symbol preset
    (e.g. "BTCUSD") or a literal symbol string passed straight through."""
    preset_key = (symbol_arg or "XAUUSD").upper()
    if preset_key in SYMBOL_PRESETS:
        return SYMBOL_PRESETS[preset_key][source], preset_key
    return symbol_arg, symbol_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD / BTCUSD technical analysis")
    parser.add_argument("--source", choices=["mt5", "twelvedata"], default="mt5")
    parser.add_argument("--symbol", default=None, help='Preset "XAUUSD" or "BTCUSD", or a literal source symbol')
    parser.add_argument("--interval", default=None, help="Defaults to H1 for mt5, 1h for twelvedata")
    parser.add_argument("--lookback", type=int, default=300)
    parser.add_argument("--api-key", default=None, help="Twelve Data API key (or set TWELVEDATA_API_KEY)")
    parser.add_argument("--mt5-login", default=None, help="Vantage MT5 account login (or set MT5_LOGIN)")
    parser.add_argument("--mt5-password", default=None, help="Vantage MT5 password (or set MT5_PASSWORD)")
    parser.add_argument("--mt5-server", default=None, help='Vantage MT5 server, e.g. "VantageInternational-Live 1" (or set MT5_SERVER)')
    parser.add_argument("--mt5-path", default=None, help="Path to terminal64.exe, if not auto-detected (or set MT5_PATH)")
    parser.add_argument("--chart-output", default="xauusd_chart.png")
    parser.add_argument("--data-output", default="xauusd_data.csv")
    parser.add_argument("--watch", action="store_true", help="Poll continuously and alert on new 20-bar breakouts instead of a one-shot analysis")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls in --watch mode")
    parser.add_argument("--webhook-url", default=None, help="Slack/Discord incoming webhook URL for breakout alerts in --watch mode")
    args = parser.parse_args()

    symbol, symbol_label = resolve_symbol(args.symbol, args.source)

    if args.source == "mt5":
        def fetch() -> pd.DataFrame:
            return fetch_price_data_mt5(
                symbol,
                args.interval or "H1",
                args.lookback,
                login=args.mt5_login,
                password=args.mt5_password,
                server=args.mt5_server,
                path=args.mt5_path,
            )
    else:
        def fetch() -> pd.DataFrame:
            return fetch_price_data_twelvedata(
                symbol, args.interval or "1h", args.lookback, api_key=args.api_key
            )

    if args.watch:
        watch_for_breakout(fetch, symbol_label, args.poll_interval, args.webhook_url)
        return

    df = compute_indicators(fetch())
    signals = generate_signals(df)
    print_summary(df, signals, symbol_label)

    df.to_csv(args.data_output)
    print(f"Data saved to {args.data_output}")
    plot_chart(df, args.chart_output, symbol_label)


if __name__ == "__main__":
    main()
