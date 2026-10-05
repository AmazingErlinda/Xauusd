#!/usr/bin/env python3
"""XAUUSD (gold) technical analysis.

Data sources:
  mt5 (default) - pulls real candles straight from a MetaTrader 5 terminal
      logged into your Vantage account. Requires `pip install MetaTrader5`
      (Windows only) and must run on the same machine/VPS as the terminal.
      If the terminal is already open and logged in, no credentials are
      needed; otherwise set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER env vars
      (or --mt5-login/--mt5-password/--mt5-server) to auto-launch and log in.

  twelvedata - fallback REST API source, free tier at https://twelvedata.com.
      Set TWELVEDATA_API_KEY or pass --api-key.

  csv - offline OHLCV file (e.g. a previous --data-output export). Needs a
      time/datetime column plus open, high, low, close (volume optional).

On top of the classic indicator signals, a quant layer (quant.py) adds regime
detection, a composite score, an ATR-sized trade plan and a backtest. Use
--lookback 1000+ for meaningful backtest statistics.

Usage:
    python xauusd_analysis.py --source mt5 --interval H1 --lookback 2000
    python xauusd_analysis.py --source twelvedata --interval 1h --lookback 2000
    python xauusd_analysis.py --source csv --csv-path xauusd_data.csv --equity 25000 --risk-pct 0.5
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

from quant import print_quant_report, run_quant_analysis

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

MT5_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


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


def load_price_data_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in ("time", "datetime", "date", "timestamp") if c in df.columns), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing columns: {sorted(missing)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


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
    has_quant = "score" in df.columns
    ratios = [3, 1, 1, 1] if has_quant else [3, 1]
    fig, axes = plt.subplots(
        len(ratios), 1, figsize=(12, 4 + 2 * len(ratios)), sharex=True, gridspec_kw={"height_ratios": ratios}
    )
    ax_price, ax_rsi = axes[0], axes[1]

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

    if has_quant:
        ax_score, ax_eq = axes[2], axes[3]
        ax_score.fill_between(df.index, 0, df["score"], where=df["score"] >= 0, color="green", alpha=0.4)
        ax_score.fill_between(df.index, 0, df["score"], where=df["score"] < 0, color="red", alpha=0.4)
        ax_score.axhline(0, color="black", linewidth=0.6)
        ax_score.set_ylim(-1, 1)
        ax_score.set_ylabel("Composite")

        buy_hold = df["close"] / df["close"].iloc[0]
        ax_eq.plot(df.index, df["equity"], label="Strategy", color="tab:blue", linewidth=1)
        ax_eq.plot(df.index, buy_hold, label="Buy & hold", color="gray", linewidth=1, linestyle="--")
        ax_eq.set_ylabel("Equity (x)")
        ax_eq.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD technical analysis")
    parser.add_argument("--source", choices=["mt5", "twelvedata", "csv"], default="mt5")
    parser.add_argument("--csv-path", default=None, help="OHLCV CSV file for --source csv")
    parser.add_argument("--symbol", default=None, help="Defaults to XAUUSD for mt5, XAU/USD for twelvedata")
    parser.add_argument("--interval", default=None, help="Defaults to H1 for mt5, 1h for twelvedata")
    parser.add_argument("--lookback", type=int, default=300)
    parser.add_argument("--api-key", default=None, help="Twelve Data API key (or set TWELVEDATA_API_KEY)")
    parser.add_argument("--mt5-login", default=None, help="Vantage MT5 account login (or set MT5_LOGIN)")
    parser.add_argument("--mt5-password", default=None, help="Vantage MT5 password (or set MT5_PASSWORD)")
    parser.add_argument("--mt5-server", default=None, help='Vantage MT5 server, e.g. "VantageInternational-Live 1" (or set MT5_SERVER)')
    parser.add_argument("--mt5-path", default=None, help="Path to terminal64.exe, if not auto-detected (or set MT5_PATH)")
    parser.add_argument("--chart-output", default="xauusd_chart.png")
    parser.add_argument("--data-output", default="xauusd_data.csv")
    parser.add_argument("--equity", type=float, default=10_000.0, help="Account equity in USD for position sizing")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Percent of equity risked per trade")
    parser.add_argument("--spread", type=float, default=0.30, help="Round-trip cost in USD/oz for the backtest")
    parser.add_argument("--entry-threshold", type=float, default=0.25, help="Composite score needed to enter (0-1)")
    parser.add_argument("--no-quant", action="store_true", help="Skip the quant layer (classic signals only)")
    args = parser.parse_args()

    if args.source == "mt5":
        df = fetch_price_data_mt5(
            args.symbol or "XAUUSD",
            args.interval or "H1",
            args.lookback,
            login=args.mt5_login,
            password=args.mt5_password,
            server=args.mt5_server,
            path=args.mt5_path,
        )
    elif args.source == "csv":
        if not args.csv_path:
            parser.error("--csv-path is required with --source csv")
        df = load_price_data_csv(args.csv_path)
        if args.lookback:
            df = df.tail(args.lookback)
    else:
        df = fetch_price_data_twelvedata(
            args.symbol or "XAU/USD", args.interval or "1h", args.lookback, api_key=args.api_key
        )

    df = compute_indicators(df)
    signals = generate_signals(df)
    print_summary(df, signals)

    if not args.no_quant:
        df, plan, result, hurst = run_quant_analysis(
            df, equity=args.equity, risk_pct=args.risk_pct, spread=args.spread,
            entry_threshold=args.entry_threshold,
        )
        print_quant_report(df, plan, result, hurst)

    df.to_csv(args.data_output)
    print(f"Data saved to {args.data_output}")
    plot_chart(df, args.chart_output)


if __name__ == "__main__":
    main()
