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

Usage:
    python xauusd_analysis.py --source mt5 --interval H1 --lookback 300
    python xauusd_analysis.py --source twelvedata --interval 1h --lookback 300
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean()


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
    out["atr_14"] = atr(df, 14)
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


def find_swings(df: pd.DataFrame, pct_threshold: float = 1.0) -> pd.DataFrame:
    """Zigzag swing detection.

    A pivot high/low is confirmed once price reverses by at least
    pct_threshold percent from the running extreme since the last
    confirmed pivot. Returns a DataFrame indexed by time with columns
    'price' and 'kind' ('high' or 'low'), oldest first. This is a
    mechanical proxy for "significant swings" - it does not know about
    Elliott Wave rules by itself, that's handled downstream.
    """
    if len(df) < 2:
        return pd.DataFrame(columns=["price", "kind"])

    threshold = pct_threshold / 100.0
    highs, lows = df["high"], df["low"]

    hi_price, hi_time = highs.iloc[0], df.index[0]
    lo_price, lo_time = lows.iloc[0], df.index[0]
    trend = 0  # 0 = undetermined, 1 = last pivot was a low, -1 = last pivot was a high
    pivots: list[tuple] = []

    for t in df.index[1:]:
        h, l = highs[t], lows[t]

        if trend == 0:
            if h > hi_price:
                hi_price, hi_time = h, t
            if l < lo_price:
                lo_price, lo_time = l, t
            if lo_price <= hi_price * (1 - threshold):
                pivots.append((hi_time, hi_price, "high"))
                trend = -1
                lo_price, lo_time = l, t
            elif hi_price >= lo_price * (1 + threshold):
                pivots.append((lo_time, lo_price, "low"))
                trend = 1
                hi_price, hi_time = h, t
        elif trend == 1:  # last pivot was a low; watching for a new high or a reversal down
            if h > hi_price:
                hi_price, hi_time = h, t
            if l <= hi_price * (1 - threshold):
                pivots.append((hi_time, hi_price, "high"))
                trend = -1
                lo_price, lo_time = l, t
        else:  # trend == -1, last pivot was a high; watching for a new low or a reversal up
            if l < lo_price:
                lo_price, lo_time = l, t
            if h >= lo_price * (1 + threshold):
                pivots.append((lo_time, lo_price, "low"))
                trend = 1
                hi_price, hi_time = h, t

    return pd.DataFrame(pivots, columns=["time", "price", "kind"]).set_index("time")


@dataclass
class ElliottWaveRead:
    direction: str  # 'up' | 'down'
    points: list  # [(time, price, kind), ...] the swing points the count is based on
    position: str
    next_target_low: float | None
    next_target_high: float | None
    notes: list[str]


def analyze_elliott_wave(swings: pd.DataFrame, max_points: int = 6) -> ElliottWaveRead | None:
    """Heuristic Elliott Wave read off the most recent zigzag swings.

    Fits the last up-to-6 alternating swing points as a 5-wave impulse
    (0-1-2-3-4-5), checks the three hard impulse rules (wave 2 can't
    retrace 100%+ of wave 1, wave 3 can't be the shortest of 1/3/5, wave 4
    can't overlap wave 1), and projects a Fibonacci target zone for
    whichever wave looks to be next. This is decision support, not a
    guaranteed count - conflicting counts are common in real markets and
    the rule-violation notes are the signal to distrust it.
    """
    if len(swings) < 2:
        return None

    recent = swings.tail(max_points)
    points = [(t, float(row["price"]), row["kind"]) for t, row in recent.iterrows()]
    direction = "up" if points[-1][2] == "high" else "down"
    expected_start_kind = "low" if direction == "up" else "high"
    if points[0][2] != expected_start_kind:
        points = points[1:]
    if len(points) < 2:
        return None

    prices = [p[1] for p in points]
    n = len(points)
    wave_count = n - 1
    sign = 1 if direction == "up" else -1
    notes: list[str] = []

    def leg(a: int, b: int) -> float:
        return (prices[b] - prices[a]) * sign

    if wave_count >= 2 and leg(1, 2) * -1 >= leg(0, 1):
        notes.append(
            "Rule violation: wave 2 retraces 100%+ of wave 1 - this may be a "
            "corrective (A-B-C), not an impulsive 5-wave move."
        )
    if wave_count >= 4:
        w1, w3 = abs(leg(0, 1)), abs(leg(2, 3))
        if wave_count >= 6:
            w5 = abs(leg(4, 5))
            if w3 < w1 and w3 < w5:
                notes.append("Rule violation: wave 3 is the shortest of 1/3/5 - invalid impulse count.")
        if sign * (prices[4] - prices[1]) < 0:
            notes.append("Rule violation: wave 4 overlaps wave 1 territory (only allowed in a diagonal).")

    next_low = next_high = None
    if wave_count == 1:
        position = "Wave 1 complete - wave 2 pullback likely in progress"
        w1 = abs(leg(0, 1))
        next_low = prices[1] - sign * w1 * 0.618
        next_high = prices[1] - sign * w1 * 0.382
    elif wave_count == 2:
        position = "Wave 2 complete - wave 3 (often the extended leg) likely starting"
        w1 = abs(leg(0, 1))
        next_low = prices[2] + sign * w1 * 1.0
        next_high = prices[2] + sign * w1 * 1.618
    elif wave_count == 3:
        position = "Wave 3 complete - wave 4 pullback likely in progress"
        w3 = abs(leg(2, 3))
        next_low = prices[3] - sign * w3 * 0.382
        next_high = prices[3] - sign * w3 * 0.236
    elif wave_count == 4:
        position = "Wave 4 complete - wave 5 (final leg) likely starting"
        w1 = abs(leg(0, 1))
        w13 = abs(prices[3] - prices[0])
        lo, hi = sorted([w1, w13 * 0.618])
        next_low = prices[4] + sign * lo
        next_high = prices[4] + sign * hi
    elif wave_count >= 5:
        position = "5-wave impulse complete - expect an A-B-C correction against the trend"
        impulse_len = abs(prices[wave_count] - prices[0])
        next_low = prices[wave_count] - sign * impulse_len * 0.618
        next_high = prices[wave_count] - sign * impulse_len * 0.382
    else:
        position = "Not enough confirmed swings yet"

    lo_hi = sorted([v for v in (next_low, next_high) if v is not None])
    return ElliottWaveRead(
        direction=direction,
        points=points,
        position=position,
        next_target_low=round(lo_hi[0], 2) if lo_hi else None,
        next_target_high=round(lo_hi[-1], 2) if lo_hi else None,
        notes=notes,
    )


GANN_ANGLE_RATIOS = {
    "1x8": 1 / 8, "1x4": 1 / 4, "1x3": 1 / 3, "1x2": 1 / 2,
    "1x1": 1.0,
    "2x1": 2.0, "3x1": 3.0, "4x1": 4.0, "8x1": 8.0,
}


def gann_angles(
    df: pd.DataFrame,
    pivot_time: pd.Timestamp,
    pivot_price: float,
    pivot_kind: str,
    unit_per_bar: float,
) -> pd.DataFrame:
    """Project classic Gann angle lines forward from a pivot.

    unit_per_bar sets the price move of the 1x1 (45-degree) angle per bar;
    the other named angles scale that unit by their ratio. Angles rise from
    a 'low' pivot (acting as support) and fall from a 'high' pivot (acting
    as resistance).
    """
    bars = df.index[df.index >= pivot_time]
    sign = 1 if pivot_kind == "low" else -1
    data = {name: [pivot_price + sign * ratio * unit_per_bar * i for i in range(len(bars))]
            for name, ratio in GANN_ANGLE_RATIOS.items()}
    return pd.DataFrame(data, index=bars)


def gann_square_of_9_levels(price: float, num_rings: int = 3) -> list[float]:
    """Gann Square-of-9 support/resistance levels via the square-root method.

    Moving sqrt(price) by increments of 0.125 (=45 degrees of rotation) and
    squaring back gives the cardinal (90/180/270/360) and diagonal
    (45/135/225/315) levels around the current price; each full 1.0 step in
    sqrt-space is one more ring outward (360 degrees).
    """
    root = math.sqrt(price)
    offsets = [ring + step * 0.125 for ring in range(num_rings) for step in range(1, 9)]
    levels = {round((root + off) ** 2, 2) for off in offsets}
    levels |= {round((root - off) ** 2, 2) for off in offsets if root - off > 0}
    return sorted(levels)


def nearest_gann_levels(price: float, levels: list[float], n: int = 3) -> tuple[list[float], list[float]]:
    below = sorted((l for l in levels if l < price), reverse=True)[:n]
    above = sorted((l for l in levels if l > price))[:n]
    return list(reversed(below)), above


GANN_TIME_CYCLE_BARS = [30, 45, 60, 72, 90, 120, 144, 180, 270, 360]


def gann_time_cycles(pivot_time: pd.Timestamp, bar_interval: pd.Timedelta) -> pd.DataFrame:
    """Classic Gann/Fibonacci bar counts projected forward from a pivot as
    dates to watch for a potential trend change (time, not price, signal)."""
    rows = [{"bars": b, "date": pivot_time + bar_interval * b} for b in GANN_TIME_CYCLE_BARS]
    return pd.DataFrame(rows)


def print_elliott_gann(df: pd.DataFrame, swings: pd.DataFrame, ew: ElliottWaveRead | None,
                        gann_pivot: tuple | None, angle_unit: float, sq9_levels: list[float],
                        cycles: pd.DataFrame | None) -> None:
    last_price = df["close"].iloc[-1]

    print("\n" + "=" * 60)
    print("ELLIOTT WAVE (heuristic zigzag read)")
    print("=" * 60)
    if ew is None:
        print("Not enough confirmed swings yet - widen --lookback or loosen --zigzag-pct.")
    else:
        print(f"Direction: {ew.direction}")
        print("Swing points used:")
        for t, price, kind in ew.points:
            print(f"  {t}  {kind:>4}  {price:.2f}")
        print(f"Position: {ew.position}")
        if ew.next_target_low is not None:
            print(f"Next target zone: {ew.next_target_low:.2f} - {ew.next_target_high:.2f}")
        for note in ew.notes:
            print(f"  ! {note}")

    print("\n" + "=" * 60)
    print("GANN ANALYSIS")
    print("=" * 60)
    if gann_pivot is None:
        print("No pivot available for Gann angles yet.")
    else:
        pivot_time, pivot_price, pivot_kind = gann_pivot
        print(f"Pivot: {pivot_kind} @ {pivot_price:.2f} on {pivot_time} (unit/bar = {angle_unit:.3f})")
        print("Angle lines at latest bar:")
        angles = gann_angles(df, pivot_time, pivot_price, pivot_kind, angle_unit)
        for name in GANN_ANGLE_RATIOS:
            print(f"  {name:>4}: {angles[name].iloc[-1]:.2f}")

    below, above = nearest_gann_levels(last_price, sq9_levels)
    print(f"\nSquare-of-9 levels near current price ({last_price:.2f}):")
    print(f"  Support:    {', '.join(f'{v:.2f}' for v in below) if below else 'n/a'}")
    print(f"  Resistance: {', '.join(f'{v:.2f}' for v in above) if above else 'n/a'}")

    if cycles is not None and gann_pivot is not None:
        print(f"\nTime-cycle watch dates (from pivot {gann_pivot[0]}):")
        for _, row in cycles.iterrows():
            print(f"  {row['bars']:>4} bars -> {row['date']}")


def plot_chart(
    df: pd.DataFrame,
    output_path: str = "xauusd_chart.png",
    ew: ElliottWaveRead | None = None,
    gann_pivot: tuple | None = None,
    angle_unit: float | None = None,
    sq9_levels: list[float] | None = None,
) -> None:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1.2)
    ax_price.plot(df.index, df["sma_20"], label="SMA 20", linewidth=1)
    ax_price.plot(df.index, df["sma_50"], label="SMA 50", linewidth=1)
    ax_price.fill_between(
        df.index, df["bb_lower"], df["bb_upper"], color="gray", alpha=0.15, label="Bollinger Bands"
    )

    if ew is not None and len(ew.points) >= 2:
        wave_times = [p[0] for p in ew.points]
        wave_prices = [p[1] for p in ew.points]
        ax_price.plot(wave_times, wave_prices, color="orange", linewidth=1.5, marker="o",
                      markersize=4, label="Elliott swings")
        for i, (t, price, _kind) in enumerate(ew.points):
            ax_price.annotate(str(i), (t, price), textcoords="offset points", xytext=(4, 4),
                               fontsize=9, color="darkorange")
        if ew.next_target_low is not None:
            ax_price.axhspan(ew.next_target_low, ew.next_target_high, color="orange", alpha=0.12,
                              label="Next EW target zone")

    if gann_pivot is not None and angle_unit:
        pivot_time, pivot_price, pivot_kind = gann_pivot
        angles = gann_angles(df, pivot_time, pivot_price, pivot_kind, angle_unit)
        for name in ("1x1", "2x1", "1x2"):
            ax_price.plot(angles.index, angles[name], linestyle="--", linewidth=0.8, alpha=0.7,
                          label=f"Gann {name}")

    if sq9_levels:
        last_price = df["close"].iloc[-1]
        below, above = nearest_gann_levels(last_price, sq9_levels, n=2)
        for level in below + above:
            ax_price.axhline(level, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.6)

    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title("XAUUSD Price, Elliott Wave & Gann Levels")
    ax_price.legend(loc="upper left", fontsize=8)

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
    parser.add_argument("--source", choices=["mt5", "twelvedata"], default="mt5")
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
    parser.add_argument("--zigzag-pct", type=float, default=1.0,
                         help="Minimum %% reversal to confirm an Elliott Wave swing pivot (default 1.0)")
    parser.add_argument("--gann-unit", type=float, default=None,
                         help="Price move per bar for the Gann 1x1 angle (default: auto from ATR14)")
    parser.add_argument("--gann-rings", type=int, default=3,
                         help="Number of Square-of-9 rings to compute (default 3)")
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
    else:
        df = fetch_price_data_twelvedata(
            args.symbol or "XAU/USD", args.interval or "1h", args.lookback, api_key=args.api_key
        )

    df = compute_indicators(df)
    signals = generate_signals(df)
    print_summary(df, signals)

    swings = find_swings(df, args.zigzag_pct)
    ew = analyze_elliott_wave(swings)

    gann_pivot = None
    angle_unit = args.gann_unit
    cycles = None
    if not swings.empty:
        last_swing = swings.iloc[-1]
        gann_pivot = (swings.index[-1], float(last_swing["price"]), last_swing["kind"])
        if angle_unit is None:
            atr_val = df["atr_14"].iloc[-1]
            angle_unit = float(atr_val) if pd.notna(atr_val) and atr_val > 0 else float(df["close"].iloc[-1]) * 0.001
        bar_interval = df.index.to_series().diff().median()
        cycles = gann_time_cycles(gann_pivot[0], bar_interval)

    sq9_levels = gann_square_of_9_levels(df["close"].iloc[-1], args.gann_rings)
    print_elliott_gann(df, swings, ew, gann_pivot, angle_unit or 0.0, sq9_levels, cycles)

    df.to_csv(args.data_output)
    print(f"\nData saved to {args.data_output}")
    plot_chart(df, args.chart_output, ew=ew, gann_pivot=gann_pivot, angle_unit=angle_unit, sq9_levels=sq9_levels)


if __name__ == "__main__":
    main()
