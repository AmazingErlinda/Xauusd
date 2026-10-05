#!/usr/bin/env python3
"""London-session intraday backtest for XAUUSD on M15 (or M5/M30/H1) candles.

Setup: Asian-range sweep and reclaim.
  1. Mark the Asian range (default 00:00-07:00 London time).
  2. After the range closes, watch for price to trade through one side (the sweep).
  3. Enter when a bar inside the trade window closes back inside the range:
     short after a sweep of the high, long after a sweep of the low.
  4. Stop beyond the sweep extreme (+ ATR buffer), target the other side of
     the range (or a fixed R multiple), flat by the flat time.

Outputs:
  - stats in R multiples (risk-normalised, so lot sizing doesn't distort them)
  - the same setup run in other time windows, to test whether London-only is better
  - an hour-by-hour volatility profile of gold in London time
  - trades CSV and a chart

All session times are London local time (BST/GMT handled automatically).

Timezones: TradingView exports are UTC (default). MT5 bars are in broker
server time; most brokers (Vantage included) use New York time + 7h
(GMT+2 winter / GMT+3 summer) - use --data-tz mt5, which is picked
automatically for --source mt5 and for MT5 "Export Bars" CSV files.
Check yours: compare Market Watch server time with UTC.

Usage:
    python london_session.py --source csv --csv-path data/XAUUSD_M15.csv
    python london_session.py --source mt5 --interval M15 --lookback 30000
    python london_session.py --source csv --csv-path data/x.csv --target 2 --bias long
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant import atr
from xauusd_analysis import fetch_price_data_mt5, fetch_price_data_twelvedata, load_price_data_csv

LONDON = "Europe/London"

# Windows compared against the London-only setup: (name, trade_start, trade_end, flat_time)
COMPARISON_WINDOWS = [
    ("London AM 08-12", "08:00", "12:00", "16:00"),
    ("London full 08-16", "08:00", "16:00", "20:00"),
    ("NY 13:30-17", "13:30", "17:00", "21:00"),
    ("All day 08-20", "08:00", "20:00", "21:30"),
]


# --------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------

def to_london_time(df: pd.DataFrame, data_tz: str = "UTC") -> pd.DataFrame:
    """Return a copy indexed in London local time.

    data_tz: an IANA zone for naive timestamps (e.g. "UTC"), or "mt5" for
    the common broker server time of New York + 7h.
    """
    idx = df.index
    if idx.tz is None:
        if data_tz == "mt5":
            idx = (idx - pd.Timedelta(hours=7)).tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="shift_forward"
            )
        else:
            idx = idx.tz_localize(data_tz, ambiguous="NaT", nonexistent="shift_forward")
    out = df.copy()
    out.index = idx.tz_convert(LONDON)
    return out[out.index.notna()]


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.hour * 60 + idx.minute).to_numpy()


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    asia_start: str = "00:00"
    asia_end: str = "07:00"
    trade_start: str = "08:00"
    trade_end: str = "12:00"     # no new entries at/after this time
    flat_time: str = "16:00"     # any open trade is closed by this time
    stop_buffer_atr: float = 0.25  # extra stop distance beyond the sweep, in bar ATRs
    target: str = "range"        # "range" (other side of Asian range) or an R multiple like "2"
    max_trades_per_day: int = 1
    spread: float = 0.30         # round-trip cost in USD/oz
    bias: str = "both"           # "both" | "long" | "short"
    min_range_atr: float = 2.0   # skip days whose Asian range is < this many bar ATRs


def _simulate_day(day: pd.DataFrame, cfg: SessionConfig) -> list[dict]:
    mins = _minute_of_day(day.index)
    a0, a1 = _minutes(cfg.asia_start), _minutes(cfg.asia_end)
    s0, s1, flat = _minutes(cfg.trade_start), _minutes(cfg.trade_end), _minutes(cfg.flat_time)

    asia = day[(mins >= a0) & (mins < a1)]
    if len(asia) < 4:
        return []
    hi, lo = asia["high"].max(), asia["low"].min()
    bar_atr = day["atr"].iloc[len(asia) - 1]
    if np.isnan(bar_atr) or (hi - lo) < cfg.min_range_atr * bar_atr:
        return []

    live = day[(mins >= a1) & (mins < flat)]
    trades: list[dict] = []
    sweep_hi = sweep_lo = None
    pos = None  # dict for the open trade

    for ts, o, h, l, c, a in live[["open", "high", "low", "close", "atr"]].itertuples():
        m = ts.hour * 60 + ts.minute

        if pos is not None:
            if pos["_side"] > 0:
                stopped, hit = l <= pos["stop"], h >= pos["target"]
            else:
                stopped, hit = h >= pos["stop"], l <= pos["target"]
            if stopped or hit:
                # Both touched in one bar: assume the stop came first (conservative).
                pos.update(exit_time=ts, exit=pos["stop"] if stopped else pos["target"],
                           reason="stop" if stopped else "target")
                trades.append(pos)
                pos = None
            continue

        if len(trades) >= cfg.max_trades_per_day:
            break

        if h > hi:
            sweep_hi = h if sweep_hi is None else max(sweep_hi, h)
        if l < lo:
            sweep_lo = l if sweep_lo is None else min(sweep_lo, l)

        if not (s0 <= m < s1):
            continue

        side = 0
        if sweep_hi is not None and lo < c < hi and cfg.bias in ("both", "short"):
            side, stop = -1, sweep_hi + cfg.stop_buffer_atr * a
        elif sweep_lo is not None and lo < c < hi and cfg.bias in ("both", "long"):
            side, stop = 1, sweep_lo - cfg.stop_buffer_atr * a
        if side == 0:
            continue

        risk = abs(c - stop)
        target = (lo if side < 0 else hi) if cfg.target == "range" else c + side * float(cfg.target) * risk
        if risk <= 0 or side * (target - c) <= 0:
            continue
        pos = dict(date=ts.date(), entry_time=ts, side="LONG" if side > 0 else "SHORT",
                   entry=c, stop=stop, target=target, risk=risk, asia_high=hi, asia_low=lo)
        pos["_side"] = side
        sweep_hi = sweep_lo = None

    if pos is not None:  # flat at the last bar before the flat time
        pos.update(exit_time=live.index[-1], exit=live["close"].iloc[-1], reason="time")
        trades.append(pos)

    for t in trades:
        t["pnl_per_oz"] = t["_side"] * (t["exit"] - t["entry"]) - cfg.spread
        t["r"] = t["pnl_per_oz"] / t["risk"]
        del t["_side"]
    return trades


def run_session_backtest(df_london: pd.DataFrame, cfg: SessionConfig) -> pd.DataFrame:
    """Run the setup over every London trading day. df_london must be London-indexed."""
    df = df_london.copy()
    df["atr"] = atr(df, 14)
    rows: list[dict] = []
    for _, day in df.groupby(df.index.date):
        rows.extend(_simulate_day(day, cfg))
    cols = ["date", "entry_time", "exit_time", "side", "entry", "stop", "target", "exit",
            "reason", "risk", "asia_high", "asia_low", "pnl_per_oz", "r"]
    return pd.DataFrame(rows, columns=cols)


def r_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    r = trades["r"]
    equity = r.cumsum()
    gross_win, gross_loss = r[r > 0].sum(), -r[r <= 0].sum()
    return {
        "trades": n,
        "win_rate": (r > 0).mean(),
        "avg_r": r.mean(),
        "total_r": r.sum(),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("nan"),
        "max_dd_r": (equity - equity.cummax()).min(),
        "t_stat": r.mean() / r.std() * np.sqrt(n) if n > 1 and r.std() > 0 else float("nan"),
        "avg_pnl_per_oz": trades["pnl_per_oz"].mean(),
    }


def hourly_profile(df_london: pd.DataFrame) -> pd.DataFrame:
    """Average bar range and absolute move per London hour."""
    hours = df_london.index.hour
    return pd.DataFrame({
        "avg_bar_range": (df_london["high"] - df_london["low"]).groupby(hours).mean(),
        "avg_abs_move": (df_london["close"] - df_london["open"]).abs().groupby(hours).mean(),
        "bars": df_london["close"].groupby(hours).size(),
    }).rename_axis("london_hour")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _f(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.2f}"


def print_report(df: pd.DataFrame, cfg: SessionConfig, trades: pd.DataFrame,
                 comparison: list[tuple[str, dict]], profile: pd.DataFrame) -> None:
    s = r_stats(trades)
    print(f"XAUUSD session backtest  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  "
          f"({df.index.normalize().nunique()} days, London time)")
    print(f"Setup: Asian range {cfg.asia_start}-{cfg.asia_end} sweep & reclaim | entries "
          f"{cfg.trade_start}-{cfg.trade_end} | flat {cfg.flat_time} | target {cfg.target} | "
          f"bias {cfg.bias} | spread ${cfg.spread}/oz")
    print("=" * 72)
    if s["trades"] == 0:
        print("No trades triggered.")
    else:
        print(f"{'trades':>16}: {s['trades']}")
        print(f"{'win rate':>16}: {_f(s['win_rate'], True)}")
        print(f"{'avg R':>16}: {s['avg_r']:+.3f}  (t-stat {_f(s['t_stat'])})")
        print(f"{'total R':>16}: {s['total_r']:+.1f}")
        print(f"{'profit factor':>16}: {_f(s['profit_factor'])}")
        print(f"{'max drawdown':>16}: {s['max_dd_r']:.1f} R")
        print(f"{'avg trade':>16}: {s['avg_pnl_per_oz']:+.2f} $/oz")
        by_side = trades.groupby("side")["r"].agg(["count", "mean", "sum"])
        by_exit = trades["reason"].value_counts()
        print(f"{'by side':>16}: " + " | ".join(
            f"{k} {int(v['count'])} trades, avg {v['mean']:+.2f}R" for k, v in by_side.iterrows()))
        print(f"{'exits':>16}: " + ", ".join(f"{k} {v}" for k, v in by_exit.items()))

    print("-" * 72)
    print("Window comparison (same setup, different entry windows):")
    print(f"{'window':<20}{'trades':>8}{'win%':>8}{'avg R':>9}{'total R':>10}{'PF':>7}{'maxDD R':>9}{'t':>7}")
    for name, st in comparison:
        if st["trades"] == 0:
            print(f"{name:<20}{0:>8}")
            continue
        print(f"{name:<20}{st['trades']:>8}{st['win_rate'] * 100:>7.1f}%{st['avg_r']:>+9.3f}"
              f"{st['total_r']:>+10.1f}{_f(st['profit_factor']):>7}{st['max_dd_r']:>9.1f}{_f(st['t_stat']):>7}")

    print("-" * 72)
    print("Gold volatility by London hour (avg bar range, $):")
    peak = profile["avg_bar_range"].max()
    for hour, row in profile.iterrows():
        bar = "#" * int(round(30 * row["avg_bar_range"] / peak)) if peak > 0 else ""
        print(f"  {hour:02d}:00  {row['avg_bar_range']:6.2f}  {bar}")

    print("-" * 72)
    if s["trades"] and (s["trades"] < 100 or abs(s.get("t_stat", 0) or 0) < 2):
        print("note: fewer than 100 trades or |t| < 2 - the edge is not statistically "
              "distinguishable from luck yet. Use more history before trusting it.")


def plot_report(trades_by_window: dict[str, pd.DataFrame], profile: pd.DataFrame, path: str) -> None:
    fig, (ax_eq, ax_hr) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    for name, tr in trades_by_window.items():
        if len(tr):
            ax_eq.plot(pd.to_datetime(tr["exit_time"].astype(str).str[:19]), tr["r"].cumsum(),
                       label=name, linewidth=1.8 if name.startswith("London AM") else 1)
    ax_eq.axhline(0, color="black", linewidth=0.6)
    ax_eq.set_ylabel("Cumulative R")
    ax_eq.set_title("Asian range sweep & reclaim - equity by entry window")
    ax_eq.legend(loc="upper left")

    colors = ["tab:blue" if 8 <= h < 12 else "lightgray" for h in profile.index]
    ax_hr.bar(profile.index, profile["avg_bar_range"], color=colors)
    ax_hr.set_xticks(range(0, 24, 2))
    ax_hr.set_xlabel("London hour (blue = 08-12 trade window)")
    ax_hr.set_ylabel("Avg bar range ($)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="London-session XAUUSD intraday backtest")
    p.add_argument("--source", choices=["mt5", "twelvedata", "csv"], default="csv")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--interval", default=None, help="Defaults to M15 for mt5, 15min for twelvedata")
    p.add_argument("--lookback", type=int, default=30000, help="Bars to fetch (M15: ~25k per year)")
    p.add_argument("--data-tz", default="auto",
                   help='Timezone of naive timestamps: "auto", "UTC", "mt5" (NY+7h server time) or an IANA zone')
    p.add_argument("--trade-start", default="08:00")
    p.add_argument("--trade-end", default="12:00")
    p.add_argument("--flat-time", default="16:00")
    p.add_argument("--asia-start", default="00:00")
    p.add_argument("--asia-end", default="07:00")
    p.add_argument("--target", default="range", help='"range" or an R multiple, e.g. 2')
    p.add_argument("--bias", choices=["both", "long", "short"], default="both")
    p.add_argument("--stop-buffer-atr", type=float, default=0.25)
    p.add_argument("--max-trades-per-day", type=int, default=1)
    p.add_argument("--spread", type=float, default=0.30, help="Round-trip cost in USD/oz")
    p.add_argument("--trades-output", default="london_session_trades.csv")
    p.add_argument("--chart-output", default="london_session_chart.png")
    args = p.parse_args()

    if args.source == "mt5":
        df = fetch_price_data_mt5(args.symbol or "XAUUSD", args.interval or "M15", args.lookback)
    elif args.source == "twelvedata":
        df = fetch_price_data_twelvedata(args.symbol or "XAU/USD", args.interval or "15min",
                                         min(args.lookback, 5000))
    else:
        if not args.csv_path:
            p.error("--csv-path is required with --source csv")
        df = load_price_data_csv(args.csv_path)

    data_tz = args.data_tz
    if data_tz == "auto":
        data_tz = "mt5" if args.source == "mt5" or df.attrs.get("format") == "mt5" else "UTC"
    print(f"Interpreting timestamps as: {data_tz}")
    df = to_london_time(df, data_tz)

    cfg = SessionConfig(
        asia_start=args.asia_start, asia_end=args.asia_end,
        trade_start=args.trade_start, trade_end=args.trade_end, flat_time=args.flat_time,
        stop_buffer_atr=args.stop_buffer_atr, target=args.target,
        max_trades_per_day=args.max_trades_per_day, spread=args.spread, bias=args.bias,
    )
    trades = run_session_backtest(df, cfg)

    trades_by_window = {}
    for name, start, end, flat in COMPARISON_WINDOWS:
        trades_by_window[name] = run_session_backtest(
            df, replace(cfg, trade_start=start, trade_end=end, flat_time=flat))
    comparison = [(name, r_stats(tr)) for name, tr in trades_by_window.items()]
    profile = hourly_profile(df)

    print_report(df, cfg, trades, comparison, profile)
    trades.to_csv(args.trades_output, index=False)
    print(f"Trades saved to {args.trades_output}")
    plot_report(trades_by_window, profile, args.chart_output)


if __name__ == "__main__":
    main()
