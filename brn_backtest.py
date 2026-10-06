#!/usr/bin/env python3
"""Big Round Number (BRN) backtest for XAUUSD.

Setup (buy side, mirrored for sells):
  1. Levels are the round numbers every --step dollars (default 100: ..., 4000, 4100, 4200).
  2. A level is armed once a bar closes at least --arm-distance above it, so
     the buy only triggers when price comes DOWN into the number.
  3. Buy-limit at the level. If a bar gaps below it, the fill is the bar open.
  4. Stop --stop dollars below the level (default 25: buy 4100, stop 4075).
  5. Target a fixed R multiple of the risk (default 2R = +50 from 4100).
  6. One open trade at a time; the level re-arms after price moves back above it.

Fills are simulated on OHLC bars, so use the lowest timeframe you can export
(M5/M15). If a bar touches both stop and target, the stop is assumed first;
an entry bar that also reaches the stop counts as a loss.

Results are in R multiples. MFE (how far price ran in your favour before the
exit, in $) shows where a better target would have been.

Usage:
    python brn_backtest.py --csv-path data/XAUUSD_M15.csv
    python brn_backtest.py --csv-path data/XAUUSD_M15.csv --levels 4100 --stop 25 --target 2
    python brn_backtest.py --csv-path data/XAUUSD_M15.csv --side both --step 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from london_session import r_stats
from xauusd_analysis import fetch_price_data_mt5, fetch_price_data_twelvedata, load_price_data_csv

TARGET_COMPARISON = [1.0, 1.5, 2.0, 3.0, 4.0]


@dataclass(frozen=True)
class BrnConfig:
    step: float = 100.0           # distance between round numbers
    levels: tuple[float, ...] = ()  # explicit levels; empty = every multiple of step
    stop: float = 25.0            # stop distance from the level, $
    target: float = 2.0           # take profit in R multiples of the actual risk
    arm_distance: float = 10.0    # price must close this far beyond the level to arm it
    side: str = "buy"             # "buy" | "sell" | "both"
    max_bars: int = 0             # close a trade after this many bars (0 = no limit)
    spread: float = 0.30          # round-trip cost in USD/oz


def _levels_in_range(cfg: BrnConfig, lo: float, hi: float) -> np.ndarray:
    if cfg.levels:
        return np.array(sorted(cfg.levels), dtype=float)
    first = np.floor(lo / cfg.step) * cfg.step
    return np.arange(first, hi + cfg.step, cfg.step)


def run_brn_backtest(df: pd.DataFrame, cfg: BrnConfig) -> pd.DataFrame:
    levels = _levels_in_range(cfg, df["low"].min(), df["high"].max())
    sides = {"buy": [1], "sell": [-1], "both": [1, -1]}[cfg.side]
    armed = {(lvl, s): False for lvl in levels for s in sides}
    trades: list[dict] = []
    pos: dict | None = None

    bars = df[["open", "high", "low", "close"]].itertuples()
    for i, (ts, o, h, l, c) in enumerate(bars):
        if pos is not None:
            s = pos["_side"]
            pos["_mfe"] = max(pos["_mfe"], s * ((h if s > 0 else l) - pos["entry"]))
            stopped = l <= pos["stop"] if s > 0 else h >= pos["stop"]
            hit = h >= pos["target"] if s > 0 else l <= pos["target"]
            timed_out = cfg.max_bars and i - pos["_i"] >= cfg.max_bars
            if stopped or hit or timed_out:
                # Both touched in one bar: assume the stop came first (conservative).
                exit_, reason = ((pos["stop"], "stop") if stopped else
                                 (pos["target"], "target") if hit else (c, "time"))
                pos.update(exit_time=ts, exit=exit_, reason=reason)
                trades.append(pos)
                pos = None
        else:
            # Buy levels are tried highest first, sell levels lowest first: the
            # first level price trades into on the way is the one that fills.
            for lvl, s in sorted(armed, key=lambda k: -k[1] * k[0]):
                if not armed[(lvl, s)] or not (l <= lvl if s > 0 else h >= lvl):
                    continue
                armed[(lvl, s)] = False  # must re-arm before this level trades again
                entry = min(o, lvl) if s > 0 else max(o, lvl)
                stop = lvl - s * cfg.stop
                risk = s * (entry - stop)
                if risk <= 0:  # gapped through the stop: no fill worth taking
                    continue
                pos = dict(entry_time=ts, side="BUY" if s > 0 else "SELL", level=lvl,
                           entry=entry, stop=stop, target=entry + s * cfg.target * risk, risk=risk,
                           _side=s, _i=i, _mfe=max(0.0, s * (c - entry)))  # bar extreme came before the fill
                # Same bar also reaches the stop: count it as a loss.
                if (l <= stop) if s > 0 else (h >= stop):
                    pos.update(exit_time=ts, exit=stop, reason="stop")
                    trades.append(pos)
                    pos = None
                break

        # Arm on the close, so a level can only fill from the next bar on.
        for lvl, s in armed:
            if s * (c - lvl) >= cfg.arm_distance:
                armed[(lvl, s)] = True

    if pos is not None:
        pos.update(exit_time=df.index[-1], exit=df["close"].iloc[-1], reason="open")
        trades.append(pos)

    for t in trades:
        t["pnl_per_oz"] = t["_side"] * (t["exit"] - t["entry"]) - cfg.spread
        t["r"] = t["pnl_per_oz"] / t["risk"]
        t["mfe"] = t.pop("_mfe")
        del t["_side"], t["_i"]
    cols = ["entry_time", "exit_time", "side", "level", "entry", "stop", "target", "exit",
            "reason", "risk", "mfe", "pnl_per_oz", "r"]
    return pd.DataFrame(trades, columns=cols)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _row(name: str, s: dict) -> str:
    if s["trades"] == 0:
        return f"{name:<14}{0:>7}"
    pf = s["profit_factor"]
    return (f"{name:<14}{s['trades']:>7}{s['win_rate'] * 100:>7.1f}%{s['avg_r']:>+8.3f}"
            f"{s['total_r']:>+9.1f}{'n/a' if np.isnan(pf) else f'{pf:.2f}':>7}{s['max_dd_r']:>8.1f}")


HEADER = f"{'':<14}{'trades':>7}{'win%':>8}{'avg R':>8}{'total R':>9}{'PF':>7}{'maxDD R':>8}"


def print_report(df: pd.DataFrame, cfg: BrnConfig, trades: pd.DataFrame) -> None:
    print(f"XAUUSD BRN backtest  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  ({len(df)} bars)")
    lv = ", ".join(f"{x:g}" for x in cfg.levels) if cfg.levels else f"every {cfg.step:g}"
    print(f"Levels {lv} | side {cfg.side} | stop {cfg.stop:g} | target {cfg.target:g}R | "
          f"arm {cfg.arm_distance:g} | spread ${cfg.spread}/oz")
    print("=" * 72)
    closed = trades[trades["reason"] != "open"]
    s = r_stats(closed)
    print(HEADER)
    print(_row("all", s))
    if s["trades"] == 0:
        return
    for side, g in closed.groupby("side"):
        print(_row(side.lower(), r_stats(g)))
    for year, g in closed.groupby(pd.to_datetime(closed["entry_time"].astype(str).str[:10]).dt.year):
        print(_row(str(year), r_stats(g)))

    print("-" * 72)
    print("Per level (most traded first):")
    by_level = closed.groupby("level")
    for lvl in by_level.size().sort_values(ascending=False).index[:10]:
        print(_row(f"{lvl:g}", r_stats(by_level.get_group(lvl))))

    print("-" * 72)
    print("How far price ran in your favour after the fill (MFE, $):")
    for q in (0.25, 0.5, 0.75):
        print(f"  {int(q * 100)}% of trades reached at least +{closed['mfe'].quantile(1 - q):.1f}")
    stops = closed[closed["reason"] == "stop"]
    if len(stops):
        print(f"  losers ran +{stops['mfe'].median():.1f} (median) before the stop - "
              f"{(stops['mfe'] >= cfg.stop).mean() * 100:.0f}% were up {cfg.stop:g}+ (1R) first")
    if trades["reason"].eq("open").any():
        print(f"  1 trade still open at the end of the data (not counted).")


def print_target_comparison(df: pd.DataFrame, cfg: BrnConfig) -> None:
    print("-" * 72)
    print(f"Same entries, different targets (stop {cfg.stop:g}):")
    print(HEADER)
    for tgt in TARGET_COMPARISON:
        tr = run_brn_backtest(df, replace(cfg, target=tgt))
        print(_row(f"{tgt:g}R (+{tgt * cfg.stop:g})", r_stats(tr[tr["reason"] != "open"])))


def main() -> None:
    p = argparse.ArgumentParser(description="Big Round Number XAUUSD backtest")
    p.add_argument("--source", choices=["mt5", "twelvedata", "csv"], default="csv")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--interval", default=None, help="Defaults to M15 for mt5, 15min for twelvedata")
    p.add_argument("--lookback", type=int, default=30000)
    p.add_argument("--step", type=float, default=100.0, help="Round-number spacing, e.g. 100 or 50")
    p.add_argument("--levels", type=float, nargs="*", default=(), help="Only trade these levels, e.g. 4100")
    p.add_argument("--stop", type=float, default=25.0, help="Stop distance from the level in $")
    p.add_argument("--target", type=float, default=2.0, help="Take profit in R multiples")
    p.add_argument("--arm-distance", type=float, default=10.0)
    p.add_argument("--side", choices=["buy", "sell", "both"], default="buy")
    p.add_argument("--max-bars", type=int, default=0)
    p.add_argument("--spread", type=float, default=0.30, help="Round-trip cost in USD/oz")
    p.add_argument("--trades-output", default="brn_trades.csv")
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

    cfg = BrnConfig(step=args.step, levels=tuple(args.levels), stop=args.stop, target=args.target,
                    arm_distance=args.arm_distance, side=args.side, max_bars=args.max_bars,
                    spread=args.spread)
    trades = run_brn_backtest(df, cfg)
    print_report(df, cfg, trades)
    print_target_comparison(df, cfg)
    trades.to_csv(args.trades_output, index=False)
    print(f"Trades saved to {args.trades_output}")


if __name__ == "__main__":
    main()
