import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from london_session import SessionConfig, run_session_backtest, to_london_time


def test_timezones_follow_bst_and_mt5_server_time():
    summer = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-07-01 07:00"]))
    winter = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-01-15 08:00"]))
    assert to_london_time(summer, "UTC").index[0].hour == 8  # BST = UTC+1
    assert to_london_time(winter, "UTC").index[0].hour == 8  # GMT = UTC
    # MT5 server time (NY+7h): 10:00 server in July = 03:00 NY = 08:00 London
    assert to_london_time(summer.set_axis(pd.DatetimeIndex(["2026-07-01 10:00"])), "mt5").index[0].hour == 8


def _day(bars, start="2026-01-15 00:00"):
    """Build a London-time M15 day: flat Asian range 100-110, then the given (high, low, close) bars from 07:00."""
    asia = [(110, 100, 105)] * 28  # 00:00-06:45
    rows = asia + bars
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="Europe/London")
    df = pd.DataFrame(rows, columns=["high", "low", "close"], index=idx)
    df["open"] = df["close"].shift(1).fillna(105)
    return df[["open", "high", "low", "close"]]


CFG = SessionConfig(spread=0.0, stop_buffer_atr=0.0, min_range_atr=0.0)


def test_sweep_of_high_then_reclaim_goes_short_to_range_low():
    pre = [(106, 104, 105)] * 4                 # 07:00-07:45 quiet
    sweep = [(113, 107, 108)]                    # 08:00 wick above 110, closes back inside -> short @108
    fall = [(108, 99, 100)]                      # hits target 100
    trades = run_session_backtest(_day(pre + sweep + fall), CFG)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert (t.side, t.entry, t.stop, t.target, t.reason) == ("SHORT", 108, 113, 100, "target")
    assert t.r == (108 - 100) / 5


def test_stop_assumed_first_when_bar_hits_both():
    pre = [(106, 104, 105)] * 4
    sweep = [(113, 107, 108)]
    both = [(114, 99, 105)]                      # touches stop 113 and target 100 in one bar
    t = run_session_backtest(_day(pre + sweep + both), CFG).iloc[0]
    assert t.reason == "stop" and t.r == -1


def test_no_entries_outside_trade_window():
    pre = [(106, 104, 105)] * 24                 # quiet until 12:45
    sweep = [(113, 107, 108)]                    # 13:00 - after the 12:00 entry cutoff
    assert run_session_backtest(_day(pre + sweep + [(108, 99, 100)]), CFG).empty
