import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brn_backtest import BrnConfig, run_brn_backtest


def _bars(rows):
    """(open, high, low, close) rows on an M15 index."""
    idx = pd.date_range("2026-10-06 00:00", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


CFG = BrnConfig(levels=(4100,), stop=25, target=2, spread=0.0)


def test_buy_4100_stop_4075_hits_2r_target():
    df = _bars([
        (4125, 4130, 4120, 4124),   # closes 24 above 4100 -> level armed
        (4124, 4126, 4098, 4105),   # trades down into 4100 -> buy-limit filled
        (4105, 4152, 4104, 4150),   # target 4100 + 2 * 25 = 4150
    ])
    t = run_brn_backtest(df, CFG).iloc[0]
    assert (t.side, t.entry, t.stop, t.target, t.exit, t.reason) == ("BUY", 4100, 4075, 4150, 4150, "target")
    assert t.r == 2


def test_stop_at_4075_is_minus_one_r():
    df = _bars([(4125, 4130, 4120, 4124), (4124, 4126, 4098, 4102), (4102, 4110, 4070, 4072)])
    t = run_brn_backtest(df, CFG).iloc[0]
    assert (t.exit, t.reason, t.r, t.mfe) == (4075, "stop", -1, 10)


def test_no_buy_when_price_comes_from_below():
    df = _bars([(4080, 4095, 4078, 4090), (4090, 4104, 4088, 4102), (4102, 4130, 4100, 4128)])
    assert run_brn_backtest(df, CFG).empty


def test_entry_bar_through_the_stop_counts_as_loss():
    df = _bars([(4125, 4130, 4120, 4124), (4124, 4125, 4060, 4065)])
    t = run_brn_backtest(df, CFG).iloc[0]
    assert (t.entry, t.reason, t.r) == (4100, "stop", -1)


def test_gap_below_level_fills_at_open_with_smaller_risk():
    df = _bars([(4125, 4130, 4120, 4124), (4090, 4140, 4088, 4135)])
    t = run_brn_backtest(df, CFG).iloc[0]
    assert (t.entry, t.risk, t.target) == (4090, 15, 4120)


def test_level_must_rearm_before_second_trade():
    df = _bars([
        (4125, 4130, 4120, 4124), (4124, 4126, 4098, 4105), (4105, 4152, 4104, 4150),  # trade 1 wins
        (4150, 4151, 4099, 4103),   # dips to 4100 again right away: re-armed by 4150 close -> trade 2
    ])
    assert len(run_brn_backtest(df, CFG)) == 2
    no_rearm = _bars([
        (4125, 4130, 4120, 4124), (4124, 4126, 4098, 4105), (4105, 4106, 4070, 4072),  # stopped
        (4072, 4105, 4071, 4104),   # back up but closes only 4 above 4100 -> not armed
        (4104, 4106, 4099, 4101),
    ])
    assert len(run_brn_backtest(no_rearm, CFG)) == 1


def test_every_hundred_levels_by_default():
    df = _bars([(4215, 4220, 4212, 4214), (4214, 4215, 4195, 4198)])
    t = run_brn_backtest(df, BrnConfig(spread=0.0)).iloc[0]
    assert (t.level, t.stop) == (4200, 4175)
