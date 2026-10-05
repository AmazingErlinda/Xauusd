import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import XAUUSD_CONTRACT_SIZE, backtest, hurst_exponent, run_quant_analysis, score_to_position, trade_plan
from xauusd_analysis import compute_indicators


def make_ohlc(n=1500, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    close = 3000 * np.exp(np.cumsum(drift + rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def test_backtest_has_no_lookahead():
    df = make_ohlc()
    # A "perfect" signal that peeks at the next bar must not be rewarded:
    # positions decided on bar t only earn bar t+1's return.
    future_sign = np.sign(df["close"].pct_change().shift(-1)).fillna(0)
    same_bar_sign = np.sign(df["close"].pct_change()).fillna(0)
    peek = backtest(df, future_sign, spread=0)
    honest = backtest(df, same_bar_sign, spread=0)
    assert peek.metrics["total_return"] > 1.0  # peeking at t+1 is (correctly) huge
    assert abs(honest.metrics["sharpe"]) < 3  # no free lunch on a random walk


def test_costs_reduce_returns():
    df = make_ohlc()
    pos = pd.Series(np.where(np.arange(len(df)) % 10 < 5, 1.0, -1.0), index=df.index)
    assert backtest(df, pos, spread=1.0).metrics["total_return"] < backtest(df, pos, spread=0).metrics["total_return"]


def test_score_bounded_and_trend_detected():
    df = compute_indicators(make_ohlc(drift=0.001))
    out, plan, result, _ = run_quant_analysis(df)
    score = out["score"].dropna()
    assert score.between(-1, 1).all()
    assert score.tail(500).mean() > 0  # strong uptrend should lean long
    assert result.metrics["bars"] == len(df)


def test_position_hysteresis():
    s = pd.Series([0.0, 0.3, 0.1, 0.04, -0.3, -0.1, np.nan])
    assert score_to_position(s).tolist() == [0, 1, 1, 0, -1, -1, 0]


def test_trade_plan_risk_within_budget():
    df = compute_indicators(make_ohlc(drift=0.001))
    out, _, _, _ = run_quant_analysis(df)
    plan = trade_plan(out, equity=50_000, risk_pct=1.0, entry_threshold=0.0)
    assert plan.direction in {"LONG", "SHORT"}
    assert plan.risk_usd <= 500 + 1e-6
    assert plan.risk_usd == round(plan.lots * plan.risk_per_oz * XAUUSD_CONTRACT_SIZE, 2)


def test_hurst_random_walk_near_half():
    rng = np.random.default_rng(1)
    walk = pd.Series(3000 * np.exp(np.cumsum(rng.normal(0, 0.003, 5000))))
    assert 0.4 < hurst_exponent(walk) < 0.6
