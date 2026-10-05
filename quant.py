"""Quantitative layer for XAUUSD technical analysis.

Builds on the classic indicators in xauusd_analysis.py with:
  - volatility / trend-strength features (ATR, ADX, stochastic, z-score,
    realized volatility, Hurst exponent)
  - regime detection (trending vs ranging, high vs low volatility)
  - a regime-weighted composite score in [-1, 1]
  - an ATR-based trade plan with position sizing
  - a vectorized backtest (no lookahead: positions act on the next bar)
    with Sharpe, Sortino, max drawdown, win rate and profit factor

Everything operates on an OHLCV DataFrame indexed by timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Standard gold contract: 1 lot = 100 troy ounces, so a $1 move = $100 per lot.
XAUUSD_CONTRACT_SIZE = 100


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_ = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({
        "adx": dx.ewm(alpha=1 / period, adjust=False).mean(),
        "plus_di": plus_di,
        "minus_di": minus_di,
    })


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d_period).mean()})


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    return (series - series.rolling(period).mean()) / series.rolling(period).std()


def realized_vol(close: pd.Series, period: int = 20, periods_per_year: float = 252) -> pd.Series:
    """Annualized rolling volatility of log returns."""
    log_ret = np.log(close).diff()
    return log_ret.rolling(period).std() * np.sqrt(periods_per_year)


def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """Estimate the Hurst exponent from the scaling of lagged log-price differences.

    H ~ 0.5 random walk, H > 0.5 trending/persistent, H < 0.5 mean-reverting.
    """
    log_p = np.log(series.dropna().to_numpy())
    max_lag = min(max_lag, len(log_p) // 4)
    if max_lag < 3:
        return float("nan")
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(log_p[lag:] - log_p[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(slope)


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year for annualizing, assuming ~252 trading days of ~23h for gold."""
    if len(index) < 2:
        return 252.0
    step = pd.Series(index).diff().median().total_seconds()
    day = 86400
    if step >= day:
        return 252.0 * day / step
    return 252.0 * 23 * 3600 / step


def compute_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add quant features. Expects the columns produced by compute_indicators()."""
    out = df.copy()
    ppy = infer_periods_per_year(out.index)

    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / out["close"]
    out = out.join(adx(out, 14))
    out = out.join(stochastic(out))
    out["zscore_20"] = zscore(out["close"], 20)
    out["roc_10"] = out["close"].pct_change(10)
    out["realized_vol_20"] = realized_vol(out["close"], 20, ppy)
    # Percentile of current vol vs. its own trailing 100-bar history.
    out["vol_rank"] = out["realized_vol_20"].rolling(100, min_periods=20).rank(pct=True)
    return out


# --------------------------------------------------------------------------
# Regime + composite score
# --------------------------------------------------------------------------

def classify_regime(df: pd.DataFrame, adx_trend: float = 25.0, vol_high: float = 0.8) -> pd.Series:
    trend = np.where(df["adx"] >= adx_trend, "trending", "ranging")
    vol = np.where(df["vol_rank"] >= vol_high, "high-vol", "normal-vol")
    return pd.Series([f"{t}/{v}" for t, v in zip(trend, vol)], index=df.index)


def composite_score(df: pd.DataFrame, adx_trend: float = 25.0) -> pd.DataFrame:
    """Regime-weighted score in [-1, 1]. Positive = long bias, negative = short bias.

    Components (each squashed to [-1, 1]):
      trend      - EMA20 vs SMA50 distance in ATRs, plus DI direction
      momentum   - MACD histogram in ATRs and 10-bar rate of change
      reversion  - inverted z-score and RSI distance from 50 (fades extremes)

    In trending regimes trend/momentum dominate; in ranging regimes the
    mean-reversion component dominates. High-volatility regimes scale the
    score down to reduce exposure.
    """
    atr_ = df["atr_14"].replace(0, np.nan)

    trend = np.tanh((df["ema_20"] - df["sma_50"]) / atr_ / 2)
    di = np.tanh((df["plus_di"] - df["minus_di"]) / 20)
    trend_score = 0.6 * trend + 0.4 * di

    momentum_score = 0.5 * np.tanh(df["macd_hist"] / atr_ * 2) + 0.5 * np.tanh(
        df["roc_10"] / (df["atr_pct"] * np.sqrt(10)).replace(0, np.nan)
    )

    reversion_score = 0.5 * np.tanh(-df["zscore_20"] / 2) + 0.5 * np.tanh(-(df["rsi_14"] - 50) / 20)

    # Smooth regime weight: 0 in a dead range, 1 in a strong trend.
    w_trend = ((df["adx"] - (adx_trend - 10)) / 20).clip(0, 1)
    score = w_trend * (0.55 * trend_score + 0.45 * momentum_score) + (1 - w_trend) * reversion_score

    vol_scale = np.where(df["vol_rank"] >= 0.8, 0.5, 1.0)
    score = (score * vol_scale).clip(-1, 1)

    return pd.DataFrame({
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "reversion_score": reversion_score,
        "trend_weight": w_trend,
        "score": score,
    }, index=df.index)


def score_to_position(score: pd.Series, entry: float = 0.25, exit_: float = 0.05) -> pd.Series:
    """Map score to {-1, 0, 1} with hysteresis: enter beyond +/-entry, flatten inside +/-exit_."""
    pos = np.zeros(len(score))
    current = 0.0
    for i, s in enumerate(score.to_numpy()):
        if np.isnan(s):
            current = 0.0
        elif s >= entry:
            current = 1.0
        elif s <= -entry:
            current = -1.0
        elif abs(s) <= exit_:
            current = 0.0
        pos[i] = current
    return pd.Series(pos, index=score.index)


# --------------------------------------------------------------------------
# Trade plan
# --------------------------------------------------------------------------

@dataclass
class TradePlan:
    direction: str  # "LONG" | "SHORT" | "FLAT"
    entry: float
    stop: float
    take_profit: float
    risk_per_oz: float
    lots: float
    risk_usd: float
    reward_risk: float


def trade_plan(
    df: pd.DataFrame,
    equity: float = 10_000.0,
    risk_pct: float = 1.0,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    entry_threshold: float = 0.25,
    lot_step: float = 0.01,
) -> TradePlan:
    last = df.iloc[-1]
    score = last["score"]
    entry = float(last["close"])
    atr_ = float(last["atr_14"])

    if np.isnan(score) or abs(score) < entry_threshold:
        return TradePlan("FLAT", entry, float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0)

    sign = 1 if score > 0 else -1
    stop = entry - sign * stop_atr * atr_
    tp = entry + sign * tp_atr * atr_
    risk_per_oz = stop_atr * atr_
    risk_budget = equity * risk_pct / 100
    raw_lots = risk_budget / (risk_per_oz * XAUUSD_CONTRACT_SIZE)
    lots = np.floor(raw_lots / lot_step) * lot_step
    return TradePlan(
        direction="LONG" if sign > 0 else "SHORT",
        entry=entry,
        stop=stop,
        take_profit=tp,
        risk_per_oz=risk_per_oz,
        lots=round(float(lots), 2),
        risk_usd=round(float(lots * risk_per_oz * XAUUSD_CONTRACT_SIZE), 2),
        reward_risk=tp_atr / stop_atr,
    )


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    trades: pd.DataFrame
    metrics: dict


def backtest(
    df: pd.DataFrame,
    position: pd.Series,
    spread: float = 0.30,
    periods_per_year: float | None = None,
) -> BacktestResult:
    """Vectorized backtest of a {-1, 0, 1} position series.

    The position decided on bar t's close is held over bar t+1 (shift by one),
    so there is no lookahead. `spread` is the round-trip cost in price units
    (USD per oz); half is charged on each position change.
    """
    ppy = periods_per_year or infer_periods_per_year(df.index)
    close = df["close"]
    ret = close.pct_change().fillna(0)
    held = position.shift(1).fillna(0)

    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * (spread / 2) / close
    strat_ret = held * ret - costs
    equity = (1 + strat_ret).cumprod()

    trades = _extract_trades(close, held, spread)
    metrics = _performance_metrics(strat_ret, equity, held, trades, ret, ppy)
    return BacktestResult(equity, strat_ret, trades, metrics)


def _extract_trades(close: pd.Series, held: pd.Series, spread: float) -> pd.DataFrame:
    rows = []
    side, entry_time, entry_px = 0.0, None, None
    prev_close = close.shift(1)
    for t, pos in held.items():
        if pos != side:
            # Exits/entries execute at the previous bar's close (when the decision was made).
            px = prev_close.loc[t]
            if side != 0:
                pnl = side * (px - entry_px) - spread
                rows.append({"entry_time": entry_time, "exit_time": t, "side": "LONG" if side > 0 else "SHORT",
                             "entry": entry_px, "exit": px, "pnl_per_oz": pnl, "return": pnl / entry_px})
            side, entry_time, entry_px = pos, t, px
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "side", "entry", "exit", "pnl_per_oz", "return"])


def _performance_metrics(
    strat_ret: pd.Series,
    equity: pd.Series,
    held: pd.Series,
    trades: pd.DataFrame,
    market_ret: pd.Series,
    ppy: float,
) -> dict:
    n = len(strat_ret)
    total_return = equity.iloc[-1] - 1 if n else 0.0
    years = n / ppy if ppy else float("nan")
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 and equity.iloc[-1] > 0 else float("nan")

    mu, sd = strat_ret.mean(), strat_ret.std()
    downside = strat_ret[strat_ret < 0].std()
    sharpe = mu / sd * np.sqrt(ppy) if sd > 0 else float("nan")
    sortino = mu / downside * np.sqrt(ppy) if downside and downside > 0 else float("nan")

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else float("nan")

    wins = trades["pnl_per_oz"] > 0 if len(trades) else pd.Series(dtype=bool)
    gross_win = trades.loc[wins, "pnl_per_oz"].sum() if len(trades) else 0.0
    gross_loss = -trades.loc[~wins, "pnl_per_oz"].sum() if len(trades) else 0.0

    return {
        "bars": n,
        "total_return": total_return,
        "buy_hold_return": (1 + market_ret).prod() - 1,
        "cagr": cagr,
        "ann_volatility": sd * np.sqrt(ppy),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "exposure": (held != 0).mean(),
        "trades": len(trades),
        "win_rate": wins.mean() if len(trades) else float("nan"),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("nan"),
        "avg_trade_per_oz": trades["pnl_per_oz"].mean() if len(trades) else float("nan"),
    }


# --------------------------------------------------------------------------
# Orchestration / reporting
# --------------------------------------------------------------------------

def run_quant_analysis(
    df: pd.DataFrame,
    equity: float = 10_000.0,
    risk_pct: float = 1.0,
    spread: float = 0.30,
    entry_threshold: float = 0.25,
) -> tuple[pd.DataFrame, TradePlan, BacktestResult, float]:
    out = compute_quant_features(df)
    out = out.join(composite_score(out))
    out["regime"] = classify_regime(out)
    out["position"] = score_to_position(out["score"], entry=entry_threshold)

    plan = trade_plan(out, equity=equity, risk_pct=risk_pct, entry_threshold=entry_threshold)
    result = backtest(out, out["position"], spread=spread)
    out["equity"] = result.equity_curve
    hurst = hurst_exponent(out["close"])
    return out, plan, result, hurst


def _fmt_pct(x: float, signed: bool = True) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x * 100:+.2f}%" if signed else f"{x * 100:.2f}%"


def _fmt_num(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.2f}"


def print_quant_report(df: pd.DataFrame, plan: TradePlan, result: BacktestResult, hurst: float) -> None:
    last = df.iloc[-1]
    hurst_label = (
        "n/a" if np.isnan(hurst)
        else "trending/persistent" if hurst > 0.55
        else "mean-reverting" if hurst < 0.45
        else "random-walk-like"
    )

    print()
    print("=" * 50)
    print("QUANT ANALYSIS")
    print("=" * 50)
    print(f"{'regime':>16}: {last['regime']}")
    print(f"{'ADX':>16}: {_fmt_num(last['adx'])}  (+DI {_fmt_num(last['plus_di'])} / -DI {_fmt_num(last['minus_di'])})")
    print(f"{'ATR(14)':>16}: {_fmt_num(last['atr_14'])}  ({_fmt_pct(last['atr_pct'], signed=False)} of price)")
    print(f"{'realized vol':>16}: {_fmt_pct(last['realized_vol_20'], signed=False)} ann.  (rank {_fmt_num(last['vol_rank'])})")
    print(f"{'z-score(20)':>16}: {_fmt_num(last['zscore_20'])}")
    print(f"{'stochastic %K':>16}: {_fmt_num(last['stoch_k'])}")
    print(f"{'Hurst':>16}: {_fmt_num(hurst)}  ({hurst_label})")
    print("-" * 50)
    print(f"{'trend score':>16}: {_fmt_num(last['trend_score'])}")
    print(f"{'momentum score':>16}: {_fmt_num(last['momentum_score'])}")
    print(f"{'reversion score':>16}: {_fmt_num(last['reversion_score'])}")
    print(f"{'trend weight':>16}: {_fmt_num(last['trend_weight'])}")
    print(f"{'COMPOSITE':>16}: {_fmt_num(last['score'])}  (-1 strong short .. +1 strong long)")
    print("-" * 50)
    if plan.direction == "FLAT":
        print("Trade plan: FLAT - composite score below entry threshold")
    else:
        print(f"Trade plan: {plan.direction}")
        print(f"{'entry':>16}: {plan.entry:.2f}")
        print(f"{'stop loss':>16}: {plan.stop:.2f}  ({plan.risk_per_oz:.2f} $/oz)")
        print(f"{'take profit':>16}: {plan.take_profit:.2f}  (R:R 1:{plan.reward_risk:.1f})")
        print(f"{'size':>16}: {plan.lots:.2f} lots  (risking ${plan.risk_usd:,.2f})")
    print("-" * 50)
    m = result.metrics
    print(f"Backtest over {m['bars']} bars (score strategy, costs included)")
    print(f"{'total return':>16}: {_fmt_pct(m['total_return'])}  (buy & hold {_fmt_pct(m['buy_hold_return'])})")
    print(f"{'CAGR':>16}: {_fmt_pct(m['cagr'])}")
    print(f"{'ann. volatility':>16}: {_fmt_pct(m['ann_volatility'], signed=False)}")
    print(f"{'Sharpe':>16}: {_fmt_num(m['sharpe'])}")
    print(f"{'Sortino':>16}: {_fmt_num(m['sortino'])}")
    print(f"{'max drawdown':>16}: {_fmt_pct(m['max_drawdown'])}")
    print(f"{'Calmar':>16}: {_fmt_num(m['calmar'])}")
    print(f"{'exposure':>16}: {_fmt_pct(m['exposure'], signed=False)}")
    print(f"{'trades':>16}: {m['trades']}")
    print(f"{'win rate':>16}: {_fmt_pct(m['win_rate'], signed=False)}")
    print(f"{'profit factor':>16}: {_fmt_num(m['profit_factor'])}")
    print(f"{'avg trade':>16}: {_fmt_num(m['avg_trade_per_oz'])} $/oz")
    if m["bars"] < 500 or m["trades"] < 30:
        print("note: small sample - treat backtest stats as indicative only")
