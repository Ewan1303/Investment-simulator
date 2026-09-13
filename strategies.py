"""
strategies.py - how stocks are picked and weighted at each rebalance.

The engine hands every strategy the same three things and nothing else:
  pool   the N largest companies in the universe on the decision date
  hist   total-return prices for the pool up to AND INCLUDING the decision date
  mcap   market caps of the pool on the decision date
Because `hist` is cut off at the decision date, a strategy cannot see the future.

select() returns a Series of portfolio weights (ticker -> weight, summing to <= 1;
anything left over stays in cash).  Adding a strategy = one small class + one entry
in STRATEGIES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21


Weights = dict[str, float]


def _equal(tickers) -> Weights:
    tickers = [str(t) for t in tickers]
    return {t: 1.0 / len(tickers) for t in tickers} if tickers else {}


def _cap_weighted(mcap: pd.Series) -> Weights:
    mcap = mcap.dropna()
    return (mcap / mcap.sum()).to_dict() if len(mcap) else {}


class Strategy:
    name = "Base"
    description = ""
    needs_history = True          # False lets the engine skip building `hist` (faster)
    rebalance_override = None     # e.g. "Never" for Buy & Hold

    def select(self, pool: list[str], hist: pd.DataFrame | None, mcap: pd.Series,
               n: int, rng: np.random.Generator) -> Weights:
        """Return {ticker: weight}. A pd.Series is accepted too. Weights sum to <= 1."""
        raise NotImplementedError

    def label(self) -> str:
        return self.name


class BuyAndHold(Strategy):
    name = "Buy & Hold"
    description = "Buy the largest companies on day one (cap-weighted) and never trade again."
    needs_history = False
    rebalance_override = "Never"

    def select(self, pool, hist, mcap, n, rng):
        return _cap_weighted(mcap.nlargest(n))


class MarketCapWeighted(Strategy):
    name = "Market Cap Weighted"
    description = "Hold the largest companies, weighted by market cap, refreshed at each rebalance."
    needs_history = False

    def select(self, pool, hist, mcap, n, rng):
        return _cap_weighted(mcap.nlargest(n))


class EqualWeight(Strategy):
    name = "Equal Weight"
    description = "Hold the largest companies with equal weights, rebalanced back to equal each period."
    needs_history = False

    def select(self, pool, hist, mcap, n, rng):
        return _equal(mcap.nlargest(n).index)


class Momentum(Strategy):
    name = "Momentum"
    description = "Rank the pool by trailing total return over the lookback and hold the strongest, equal-weighted."

    def __init__(self, lookback_months: int = 6):
        self.lookback_months = int(lookback_months)

    def label(self) -> str:
        return f"Momentum ({self.lookback_months}m)"

    def select(self, pool, hist, mcap, n, rng):
        lag = self.lookback_months * TRADING_DAYS_PER_MONTH
        if hist is None or len(hist) <= lag:
            return {}
        score = (hist.iloc[-1] / hist.iloc[-1 - lag] - 1.0).dropna()
        return _equal(score.nlargest(n).index)


class TrendFollowing(Strategy):
    name = "Trend Following"
    description = ("Only hold stocks trading above their moving average, ranked by how far above it they are, "
                   "equal-weighted. Anything left over stays in cash.")

    def __init__(self, ma_days: int = 200):
        self.ma_days = int(ma_days)

    def label(self) -> str:
        return f"Trend ({self.ma_days}d MA)"

    def select(self, pool, hist, mcap, n, rng):
        if hist is None or len(hist) < self.ma_days:
            return {}
        window = hist.iloc[-self.ma_days:]
        sma = window.mean()
        score = (window.iloc[-1] / sma - 1.0)
        score = score[(window.notna().sum() >= self.ma_days * 0.9) & (score > 0)].dropna()
        return _equal(score.nlargest(n).index)


class RandomSelection(Strategy):
    name = "Random Selection"
    description = "Pick holdings from the pool at random (equal-weighted). Use Monte Carlo to see the spread."
    needs_history = False

    def select(self, pool, hist, mcap, n, rng):
        k = min(n, len(pool))
        if k == 0:
            return {}
        picks = rng.choice(len(pool), size=k, replace=False)
        return _equal(pool[i] for i in picks)


STRATEGIES: dict[str, type[Strategy]] = {
    BuyAndHold.name: BuyAndHold,
    MarketCapWeighted.name: MarketCapWeighted,
    EqualWeight.name: EqualWeight,
    Momentum.name: Momentum,
    TrendFollowing.name: TrendFollowing,
    RandomSelection.name: RandomSelection,
}

MOMENTUM_LOOKBACKS = [1, 3, 6, 12]       # months
TREND_WINDOWS = [50, 100, 200]           # days


def make_strategy(name: str, **params) -> Strategy:
    cls = STRATEGIES[name]
    if cls is Momentum:
        return Momentum(params.get("lookback_months", 6))
    if cls is TrendFollowing:
        return TrendFollowing(params.get("ma_days", 200))
    return cls()
