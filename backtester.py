"""
backtester.py - the simulation engine.

Timeline at every rebalance (no look-ahead, enforced by construction):

  decision day  d      the strategy sees prices up to and including the close of d,
                       the universe membership on d and market caps on d (built from
                       share counts published at least 60 days earlier).
  execution day d + 1  orders are filled at the close of the NEXT trading day.

Positions are tracked as currency values that move with total-return prices
(dividends reinvested).  A stock that stops trading is frozen at its last price
and sold at the next rebalance.  Costs = (transaction cost + slippage) x traded value.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from analytics import cagr, max_drawdown
from data import MarketData
from strategies import MarketCapWeighted, RandomSelection, Strategy

REBALANCE_OPTIONS = ["Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Never"]
HISTORY_ROWS = 300          # trading days of history handed to strategies (>= 12 months + margin)
ALL = 10 ** 6               # "every company in the universe"
ONE_DAY = np.timedelta64(1, "D")


@dataclass
class BacktestConfig:
    top_n: int
    n_holdings: int
    rebalance: str
    start: pd.Timestamp
    end: pd.Timestamp
    capital: float = 100_000.0
    tx_cost: float = 0.001
    slippage: float = 0.0005
    seed: int = 0

    @property
    def cost_rate(self) -> float:
        return self.tx_cost + self.slippage


@dataclass
class BacktestResult:
    label: str
    strategy_name: str
    config: BacktestConfig
    currency: str
    values: pd.Series               # daily portfolio value
    trades: pd.DataFrame            # Date, Ticker, Action, Price, Quantity, Value, Cost
    weights: pd.DataFrame           # rows = execution dates, columns = tickers, values = weight of portfolio
    pool: dict                      # decision date -> the Top-N pool that day
    costs_paid: float
    turnover_value: float           # total currency value traded (buys + sells)
    holding_days: list

    @property
    def final_value(self) -> float:
        return float(self.values.iloc[-1])

    @property
    def n_trades(self) -> int:
        return int(len(self.trades))


class Backtester:
    def __init__(self, md: MarketData):
        self.md = md
        self._pool_cache: dict[tuple[int, int], tuple] = {}

    def _pool(self, di: int, top_n: int) -> tuple[np.ndarray, list[str], pd.Index, pd.Series]:
        """The Top-N pool on decision day `di`: (column indices, tickers, Index, market caps). Cached."""
        key = (di, top_n)
        hit = self._pool_cache.get(key)
        if hit is None:
            m = self.md.mcap[di]
            cand = np.flatnonzero(self.md.member[di] & np.isfinite(m) & (m > 0))
            order = cand[np.argsort(-m[cand], kind="stable")][:top_n]
            pool = [self.md.tickers[j] for j in order]
            index = pd.Index(pool, dtype=object)
            hit = self._pool_cache[key] = (order, pool, index, pd.Series(m[order], index=index))
        return hit

    # ---- rebalance calendar ------------------------------------------------------------
    def _decision_indices(self, i0: int, i1: int, freq: str) -> np.ndarray:
        d = self.md.dates[i0:i1 + 1]
        if freq == "Never":
            key = np.zeros(len(d), dtype=int)
        elif freq == "Daily":
            key = np.arange(len(d))
        elif freq == "Weekly":
            iso = d.isocalendar()
            key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy())
        elif freq == "Monthly":
            key = d.year.to_numpy() * 12 + d.month.to_numpy()
        elif freq == "Quarterly":
            key = d.year.to_numpy() * 4 + (d.month.to_numpy() - 1) // 3
        elif freq == "Annually":
            key = d.year.to_numpy()
        else:
            raise ValueError(f"Unknown rebalance frequency {freq!r}")
        first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) + i0
        return first[first + 1 <= i1]          # the execution day (d + 1) must exist

    # ---- one backtest -------------------------------------------------------------------
    def run(self, cfg: BacktestConfig, strategy: Strategy, label: str = "", record_pool: bool = False) -> BacktestResult:
        md = self.md
        dates = md.dates
        i0 = int(dates.searchsorted(pd.Timestamp(cfg.start)))
        i1 = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) - 1
        if i1 - i0 < 5:
            raise ValueError("Backtest window is too short.")
        freq = strategy.rebalance_override or cfg.rebalance
        decisions = self._decision_indices(i0, i1, freq)

        tickers = np.array(md.tickers)
        dates_np = dates.to_numpy()
        T = len(tickers)
        rate = cfg.cost_rate
        rng = np.random.default_rng(cfg.seed)

        pos = np.zeros(T)                     # position values in currency
        cash = float(cfg.capital)
        values = np.empty(i1 - i0 + 1)
        values[0] = cash
        trades: list[tuple] = []
        weights_log: list[tuple] = []       # (execution date, held column indices, weights)
        pool_log: dict = {}
        open_since: dict[int, int] = {}
        holding_days: list[int] = []
        costs_paid = 0.0
        turnover_value = 0.0
        prev = i0

        def carry_forward(pos, to_idx):
            """Move positions from `prev` to the close of `to_idx`, filling in daily values."""
            held = np.flatnonzero(pos)
            if held.size:
                seg = md.adj[prev:to_idx + 1][:, held]
                rel = np.nan_to_num(seg / seg[0], nan=1.0)
                values[prev + 1 - i0:to_idx + 1 - i0] = cash + rel[1:] @ pos[held]
                pos[held] = pos[held] * rel[-1]
            else:
                values[prev + 1 - i0:to_idx + 1 - i0] = cash
            return pos

        for di in decisions:
            ei = di + 1
            pos = carry_forward(pos, ei)

            # ---- decide with information available at the close of day `di` only ----
            order, pool, pool_index, pool_mcap = self._pool(di, cfg.top_n)
            target = np.zeros(T)
            total = cash + pos.sum()
            if order.size:
                hist = None
                if strategy.needs_history:
                    lo = max(0, di - HISTORY_ROWS + 1)
                    hist = pd.DataFrame(md.adj[lo:di + 1][:, order], index=dates[lo:di + 1], columns=pool_index)
                    assert hist.index[-1] <= dates[di]     # explicit no-look-ahead guard
                w = strategy.select(pool, hist, pool_mcap, cfg.n_holdings, rng)
                if isinstance(w, pd.Series):
                    w = w.to_dict()
                if record_pool:
                    pool_log[dates[di]] = pool
                items = [(t, float(v)) for t, v in w.items() if v > 0]
                if items:
                    col = {t: j for t, j in zip(pool, order)}
                    idx = np.array([col[t] for t, _ in items], dtype=int)
                    wv = np.array([v for _, v in items])
                    est_traded = np.abs(wv * total - pos[idx]).sum() + (pos.sum() - pos[idx].sum())
                    investable = max(total - est_traded * rate, 0.0)
                    target[idx] = wv * investable

            # ---- execute at the close of day `ei` ----
            diff = target - pos
            moved = np.abs(diff)
            small = moved < max(1.0, 1e-4 * total)     # ignore dust-sized rebalancing trades
            target[small] = pos[small]
            diff[small] = 0.0
            moved[small] = 0.0
            active = np.flatnonzero(moved)
            cost = float(moved[active].sum() * rate)
            cash = cash - float(diff[active].sum()) - cost
            costs_paid += cost
            turnover_value += float(moved[active].sum())
            when = dates_np[ei]
            for j in active:
                price = md.close[ei, j]
                action = "BUY" if diff[j] > 0 else "SELL"
                trades.append((when, tickers[j], action, price, moved[j] / price, moved[j], moved[j] * rate))
                if action == "BUY" and pos[j] == 0:
                    open_since[j] = ei
                if target[j] == 0 and pos[j] > 0:
                    holding_days.append(int((when - dates_np[open_since.pop(j, ei)]) / ONE_DAY))
            pos = target
            values[ei - i0] = cash + pos.sum()
            held = np.flatnonzero(pos)
            weights_log.append((when, held, pos[held] / values[ei - i0]))
            prev = ei

        pos = carry_forward(pos, i1)
        holding_days += [int((dates_np[i1] - dates_np[o]) / ONE_DAY) for o in open_since.values()]

        trades_df = pd.DataFrame(trades, columns=["Date", "Ticker", "Action", "Price", "Quantity", "Value", "Cost"])
        used = sorted({int(j) for _, held, _ in weights_log for j in held})
        col_of = {j: k for k, j in enumerate(used)}
        w_mat = np.zeros((len(weights_log), len(used)))
        for r, (_, held, w) in enumerate(weights_log):
            for j, wj in zip(held, w):
                w_mat[r, col_of[int(j)]] = wj
        weights_df = pd.DataFrame(w_mat, index=pd.DatetimeIndex([d for d, _, _ in weights_log]),
                                  columns=[md.tickers[j] for j in used])
        return BacktestResult(label=label or strategy.label(), strategy_name=strategy.label(), config=cfg,
                              currency=md.currency, values=pd.Series(values, index=dates[i0:i1 + 1], name=label),
                              trades=trades_df, weights=weights_df, pool=pool_log, costs_paid=costs_paid,
                              turnover_value=turnover_value, holding_days=holding_days)

    # ---- active strategy + passive benchmarks --------------------------------------------
    def universe_max_size(self) -> int:
        return int(self.md.member.sum(axis=1).max())

    def run_passive(self, cfg: BacktestConfig, top_n: int, label: str | None = None) -> BacktestResult:
        """Passive benchmark: hold the N largest companies cap-weighted, refreshed quarterly, same costs.
        top_n >= the universe size means the whole universe."""
        whole = top_n >= self.universe_max_size()
        n = ALL if whole else top_n
        passive_cfg = replace(cfg, rebalance="Quarterly", top_n=n, n_holdings=n)
        if label is None:
            label = (f"{self.md.universe} (cap-weighted, all)" if whole
                     else f"Passive Top {top_n} (cap-weighted) | {self.md.universe}")
        return self.run(passive_cfg, MarketCapWeighted(), label)

    def run_with_benchmarks(self, cfg: BacktestConfig, strategy: Strategy, label: str) -> dict:
        md = self.md
        active = self.run(cfg, strategy, label, record_pool=True)
        out = {"active": active}
        if cfg.top_n < self.universe_max_size():
            out["passive"] = self.run_passive(cfg, cfg.top_n, f"Passive Top {cfg.top_n} (cap-weighted)")
        out["universe"] = self.run_passive(cfg, ALL)
        idx = active.values.index
        bench = md.benchmark.reindex(idx).ffill()
        out["index"] = (bench / bench.iloc[0] * cfg.capital).rename(md.benchmark_label)
        return out

    # ---- Monte Carlo over random selections -------------------------------------------------
    def monte_carlo(self, cfg: BacktestConfig, n_sims: int, progress=None) -> pd.DataFrame:
        rows = []
        for s in range(n_sims):
            r = self.run(replace(cfg, seed=s), RandomSelection(), f"Random #{s}")
            rows.append({"seed": s, "final_value": r.final_value, "cagr": cagr(r.values),
                         "max_drawdown": max_drawdown(r.values)})
            if progress and (s % 10 == 0 or s == n_sims - 1):
                progress((s + 1) / n_sims, f"Random portfolio {s + 1} of {n_sims}")
        return pd.DataFrame(rows)
