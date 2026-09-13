"""
backtester.py - the simulation engine.

Timeline at every rebalance (no look-ahead, enforced by construction):

  decision day  d      the strategy sees prices up to and including the close of d,
                       the universe membership on d and market caps on d (built from
                       share counts published at least 60 days earlier).
  execution day d + 1  orders are filled at the close of the NEXT trading day. The chosen
                       weights are carried to that day's prices (i.e. the order is a number of
                       shares decided on day d), so a cap-weighted portfolio only trades when
                       membership, share counts or dividends change, not on every price move.

Positions are tracked as currency values that move with total-return prices
(dividends reinvested).  A stock that stops trading is frozen at its last price
and sold at the next rebalance.  Costs = (transaction cost + slippage) x traded value.

Leverage (optional).  Cash may go negative: that is the margin loan, and it accrues
interest daily.  Three methods:
  "Daily reset"          exposure is reset to L x equity every day, like a leveraged ETF.
                         Volatility drag and the daily re-hedging costs are modelled; no
                         margin calls are possible, but a one-day fall of more than 1/L wipes
                         the account out.
  "Reset at rebalance"   exposure is set to L x equity at each rebalance and drifts in between.
  "Initial loan only"    borrow (L-1) x capital once; the loan is never topped up, so leverage
                         falls as the portfolio grows and rises as it falls.
For the last two, equity / exposure is checked every day against the maintenance margin.
Below it, positions are sold pro rata until leverage is back to the target (or to 10 points
above maintenance if the target is more aggressive), and the sale is logged as a margin
call.  If equity reaches zero the account is wiped out and stays at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from analytics import TRADING_DAYS, cagr, max_drawdown
from data import MarketData
from strategies import MarketCapWeighted, RandomSelection, Strategy

REBALANCE_OPTIONS = ["Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Never"]
LEVERAGE_METHODS = ["Reset at rebalance", "Daily reset", "Initial loan only"]
HISTORY_ROWS = 300          # trading days of history handed to strategies (>= 12 months + margin)
ALL = 10 ** 6               # "every company in the universe"
ONE_DAY = np.timedelta64(1, "D")
TRADE_COLUMNS = ["Date", "Ticker", "Action", "Price", "Quantity", "Value", "Cost", "Reason"]
CALL_COLUMNS = ["Date", "Equity before", "Exposure before", "Leverage before", "Sold", "Equity after",
                "Leverage after", "Type"]


@dataclass
class BacktestConfig:
    top_n: int                  # last market-cap rank in the pool (ALL = whole universe)
    n_holdings: int
    rebalance: str
    start: pd.Timestamp
    end: pd.Timestamp
    capital: float = 100_000.0
    tx_cost: float = 0.001
    slippage: float = 0.0005
    seed: int = 0
    rank_from: int = 1          # first market-cap rank in the pool (1 = largest company)
    leverage: float = 1.0       # gross exposure / equity (1 = no leverage)
    leverage_method: str = "Reset at rebalance"
    borrow_rate: float = 0.04   # annual interest on the margin loan
    maintenance_margin: float = 0.25   # margin call when equity / exposure falls below this

    @property
    def cost_rate(self) -> float:
        return self.tx_cost + self.slippage

    @property
    def leveraged(self) -> bool:
        return self.leverage > 1.0 + 1e-9


def pool_label(rank_from: int, top_n: int, whole: bool = False) -> str:
    """'All', 'Top 10' or 'Ranks 25-50'."""
    if whole or (rank_from == 1 and top_n >= ALL):
        return "All"
    last = "end" if top_n >= ALL else str(top_n)
    return f"Top {top_n}" if rank_from == 1 else f"Ranks {rank_from}-{last}"


@dataclass
class BacktestResult:
    label: str
    strategy_name: str
    config: BacktestConfig
    currency: str
    values: pd.Series               # daily equity (portfolio value net of any loan)
    trades: pd.DataFrame            # Date, Ticker, Action, Price, Quantity, Value, Cost, Reason
    weights: pd.DataFrame           # rows = execution dates, columns = tickers, values = weight of equity
    dividends: pd.Series            # cash dividends received each day (reinvested the same day)
    pool: dict                      # decision date -> the pool that day
    costs_paid: float
    turnover_value: float           # total currency value traded (buys + sells)
    holding_days: list
    leverage_path: pd.Series        # gross exposure / equity each day
    margin_calls: pd.DataFrame      # one row per forced sale
    interest_paid: float
    interest: pd.Series             # loan interest charged each day
    wiped_out: pd.Timestamp | None  # date equity hit zero, if it did

    @property
    def final_value(self) -> float:
        return float(self.values.iloc[-1])

    @property
    def n_trades(self) -> int:
        return int(len(self.trades))


class Backtester:
    def __init__(self, md: MarketData):
        self.md = md
        self._pool_cache: dict[tuple[int, int, int], tuple] = {}

    def _pool(self, di: int, top_n: int, rank_from: int = 1) -> tuple[np.ndarray, list[str], pd.Index, pd.Series]:
        """The pool on decision day `di`: members ranked rank_from..top_n by market cap.
        Returns (column indices, tickers, Index, market caps). Cached per day."""
        key = (di, rank_from, top_n)
        hit = self._pool_cache.get(key)
        if hit is None:
            m = self.md.mcap[di]
            cand = np.flatnonzero(self.md.member[di] & np.isfinite(m) & (m > 0))
            order = cand[np.argsort(-m[cand], kind="stable")][max(rank_from, 1) - 1:top_n]
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

        # leverage settings
        L = max(float(cfg.leverage), 1.0)
        method = cfg.leverage_method if cfg.leveraged else None
        r_d = cfg.borrow_rate / TRADING_DAYS if cfg.leveraged else 0.0   # no loan without leverage
        m_call = float(cfg.maintenance_margin)
        restore = max(1.0 / L, m_call + 0.10)   # equity / exposure after a margin call

        pos = np.zeros(T)                     # position values (gross exposure) in currency
        cash = float(cfg.capital)             # negative = margin loan
        n_days = i1 - i0 + 1
        values = np.empty(n_days)
        values[0] = cash
        dividends = np.zeros(n_days)
        lev_path = np.ones(n_days)
        interest_d = np.zeros(n_days)
        trades: list[tuple] = []
        margin_calls: list[dict] = []
        weights_log: list[tuple] = []       # (execution date, held column indices, weights of equity)
        pool_log: dict = {}
        open_since: dict[int, int] = {}
        holding_days: list[int] = []
        state = {"costs": 0.0, "turnover": 0.0, "interest": 0.0, "prev": i0, "cash": cash, "wiped": None}

        def liquidate(pos: np.ndarray, frac: float, day: int, reason: str) -> float:
            """Sell `frac` of every position at day `day`; returns the cash raised after costs."""
            sold_total = 0.0
            when = dates_np[day]
            for j in np.flatnonzero(pos):
                amount = pos[j] * frac
                if amount <= 0:
                    continue
                price = md.close[day, j]
                trades.append((when, tickers[j], "SELL", price, amount / price, amount, amount * rate, reason))
                sold_total += amount
                if frac >= 1.0 - 1e-12:
                    holding_days.append(int((when - dates_np[open_since.pop(j, day)]) / ONE_DAY))
            pos *= (1.0 - frac)
            if frac >= 1.0 - 1e-12:
                pos[:] = 0.0
            cost = sold_total * rate
            state["costs"] += cost
            state["turnover"] += sold_total
            return sold_total - cost

        def carry_forward(pos: np.ndarray, to_idx: int) -> np.ndarray:
            """Move positions from state['prev'] to the close of `to_idx`, filling in daily equity,
            dividends, interest and (for margin methods) any forced sales along the way."""
            while True:
                prev = state["prev"]
                cash = state["cash"]
                n = to_idx - prev
                if n <= 0:
                    return pos
                held = np.flatnonzero(pos)
                if held.size:
                    seg = md.adj[prev:to_idx + 1][:, held]
                    vals = np.nan_to_num(seg / seg[0], nan=1.0) * pos[held]      # rows prev..to_idx
                    exposure = vals[1:].sum(axis=1)
                    divs = (vals[:-1] * md.div_yield[prev + 1:to_idx + 1][:, held]).sum(axis=1)
                else:
                    vals = None
                    exposure = np.zeros(n)
                    divs = np.zeros(n)
                sl = slice(prev + 1 - i0, to_idx + 1 - i0)

                if method == "Daily reset" and held.size and cash < 0:
                    # equity compounds at L x the mix return minus interest and daily re-hedging costs
                    V = vals.sum(axis=1)
                    R = V[1:] / V[:-1] - 1.0
                    E0 = cash + V[0]
                    churn = L * (L - 1.0) * np.abs(R)
                    growth = np.maximum(1.0 + L * R - (L - 1.0) * r_d - churn * rate, 0.0)
                    E = E0 * np.cumprod(growth)
                    E_prev = np.r_[E0, E[:-1]]
                    values[sl] = E
                    dividends[sl] += divs * (L * E_prev / V[:-1])
                    lev_path[sl] = L
                    interest_d[sl] += (L - 1.0) * r_d * E_prev
                    state["interest"] += float(((L - 1.0) * r_d * E_prev).sum())
                    state["costs"] += float((churn * rate * E_prev).sum())
                    state["turnover"] += float((churn * E_prev).sum())
                    E_end = float(E[-1])
                    if E_end <= 0.0:
                        margin_calls.append({"Date": dates_np[to_idx], "Equity before": 0.0,
                                             "Exposure before": float(L * E_prev[-1]), "Leverage before": L,
                                             "Sold": float(L * E_prev[-1]), "Equity after": 0.0,
                                             "Leverage after": np.nan, "Type": "wiped out"})
                        pos[:] = 0.0
                        state["cash"] = 0.0
                        state["wiped"] = dates_np[to_idx]
                        values[to_idx - i0] = 0.0
                    else:
                        pos[held] = vals[-1] * (L * E_end / V[-1])
                        state["cash"] = E_end - L * E_end
                    state["prev"] = to_idx
                    return pos

                # loan interest accrues daily on negative cash
                if cash < 0:
                    cash_path = cash * (1.0 + r_d) ** np.arange(1, n + 1)
                else:
                    cash_path = np.full(n, cash)
                equity = cash_path + exposure
                call_at = None
                if method in ("Reset at rebalance", "Initial loan only") and cash < 0:
                    ratio = np.where(exposure > 0, equity / np.maximum(exposure, 1e-12), 1.0)
                    bad = np.flatnonzero(ratio < m_call)
                    if bad.size:
                        call_at = int(bad[0])
                stop = n if call_at is None else call_at + 1
                sl = slice(prev + 1 - i0, prev + 1 - i0 + stop)
                values[sl] = equity[:stop]
                dividends[sl] += divs[:stop]
                lev_path[sl] = np.where(equity[:stop] > 0, exposure[:stop] / np.maximum(equity[:stop], 1e-12), np.nan)
                if cash < 0:
                    interest_d[sl] += (np.r_[cash, cash_path[:-1]] - cash_path)[:stop]
                    state["interest"] += float(cash - cash_path[stop - 1])
                state["cash"] = float(cash_path[stop - 1])
                if held.size:
                    pos[held] = vals[stop]
                state["prev"] = prev + stop
                if call_at is None:
                    return pos

                # ---- margin call at the close of day state['prev'] ----
                day = state["prev"]
                X = float(pos.sum())
                E = state["cash"] + X
                lev_before = X / E if E > 0 else np.inf
                if E <= 0.0:
                    raised = liquidate(pos, 1.0, day, "wiped out")
                    state["cash"] = min(state["cash"] + raised, 0.0)   # shortfall is the broker's loss
                    margin_calls.append({"Date": dates_np[day], "Equity before": E, "Exposure before": X,
                                         "Leverage before": lev_before, "Sold": X, "Equity after": 0.0,
                                         "Leverage after": np.nan, "Type": "wiped out"})
                    state["cash"] = 0.0
                    state["wiped"] = dates_np[day]
                    values[day - i0] = 0.0
                    lev_path[day - i0] = np.nan
                    return pos
                sell = min(max((X * restore - E) / (restore - rate), 0.0), X)
                raised = liquidate(pos, sell / X, day, "margin call")
                state["cash"] += raised
                E_after = state["cash"] + float(pos.sum())
                values[day - i0] = E_after
                lev_path[day - i0] = float(pos.sum()) / E_after if E_after > 0 else np.nan
                margin_calls.append({"Date": dates_np[day], "Equity before": E, "Exposure before": X,
                                     "Leverage before": lev_before, "Sold": sell, "Equity after": E_after,
                                     "Leverage after": lev_path[day - i0], "Type": "margin call"})

        first_execution = True
        for di in decisions:
            ei = di + 1
            pos = carry_forward(pos, ei)
            cash = state["cash"]
            if state["wiped"] is not None:
                continue

            # ---- decide with information available at the close of day `di` only ----
            order, pool, pool_index, pool_mcap = self._pool(di, cfg.top_n, cfg.rank_from)
            target = np.zeros(T)
            total = cash + pos.sum()                    # equity
            if order.size and total > 0:
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
                    # The weights were chosen with decision-day prices. Carry them to the execution
                    # close so the order means "buy this many shares", not "chase yesterday's weights"
                    # (otherwise every daily mover is trimmed and bought back the next day).
                    drift = np.nan_to_num(md.adj[ei, idx] / md.adj[di, idx], nan=1.0)
                    wv = wv * drift * (wv.sum() / max(float((wv * drift).sum()), 1e-12))
                    # gross exposure to aim for
                    if method == "Initial loan only":
                        loan = (L - 1.0) * total if first_execution else max(-cash, 0.0)
                        gross_mult = (total + loan) / total
                    else:
                        gross_mult = L
                    est_traded = np.abs(wv * gross_mult * total - pos[idx]).sum() + (pos.sum() - pos[idx].sum())
                    equity_after = max(total - est_traded * rate, 0.0)
                    target[idx] = wv * gross_mult * equity_after
            else:
                total = max(total, 0.0)

            # ---- execute at the close of day `ei` ----
            diff = target - pos
            moved = np.abs(diff)
            dust = max(1.0, 1e-4 * max(total, 1.0))
            small = moved < dust                                   # ignore dust-sized rebalancing trades
            planned = target.sum()
            target[small] = pos[small]
            diff[small] = 0.0
            moved[small] = 0.0
            active = np.flatnonzero(moved)
            cost = float(moved[active].sum() * rate)
            cash = cash - float(diff[active].sum()) - cost
            state["costs"] += cost
            state["turnover"] += float(moved[active].sum())
            when = dates_np[ei]
            for j in active:
                price = md.close[ei, j]
                action = "BUY" if diff[j] > 0 else "SELL"
                trades.append((when, tickers[j], action, price, moved[j] / price, moved[j], moved[j] * rate, "rebalance"))
                if action == "BUY" and pos[j] == 0:
                    open_since[j] = ei
                if target[j] == 0 and pos[j] > 0:
                    holding_days.append(int((when - dates_np[open_since.pop(j, ei)]) / ONE_DAY))
            pos = target
            # Cash sweep: the skipped dust trades leave cash away from the plan. Once that gap exceeds
            # dust, buy or sell a uniform slice of the whole portfolio so exposure stays as intended
            # (otherwise cash silently piles up across hundreds of names, or a small free loan appears).
            residual = planned - pos.sum()
            if abs(residual) >= dust and pos.sum() > 0:
                sweep_cost = abs(residual) * rate
                pos *= 1.0 + residual / pos.sum()
                cash -= residual + sweep_cost
                state["costs"] += sweep_cost
                state["turnover"] += abs(residual)
                trades.append((when, "PORTFOLIO", "BUY" if residual > 0 else "SELL", np.nan, np.nan,
                               abs(residual), sweep_cost, "cash sweep"))
            state["cash"] = cash
            equity = cash + pos.sum()
            values[ei - i0] = equity
            lev_path[ei - i0] = pos.sum() / equity if equity > 0 else np.nan
            held = np.flatnonzero(pos)
            weights_log.append((when, held, pos[held] / max(equity, 1e-12)))
            if active.size:
                first_execution = False

        pos = carry_forward(pos, i1)
        holding_days += [int((dates_np[i1] - dates_np[o]) / ONE_DAY) for o in open_since.values()]

        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        used = sorted({int(j) for _, held, _ in weights_log for j in held})
        col_of = {j: k for k, j in enumerate(used)}
        w_mat = np.zeros((len(weights_log), len(used)))
        for r, (_, held, w) in enumerate(weights_log):
            for j, wj in zip(held, w):
                w_mat[r, col_of[int(j)]] = wj
        weights_df = pd.DataFrame(w_mat, index=pd.DatetimeIndex([d for d, _, _ in weights_log]),
                                  columns=[md.tickers[j] for j in used])
        index = dates[i0:i1 + 1]
        calls_df = pd.DataFrame(margin_calls, columns=CALL_COLUMNS)
        return BacktestResult(label=label or strategy.label(), strategy_name=strategy.label(), config=cfg,
                              currency=md.currency, values=pd.Series(values, index=index, name=label),
                              trades=trades_df, weights=weights_df,
                              dividends=pd.Series(dividends, index=index),
                              pool=pool_log, costs_paid=state["costs"],
                              turnover_value=state["turnover"], holding_days=holding_days,
                              leverage_path=pd.Series(lev_path, index=index), margin_calls=calls_df,
                              interest_paid=state["interest"], interest=pd.Series(interest_d, index=index),
                              wiped_out=pd.Timestamp(state["wiped"]) if state["wiped"] is not None else None)

    # ---- active strategy + passive benchmarks --------------------------------------------
    def universe_max_size(self) -> int:
        return int(self.md.member.sum(axis=1).max())

    def run_passive(self, cfg: BacktestConfig, top_n: int, label: str | None = None,
                    rank_from: int = 1) -> BacktestResult:
        """Passive benchmark: hold every company ranked rank_from..top_n by market cap, cap-weighted,
        refreshed quarterly, same costs, no leverage. rank_from == 1 and top_n >= size = whole universe."""
        whole = rank_from == 1 and top_n >= self.universe_max_size()
        n = ALL if whole else top_n
        passive_cfg = replace(cfg, rebalance="Quarterly", top_n=n, n_holdings=ALL, rank_from=rank_from, leverage=1.0)
        if label is None:
            label = (f"{self.md.universe} (cap-weighted, all)" if whole
                     else f"Passive {pool_label(rank_from, top_n)} (cap-weighted) | {self.md.universe}")
        return self.run(passive_cfg, MarketCapWeighted(), label)

    def run_with_benchmarks(self, cfg: BacktestConfig, strategy: Strategy, label: str) -> dict:
        md = self.md
        active = self.run(cfg, strategy, label, record_pool=True)
        out = {"active": active}
        if not (cfg.rank_from == 1 and cfg.top_n >= self.universe_max_size()):
            out["passive"] = self.run_passive(cfg, cfg.top_n, f"Passive {pool_label(cfg.rank_from, cfg.top_n)} "
                                                              f"(cap-weighted)", rank_from=cfg.rank_from)
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
