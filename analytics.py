"""
analytics.py - performance statistics computed from a daily portfolio-value series.

All functions take a pd.Series of portfolio values indexed by trading day.
Risk-free rate is assumed to be zero for Sharpe / Sortino.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def returns(values: pd.Series) -> pd.Series:
    return values.pct_change().dropna()


def total_return(values: pd.Series) -> float:
    return float(values.iloc[-1] / values.iloc[0] - 1.0)


def years_spanned(values: pd.Series) -> float:
    return max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)


def cagr(values: pd.Series) -> float:
    return float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years_spanned(values)) - 1.0)


def annual_volatility(values: pd.Series) -> float:
    r = returns(values)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def sharpe(values: pd.Series) -> float:
    r = returns(values)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if len(r) > 1 and sd > 0 else float("nan")


def sortino(values: pd.Series) -> float:
    r = returns(values)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else float("nan")


def drawdown_series(values: pd.Series) -> pd.Series:
    return values / values.cummax() - 1.0


def max_drawdown(values: pd.Series) -> float:
    return float(drawdown_series(values).min())


def _period_returns(values: pd.Series, rule: str) -> pd.Series:
    """Returns per calendar period, the first period measured from the starting value."""
    ends = values.resample(rule).last().dropna()
    base = pd.concat([pd.Series([values.iloc[0]], index=[values.index[0] - pd.Timedelta(days=1)]), ends])
    return base.pct_change().dropna()


def annual_returns(values: pd.Series) -> pd.Series:
    r = _period_returns(values, "YE")
    r.index = r.index.year
    return r


def monthly_returns(values: pd.Series) -> pd.DataFrame:
    r = _period_returns(values, "ME")
    table = pd.DataFrame({"Year": r.index.year, "Month": r.index.month, "Return": r.values})
    pivot = table.pivot(index="Year", columns="Month", values="Return").reindex(columns=range(1, 13))
    pivot.columns = MONTHS
    return pivot


def annual_dividends(dividends: pd.Series) -> pd.Series:
    """Cash dividends received per calendar year."""
    s = dividends.resample("YE").sum()
    s.index = s.index.year
    return s


def annual_dividend_yield(dividends: pd.Series, values: pd.Series) -> pd.Series:
    """Dividends received in each year as a fraction of that year's average portfolio value."""
    avg = values.resample("YE").mean()
    avg.index = avg.index.year
    return (annual_dividends(dividends) / avg).replace([np.inf, -np.inf], np.nan).dropna()


def deflator(cpi: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Multiplier that turns nominal amounts into start-date money: CPI(start) / CPI(t)."""
    c = cpi.reindex(cpi.index.union(index)).ffill().reindex(index).bfill()
    return c.iloc[0] / c


def deflate_series(s: pd.Series, defl: pd.Series) -> pd.Series:
    return s * defl.reindex(s.index).ffill().bfill()


def deflate_result(result, defl: pd.Series):
    """Copy of a BacktestResult with every money amount expressed in start-date money."""
    d = defl.reindex(result.values.index).ffill().bfill()
    trades = result.trades.copy()
    if len(trades):
        f = d.reindex(pd.to_datetime(trades["Date"])).ffill().bfill().to_numpy()
        for col in ("Price", "Value", "Cost"):
            trades[col] = trades[col].to_numpy() * f
    calls = result.margin_calls.copy()
    if len(calls):
        f = d.reindex(pd.to_datetime(calls["Date"])).ffill().bfill().to_numpy()
        for col in ("Equity before", "Exposure before", "Sold", "Equity after"):
            calls[col] = calls[col].to_numpy() * f
    interest = result.interest * d
    return replace(result, values=result.values * d, dividends=result.dividends * d, interest=interest,
                   trades=trades, margin_calls=calls,
                   costs_paid=float(trades["Cost"].sum()) if len(trades) else 0.0,
                   interest_paid=float(interest.sum()),
                   turnover_value=float(trades["Value"].sum()) if len(trades) else 0.0)


def stocks_owned(result) -> pd.Series:
    """Daily value of the stocks held (gross exposure; above equity when leveraged)."""
    return (result.values * result.leverage_path.fillna(0.0)).fillna(0.0)


def annual_costs(result) -> pd.DataFrame:
    """Per calendar year: trading costs, loan interest, total, average value of stocks owned,
    total costs as a share of that (an expense-ratio-like figure) and number of trades."""
    t = result.trades
    if len(t):
        by_date = pd.Series(t["Cost"].to_numpy(), index=pd.to_datetime(t["Date"]))
        trading = by_date.resample("YE").sum()
        n_trades = pd.Series(1.0, index=pd.to_datetime(t["Date"])).resample("YE").sum()
    else:
        trading = pd.Series(dtype="float64")
        n_trades = pd.Series(dtype="float64")
    interest = result.interest.resample("YE").sum()
    owned = stocks_owned(result).resample("YE").mean()
    df = pd.DataFrame({"Trading costs": trading, "Interest": interest, "Stocks owned (avg)": owned,
                       "Trades": n_trades}).reindex(owned.index).fillna(0.0)
    df["Total costs"] = df["Trading costs"] + df["Interest"]
    df["Cost ratio"] = (df["Total costs"] / df["Stocks owned (avg)"].replace(0.0, np.nan))
    df.index = df.index.year
    return df


def _year_label(year: int, values: pd.Series) -> str:
    last = values.index[-1]
    partial = year == last.year and not (last.month == 12 and last.day >= 28)
    return f"{year} YTD" if partial else str(year)


def series_metrics(values: pd.Series) -> dict:
    ann = annual_returns(values)
    return {
        "Starting capital": float(values.iloc[0]),
        "Final value": float(values.iloc[-1]),
        "Total return": total_return(values),
        "CAGR": cagr(values),
        "Annual volatility": annual_volatility(values),
        "Max drawdown": max_drawdown(values),
        "Sharpe ratio": sharpe(values),
        "Sortino ratio": sortino(values),
        "Best year": f"{_year_label(int(ann.idxmax()), values)} ({ann.max():+.1%})" if len(ann) else "n/a",
        "Worst year": f"{_year_label(int(ann.idxmin()), values)} ({ann.min():+.1%})" if len(ann) else "n/a",
    }


def result_metrics(result) -> dict:
    """Full metric set for a BacktestResult (adds trading statistics)."""
    v = result.values
    m = series_metrics(v)
    yrs = years_spanned(v)
    avg_value = float(v.mean())
    yields = annual_dividend_yield(result.dividends, v)
    m.update({
        "Dividends received": float(result.dividends.sum()),
        "Average dividend yield": float(yields.mean()) if len(yields) else float("nan"),
        "Number of trades": int(len(result.trades)),
        "Turnover (per year)": float(result.turnover_value / 2.0 / avg_value / yrs) if avg_value > 0 else float("nan"),
        "Transaction costs": float(result.costs_paid),
        "Average holding period": (f"{np.mean(result.holding_days) / 30.44:.1f} months"
                                   if result.holding_days else "n/a"),
    })
    owned = stocks_owned(result).mean()
    m["Costs vs stocks owned (per year)"] = (float((result.costs_paid + result.interest_paid) / owned / yrs)
                                             if owned > 0 else float("nan"))
    if result.config.leveraged:
        m.update({
            "Average leverage": float(np.nanmean(result.leverage_path.to_numpy())),
            "Interest paid": float(result.interest_paid),
            "Margin calls": int(len(result.margin_calls)),
            "Wiped out": result.wiped_out.date().isoformat() if result.wiped_out is not None else "no",
        })
    return m


MONEY_KEYS = {"Starting capital", "Final value", "Transaction costs", "Dividends received", "Interest paid"}
PCT_KEYS = {"Total return", "CAGR", "Annual volatility", "Max drawdown", "Turnover (per year)", "Average dividend yield",
            "Costs vs stocks owned (per year)"}
RATIO_KEYS = {"Sharpe ratio", "Sortino ratio"}
MULTIPLE_KEYS = {"Average leverage"}


def fmt_money(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.0f}" if np.isfinite(x) else "n/a"


def fmt_pct(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def format_metric(key: str, value, symbol: str) -> str:
    if key in MONEY_KEYS:
        return fmt_money(value, symbol)
    if key.startswith("Costs vs"):
        return f"{value:.2%}" if np.isfinite(value) else "n/a"
    if key in PCT_KEYS:
        return fmt_pct(value, signed=key in {"Total return", "CAGR", "Max drawdown"})
    if key in RATIO_KEYS:
        return f"{value:.2f}" if np.isfinite(value) else "n/a"
    if key in MULTIPLE_KEYS:
        return f"{value:.2f}x" if np.isfinite(value) else "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def metrics_table(columns: dict[str, dict], symbol: str) -> pd.DataFrame:
    """columns: {column label -> metrics dict}. Missing keys show as '-'."""
    keys: list[str] = []
    for m in columns.values():
        for k in m:
            if k not in keys:
                keys.append(k)
    data = {label: [format_metric(k, m[k], symbol) if k in m else "-" for k in keys] for label, m in columns.items()}
    return pd.DataFrame(data, index=keys)
