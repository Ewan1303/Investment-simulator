"""
analytics.py - performance statistics computed from a daily portfolio-value series.

All functions take a pd.Series of portfolio values indexed by trading day.
Risk-free rate is assumed to be zero for Sharpe / Sortino.
"""
from __future__ import annotations

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
    m.update({
        "Number of trades": int(len(result.trades)),
        "Turnover (per year)": float(result.turnover_value / 2.0 / avg_value / yrs) if avg_value > 0 else float("nan"),
        "Transaction costs": float(result.costs_paid),
        "Average holding period": (f"{np.mean(result.holding_days) / 30.44:.1f} months"
                                   if result.holding_days else "n/a"),
    })
    return m


MONEY_KEYS = {"Starting capital", "Final value", "Transaction costs"}
PCT_KEYS = {"Total return", "CAGR", "Annual volatility", "Max drawdown", "Turnover (per year)"}
RATIO_KEYS = {"Sharpe ratio", "Sortino ratio"}


def fmt_money(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.0f}" if np.isfinite(x) else "n/a"


def fmt_pct(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def format_metric(key: str, value, symbol: str) -> str:
    if key in MONEY_KEYS:
        return fmt_money(value, symbol)
    if key in PCT_KEYS:
        return fmt_pct(value, signed=key in {"Total return", "CAGR", "Max drawdown"})
    if key in RATIO_KEYS:
        return f"{value:.2f}" if np.isfinite(value) else "n/a"
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
