"""
charts.py - Plotly figures for the dashboard.

Colours are assigned by role and never re-cycled: the active strategy is always blue,
the passive Top-N benchmark orange, the whole universe aqua and the index ETF grey.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ROLE = {"active": "#2a78d6", "passive": "#eb6834", "universe": "#1baf7a", "index": "#898781"}
GOOD, BAD = "#0ca30c", "#d03b3b"
GRID = "rgba(137,135,129,0.22)"
MUTED = "#898781"
OTHER = "#b5b3ad"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, height: int = 430, hovermode: str = "x unified", **kwargs) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
        hovermode=hovermode, hoverlabel=dict(font=dict(family=FONT, size=12)),
        **kwargs,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED))
    return fig


def _money_fmt(symbol: str) -> str:
    return symbol + "%{y:,.0f}"


def _short_money(x: float, symbol: str) -> str:
    if abs(x) >= 1e6:
        return f"{symbol}{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"{symbol}{x / 1e3:,.0f}k"
    return f"{symbol}{x:,.0f}"


def growth_chart(series: dict[str, tuple[pd.Series, str]], symbol: str, log: bool = False,
                 height: int = 470, dashes: dict[str, str] | None = None) -> go.Figure:
    """series: {label: (daily values, colour)}. Draws every series from the same starting capital.
    dashes: optional {label: 'dash' | 'dot'} secondary encoding when colours have to repeat."""
    fig = go.Figure()
    for k, (label, (s, color)) in enumerate(series.items()):
        width = 2.4 if k == 0 else 1.8
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label, mode="lines",
                                 line=dict(color=color, width=width, dash=(dashes or {}).get(label, "solid")),
                                 hovertemplate=_money_fmt(symbol) + "<extra>%{fullData.name}</extra>"))
        if len(series) <= 4:
            fig.add_annotation(x=s.index[-1], y=s.values[-1], text=f" {_short_money(s.values[-1], symbol)}",
                               showarrow=False, xanchor="left", font=dict(size=11, color=color))
    _layout(fig, height=height)
    fig.update_yaxes(tickprefix=symbol, tickformat=",.0f", type="log" if log else "linear",
                     title=f"Portfolio value ({symbol})")
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig


def cumulative_return_chart(series: dict[str, tuple[pd.Series, str]], height: int = 400) -> go.Figure:
    fig = go.Figure()
    for label, (s, color) in series.items():
        r = s / s.iloc[0] - 1.0
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name=label, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate="%{y:+.1%}<extra>%{fullData.name}</extra>"))
    _layout(fig, height=height)
    fig.update_yaxes(tickformat="+.0%", title="Cumulative return")
    return fig


def drawdown_chart(series: dict[str, tuple[pd.Series, str]], height: int = 400) -> go.Figure:
    fig = go.Figure()
    for k, (label, (dd, color)) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=label, mode="lines",
                                 line=dict(color=color, width=2 if k == 0 else 1.5),
                                 fill="tozeroy" if k == 0 else None,
                                 fillcolor="rgba(42,120,214,0.15)" if k == 0 else None,
                                 hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    _layout(fig, height=height)
    fig.update_yaxes(tickformat=".0%", title="Drawdown from peak")
    return fig


def annual_returns_chart(annual: dict[str, tuple[pd.Series, str]], height: int = 400,
                         y_title: str = "Annual return", tickformat: str = "+.0%",
                         hoverformat: str = "+.1%") -> go.Figure:
    fig = go.Figure()
    for label, (s, color) in annual.items():
        fig.add_trace(go.Bar(x=[str(y) for y in s.index], y=s.values, name=label, marker_color=color,
                             hovertemplate="%{y:" + hoverformat + "}<extra>%{fullData.name}</extra>"))
    _layout(fig, height=height, barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(tickformat=tickformat, title=y_title)
    fig.update_xaxes(type="category")
    return fig


def monthly_heatmap(table: pd.DataFrame, height: int | None = None) -> go.Figure:
    z = table.to_numpy(dtype="float64")
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.1
    lim = max(lim, 0.01)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(table.columns), y=[str(y) for y in table.index],
        colorscale=[[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]], zmid=0, zmin=-lim, zmax=lim,
        text=np.where(np.isfinite(z), z, np.nan), texttemplate="%{text:.1%}", textfont=dict(size=11),
        xgap=2, ygap=2, hovertemplate="%{y} %{x}: %{z:+.1%}<extra></extra>",
        colorbar=dict(tickformat="+.0%", thickness=12, outlinewidth=0),
    ))
    _layout(fig, height=height or max(260, 34 * len(table) + 80), hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top")
    return fig


def composition_chart(weights: pd.DataFrame, max_series: int = 8, height: int = 430) -> go.Figure:
    """Stacked area of portfolio weights over time (top holdings by average weight, rest as Other)."""
    fig = go.Figure()
    if weights.empty:
        return _layout(fig, height=height)
    order = weights.mean().sort_values(ascending=False).index.tolist()
    top = order[:max_series]
    other = weights[order[max_series:]].sum(axis=1) if len(order) > max_series else None
    cash = (1.0 - weights.sum(axis=1)).clip(lower=0)
    for k, t in enumerate(top):
        fig.add_trace(go.Scatter(x=weights.index, y=weights[t].values, name=t, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=SERIES[k], shape="hv"),
                                 fillcolor=SERIES[k], hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    if other is not None and other.sum() > 0:
        fig.add_trace(go.Scatter(x=weights.index, y=other.values, name=f"Other ({len(order) - max_series})",
                                 mode="lines", stackgroup="one", line=dict(width=0.5, color=OTHER, shape="hv"),
                                 fillcolor=OTHER, hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    if cash.max() > 0.005:
        fig.add_trace(go.Scatter(x=weights.index, y=cash.values, name="Cash", mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color="#e1e0d9", shape="hv"), fillcolor="#e1e0d9",
                                 hovertemplate="%{y:.1%}<extra>Cash</extra>"))
    _layout(fig, height=height)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Weight")
    return fig


def holdings_bar(weights: pd.Series, height: int | None = None) -> go.Figure:
    w = weights[weights > 0].sort_values()
    fig = go.Figure(go.Bar(x=w.values, y=w.index.tolist(), orientation="h", marker_color=ROLE["active"],
                           text=[f"{v:.1%}" for v in w.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    _layout(fig, height=height or max(220, 26 * len(w) + 60), hovermode="closest", bargap=0.3)
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.1, float(w.max()) * 1.25)], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def trades_chart(price: pd.Series, trades: pd.DataFrame, ticker: str, symbol: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name=f"{ticker} price", mode="lines",
                             line=dict(color=ROLE["index"], width=1.6),
                             hovertemplate=_money_fmt(symbol) + "<extra>%{fullData.name}</extra>"))
    for action, color, sym in (("BUY", GOOD, "triangle-up"), ("SELL", BAD, "triangle-down")):
        t = trades[trades["Action"] == action]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["Date"], y=t["Price"], name=f"{action} ▲" if action == "BUY" else f"{action} ▼", mode="markers",
            marker=dict(symbol=sym, size=11, color=color, line=dict(color="white", width=1.5)),
            customdata=np.c_[t["Quantity"].values, t["Value"].values],
            hovertemplate=(f"{action} at " + symbol + "%{y:,.2f}<br>%{customdata[0]:,.1f} shares, "
                           + symbol + "%{customdata[1]:,.0f}<extra></extra>")))
    _layout(fig, height=height, hovermode="closest")
    fig.update_yaxes(tickprefix=symbol, title=f"Price ({symbol})")
    return fig


def histogram(values: np.ndarray, x_title: str, markers: dict[str, tuple[float, str]],
              symbol: str | None = None, pct: bool = False, height: int = 380) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=40, marker_color=SERIES[0], opacity=0.85,
                                 marker_line=dict(color="white", width=1), name="Random portfolios",
                                 hovertemplate="%{x}: %{y} portfolios<extra></extra>"))
    for k, (label, (x, color)) in enumerate(markers.items()):
        if x is None or not np.isfinite(x):
            continue
        fig.add_vline(x=x, line=dict(color=color, width=2, dash="dash" if k else "solid"))
        fig.add_annotation(x=x, y=1.0 + 0.09 * k, yref="paper", text=label, showarrow=False,
                           font=dict(size=11, color=color), xanchor="left", xshift=4)
    _layout(fig, height=height, hovermode="closest", showlegend=False, bargap=0.02)
    if pct:
        fig.update_xaxes(tickformat="+.0%", title=x_title)
    else:
        fig.update_xaxes(tickprefix=symbol or "", tickformat=",.0f", title=x_title)
    fig.update_yaxes(title="Number of portfolios")
    fig.update_layout(margin=dict(t=60))
    return fig


def money_bars_chart(annual: dict[str, tuple[pd.Series, str]], symbol: str, y_title: str,
                     height: int = 380) -> go.Figure:
    """Grouped bars of a money amount per year, e.g. dividend income."""
    fig = go.Figure()
    for label, (s, color) in annual.items():
        fig.add_trace(go.Bar(x=[str(y) for y in s.index], y=s.values, name=label, marker_color=color,
                             hovertemplate=_money_fmt(symbol) + "<extra>%{fullData.name}</extra>"))
    _layout(fig, height=height, barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(tickprefix=symbol, tickformat=",.0f", title=y_title)
    fig.update_xaxes(type="category")
    return fig


def ratio_chart(series: dict[str, tuple[pd.Series, str]], y_title: str, height: int = 340,
                reference: float | None = None) -> go.Figure:
    """Lines of a multiple over time, e.g. leverage (2.0x)."""
    fig = go.Figure()
    for label, (s, color) in series.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate="%{y:.2f}x<extra>%{fullData.name}</extra>"))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=MUTED, width=1, dash="dot"))
    _layout(fig, height=height)
    fig.update_yaxes(ticksuffix="x", tickformat=".1f", title=y_title, rangemode="tozero")
    return fig


def equity_with_calls_chart(series: dict[str, tuple[pd.Series, str]], calls: pd.DataFrame, symbol: str,
                            log: bool = False, height: int = 430) -> go.Figure:
    """Equity curves with margin calls (red) and wipe-outs (black) marked on the first series."""
    fig = growth_chart(series, symbol, log=log, height=height)
    if calls is not None and len(calls):
        first = next(iter(series.values()))[0]
        for kind, color, sym in (("margin call", BAD, "x"), ("wiped out", "#0b0b0b", "x-open")):
            c = calls[calls["Type"] == kind]
            if c.empty:
                continue
            y = first.reindex(pd.to_datetime(c["Date"])).to_numpy()
            fig.add_trace(go.Scatter(x=pd.to_datetime(c["Date"]), y=y, name=f"{kind.capitalize()} ✕", mode="markers",
                                     marker=dict(symbol=sym, size=12, color=color, line=dict(color=color, width=2)),
                                     customdata=np.c_[c["Sold"].to_numpy(), c["Leverage before"].to_numpy()],
                                     hovertemplate=(f"{kind} on %{{x|%Y-%m-%d}}<br>sold {symbol}%{{customdata[0]:,.0f}} "
                                                    "at %{customdata[1]:.2f}x<extra></extra>")))
    return fig


def money_lines_chart(series: dict[str, tuple[pd.Series, str]], symbol: str, y_title: str,
                      height: int = 380) -> go.Figure:
    """Lines of a money amount over time, e.g. cumulative dividends."""
    fig = go.Figure()
    for label, (s, color) in series.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=_money_fmt(symbol) + "<extra>%{fullData.name}</extra>"))
    _layout(fig, height=height)
    fig.update_yaxes(tickprefix=symbol, tickformat=",.0f", title=y_title)
    return fig
