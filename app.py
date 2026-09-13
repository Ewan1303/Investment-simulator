"""
app.py - Stock Active Management Simulator (Streamlit UI).

    streamlit run app.py

Answers one question: if I actively managed the largest companies in an investment
universe, would I have made more money than simply holding the universe?
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import charts
from analytics import (annual_costs, annual_dividend_yield, annual_dividends, annual_returns, cagr, deflate_result,
                       deflate_series, deflator, drawdown_series, max_drawdown, metrics_table, monthly_returns,
                       result_metrics, series_metrics, stocks_owned, years_spanned)
from backtester import ALL, Backtester, BacktestConfig, LEVERAGE_METHODS, REBALANCE_OPTIONS, pool_label
from data import MARKET_BENCHMARKS, UNIVERSES, FreeDataProvider, benchmark_values, build_market_data
from strategies import MOMENTUM_LOOKBACKS, STRATEGIES, TREND_WINDOWS, make_strategy

st.set_page_config(page_title="Active Management Simulator", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
  div[data-testid="stMetric"] { background: rgba(137,135,129,0.07); border-radius: 10px; padding: 14px 18px; }
  div[data-testid="stMetricLabel"] p { font-size: 0.85rem; color: #898781; }
  div[data-testid="stMetricValue"] { font-size: 1.7rem; }
  section[data-testid="stSidebar"] { min-width: 340px; }
  h1 { margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

TOP_N_OPTIONS = [1, 5, 10, 25, 50, 100, 250, 500]
HOLDINGS_OPTIONS = [1, 2, 3, 5, 10, 20, 50, 100]
MAX_COMPARE = 12   # colours repeat after 8, so entries 9-12 are drawn dashed
CURRENCY_MODES = {"GBP (fixed rate)": "GBP_FIXED", "GBP (actual rate)": "GBP", "USD": "USD"}


# ---------------------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def provider() -> FreeDataProvider:
    return FreeDataProvider()


@st.cache_resource(show_spinner=False)
def universe_size(universe: str, custom: tuple) -> int:
    return provider().universe_size(universe, list(custom))


@st.cache_resource(show_spinner=False, max_entries=3)
def market_data(universe: str, custom: tuple, start: str, end: str, currency: str, _progress=None):
    return build_market_data(provider(), universe, start, end, currency, list(custom), progress=_progress)


def parse_custom(text: str) -> tuple:
    return tuple(sorted({t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()}))


def parse_ranges(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """'1-1, 2-2, 3-10, 25-end, 50, all' -> [(1, 1), (2, 2), (3, 10), (25, ALL), (1, 50), (1, ALL)]."""
    pools: list[tuple[int, int]] = []
    bad: list[str] = []
    for tok in text.replace(";", ",").split(","):
        t = tok.strip().lower().replace(" ", "")
        if not t:
            continue
        if t == "all":
            pools.append((1, ALL))
            continue
        m = re.fullmatch(r"(\d+)(-(\d+|end|all)?)?", t)
        if not m or int(m.group(1)) < 1:
            bad.append(tok.strip())
            continue
        a = int(m.group(1))
        if m.group(2) is None:                       # a single number means Top N
            pools.append((1, a))
        else:
            b = m.group(3)
            hi = ALL if b in (None, "end", "all") else int(b)
            if hi != ALL and hi < a:
                a, hi = hi, a
            pools.append((a, hi))
    return pools, bad


def money(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.0f}"


def pct(x: float, signed: bool = True) -> str:
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:+.1%}" if signed else f"{x:.1%}")


# ---------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------
with st.sidebar:
    st.title("Backtest")
    universe = st.selectbox("Universe", UNIVERSES, index=0)
    custom: tuple = ()
    if universe == "Custom":
        custom = parse_custom(st.text_input("Tickers (comma-separated)", "AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, XOM"))
    try:
        size = max(universe_size(universe, custom), 1)
    except Exception as exc:
        st.error(f"Could not load the universe: {exc}")
        st.stop()

    st.markdown("Companies by market cap at the time (rank 1 = largest)")
    st.session_state.setdefault("rank_from", 1)
    st.session_state.setdefault("rank_to", min(10, size))
    st.session_state["rank_from"] = min(int(st.session_state["rank_from"]), size)
    st.session_state["rank_to"] = min(int(st.session_state["rank_to"]), size)
    c1, c2, c3 = st.columns([1, 1, 0.55])
    rank_from = int(c1.number_input("From rank", 1, size, key="rank_from", help="1 = the largest company"))
    rank_to = int(c2.number_input("To rank", 1, size, key="rank_to",
                                  help=f"{size} = the whole universe. 25 to 50 = the 25th to 50th largest."))
    c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    c3.button("Max", width="stretch", help="Whole universe",
              on_click=lambda: st.session_state.update(rank_to=size))
    if rank_to < rank_from:
        rank_from, rank_to = rank_to, rank_from
    top_n = rank_to
    whole = rank_from == 1 and rank_to >= size
    pool_size = rank_to - rank_from + 1
    pool_name = "All" if whole else (f"Top {rank_to}" if rank_from == 1 else f"Ranks {rank_from}-{rank_to}")
    st.caption(f"Pool: {pool_name} ({pool_size} companies) of the {size} in the universe today.")

    strategy_name = st.selectbox("Strategy", list(STRATEGIES))
    strategy_params: dict = {}
    if strategy_name == "Momentum":
        strategy_params["lookback_months"] = st.select_slider("Momentum lookback (months)", MOMENTUM_LOOKBACKS, value=6)
    elif strategy_name == "Trend Following":
        strategy_params["ma_days"] = st.select_slider("Moving average (days)", TREND_WINDOWS, value=200)
    seed = 0
    if strategy_name == "Random Selection":
        seed = int(st.number_input("Random seed", 0, 1_000_000, 0))
    st.caption(STRATEGIES[strategy_name].description)

    h_options = sorted({h for h in HOLDINGS_OPTIONS if h < pool_size} | {pool_size})
    n_holdings = st.select_slider("Number of holdings", options=h_options,
                                  value=5 if 5 in h_options else h_options[-1])
    buy_hold = strategy_name == "Buy & Hold"
    rebalance = st.selectbox("Rebalance", REBALANCE_OPTIONS, index=5 if buy_hold else 2, disabled=buy_hold,
                             help="How often the active strategy re-selects and re-weights its holdings. The "
                                  "passive benchmark lines are always refreshed quarterly, like an index fund, so "
                                  "they stay the same while you experiment. To test a cap-weighted portfolio at "
                                  "another frequency, choose Market Cap Weighted as the strategy.")

    currency_choice = st.radio("Currency", list(CURRENCY_MODES), index=0,
                               help="US stocks are priced in dollars. 'GBP (fixed rate)' shows pounds but keeps "
                                    "the exchange rate constant, so you see only the stocks' gains. 'GBP (actual "
                                    "rate)' converts at each day's real rate, so the pound's rise or fall is "
                                    "included. 'USD' is the stocks' own currency.")
    currency = CURRENCY_MODES[currency_choice]
    symbol = "$" if currency == "USD" else "£"
    st.checkbox("Adjust for inflation", value=False, key="inflation_adj",
                help="Show every value in start-year money using consumer price inflation (US CPI for dollars, "
                     "UK CPI for pounds), so returns reflect what the money could actually buy. Applies instantly "
                     "to the last run; no need to re-run.")
    capital = float(st.number_input(f"Starting capital ({symbol})", min_value=1_000, value=100_000, step=10_000))
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start date", date(2005, 1, 1), min_value=date(1997, 1, 1), max_value=date.today())
    end_date = c2.date_input("End date", date.today(), min_value=date(1998, 1, 1), max_value=date.today())
    c1, c2 = st.columns(2)
    tx_cost = c1.number_input("Transaction cost (%)", 0.0, 5.0, 0.10, 0.05, format="%.2f") / 100
    slippage = c2.number_input("Slippage (%)", 0.0, 5.0, 0.05, 0.05, format="%.2f") / 100

    with st.expander("Leverage (optional)"):
        leverage = float(st.number_input("Leverage multiple", 1.0, 5.0, 1.0, 0.25, format="%.2f",
                                         help="Gross exposure as a multiple of your equity. 1 = no borrowing, "
                                              "2 = borrow as much again as you put in."))
        leverage_method = st.radio("Method", LEVERAGE_METHODS, index=0, disabled=leverage <= 1.0,
                                   help="Reset at rebalance: leverage is set back to the multiple at each rebalance "
                                        "and drifts in between. Daily reset: like a leveraged ETF, reset every "
                                        "day (no margin calls, but volatility drag). Initial loan only: borrow "
                                        "once at the start and never again.")
        borrow_rate = st.number_input("Borrowing rate (% per year)", 0.0, 20.0, 4.0, 0.5,
                                      disabled=leverage <= 1.0) / 100
        maintenance = st.number_input("Maintenance margin (%)", 5.0, 60.0, 25.0, 5.0, disabled=leverage <= 1.0,
                                      help="Margin call when equity falls below this share of gross exposure. "
                                           "US brokers require at least 25%.") / 100
        if leverage > 1.0:
            st.caption(f"Only the active strategy is leveraged; benchmarks stay at 1x. At {leverage:.2f}x a "
                       f"{1 / leverage - maintenance:.0%} fall in the holdings between rebalances triggers a "
                       f"margin call." if leverage_method != "Daily reset" else
                       f"Only the active strategy is leveraged; benchmarks stay at 1x. A one-day fall of "
                       f"{1 / leverage:.0%} would wipe the account out.")

    run_clicked = st.button("RUN BACKTEST", type="primary", width="stretch")

    st.divider()
    with st.expander("Data"):
        info = provider().price_cache_info()
        st.caption(f"Prices cached for {info['tickers']} tickers (downloaded {info['as_of'] or 'never'}); "
                   f"{info['failed']} tickers have no Yahoo data.")
        st.caption("Sources: Yahoo Finance prices, SEC filings for share counts, point-in-time S&P 500 "
                   "constituents from GitHub (fja05680/sp500). The first download takes a few minutes.")
        if st.button("Refresh price data (re-download)"):
            provider().clear_price_cache()
            market_data.clear()
            st.session_state.pop("run", None)
            st.rerun()


# ---------------------------------------------------------------------------------------
# Run the backtest
# ---------------------------------------------------------------------------------------
st.title("Stock Active Management Simulator")
st.caption("If I actively managed the largest companies in this universe, would I have made more money "
           "than simply holding the universe?")

if run_clicked:
    if (end_date - start_date).days < 180:
        st.error("Please choose an end date at least six months after the start date.")
        st.stop()
    status = st.status("Loading market data ...", expanded=True)
    bar = status.progress(0.0)
    note = status.empty()

    def progress(frac: float, msg: str) -> None:
        bar.progress(min(max(float(frac), 0.0), 1.0))
        note.write(msg)

    try:
        md = market_data(universe, custom, str(start_date), str(end_date), currency, _progress=progress)
        status.update(label="Market data ready", state="complete", expanded=False)
        strategy = make_strategy(strategy_name, **strategy_params)
        cfg = BacktestConfig(top_n=ALL if whole else top_n, rank_from=rank_from,
                             n_holdings=ALL if (whole and n_holdings >= size) else n_holdings,
                             rebalance=rebalance, start=pd.Timestamp(start_date), end=pd.Timestamp(end_date),
                             capital=capital, tx_cost=tx_cost, slippage=slippage, seed=seed,
                             leverage=leverage, leverage_method=leverage_method, borrow_rate=borrow_rate,
                             maintenance_margin=maintenance)
        top_label = pool_name
        label = f"{strategy.label()} | {universe} {top_label} → {n_holdings} holdings, {strategy.rebalance_override or rebalance}"
        if cfg.leveraged:
            label += f", {leverage:g}x leverage ({leverage_method.lower()})"
        with st.spinner("Running backtest and benchmarks ..."):
            results = Backtester(md).run_with_benchmarks(cfg, strategy, label)
        st.session_state["run"] = {"md": md, "results": results, "cfg": cfg, "label": label,
                                   "strategy_name": strategy_name, "strategy_params": strategy_params,
                                   "universe": universe, "size": size}
        st.session_state.pop("mc", None)
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(f"Backtest failed: {exc}")
        st.stop()
    st.rerun()   # redraw without the progress box so later clicks keep the selected tab

state = st.session_state.get("run")
if not state:
    st.markdown("""
Pick a universe and strategy in the sidebar, then press **RUN BACKTEST**.

At every rebalance the simulator:

1. finds the universe's constituents **on that historical date**,
2. keeps the **N largest by market cap** at the time,
3. ranks them with the chosen strategy and buys the best **K**,
4. fills the trades at the *next* day's close, paying transaction costs and slippage.

Only information available on the decision date is ever used. Every run is compared with the
passive cap-weighted Top-N, the whole universe and the S&P 500 ETF from the same starting capital.

**Currency.** US stocks are priced in dollars. The default, *GBP (fixed rate)*, shows pounds but keeps the
exchange rate constant, so you see only the stocks' gains. *GBP (actual rate)* adds the pound's own rise
or fall over the period, which is what a UK investor really experienced. *USD* matches published S&P 500
figures. A full explanation appears under the main chart after a run.
""")
    st.stop()

md = state["md"]
res = state["results"]
cfg: BacktestConfig = state["cfg"]
symbol = md.symbol
active = res["active"]
passive = res.get("passive")
universe_res = res["universe"]
index_series = res["index"]
inflation_on = bool(st.session_state.get("inflation_adj")) and md.cpi is not None
defl = None
if inflation_on:   # display everything in start-year money; the engine itself stays nominal
    defl = deflator(md.cpi, active.values.index)
    active = deflate_result(active, defl)
    universe_res = deflate_result(universe_res, defl)
    passive = deflate_result(passive, defl) if passive is not None else None
    index_series = deflate_series(index_series, defl)
reference = passive or universe_res

if md.warnings:
    st.warning("**Data caveats**\n\n" + "\n\n".join(f"- {w}" for w in md.warnings))
if active.wiped_out is not None:
    st.error(f"**Wiped out on {active.wiped_out.date()}.** The leveraged portfolio's equity fell to zero, so the "
             f"run stays at zero from that day. See the Leverage tab for the forced sales that led there.")
elif cfg.leveraged and len(active.margin_calls):
    st.info(f"This run had {len(active.margin_calls)} margin call(s). See the Leverage tab.")

# ---- headline statistics --------------------------------------------------------------
m_active = result_metrics(active)
m_ref = series_metrics(reference.values)
ref_name = f"passive {pool_label(cfg.rank_from, cfg.top_n)}" if passive is not None else "universe"
row1 = st.columns(3)
row1[0].metric("Final value", money(m_active["Final value"], symbol),
               delta=f"{money(m_active['Final value'] - m_ref['Final value'], symbol)} vs {ref_name}")
row1[1].metric("CAGR", pct(m_active["CAGR"]),
               delta=f"{(m_active['CAGR'] - m_ref['CAGR']) * 100:+.1f} pp vs {ref_name}")
row1[2].metric("Total return", pct(m_active["Total return"]),
               delta=f"{(m_active['Total return'] - m_ref['Total return']) * 100:+.0f} pp vs {ref_name}")
row2 = st.columns(3)
row2[0].metric("Max drawdown", pct(m_active["Max drawdown"]),
               delta=f"{(m_active['Max drawdown'] - m_ref['Max drawdown']) * 100:+.1f} pp vs {ref_name}")
row2[1].metric("Sharpe ratio", f"{m_active['Sharpe ratio']:.2f}",
               delta=f"{m_active['Sharpe ratio'] - m_ref['Sharpe ratio']:+.2f} vs {ref_name}")
row2[2].metric("Trades", f"{m_active['Number of trades']:,}",
               delta=f"{money(m_active['Transaction costs'], symbol)} in costs", delta_color="off")

# ---- main chart ------------------------------------------------------------------------
lines = {active.label: (active.values, charts.ROLE["active"])}
if passive is not None:
    lines[passive.label] = (passive.values, charts.ROLE["passive"])
lines[universe_res.label] = (universe_res.values, charts.ROLE["universe"])
lines[md.benchmark_label] = (index_series, charts.ROLE["index"])

head, toggle = st.columns([6, 1])
head.subheader(f"Portfolio value (in {active.values.index[0].year} money)" if inflation_on else "Portfolio value")
log_scale = toggle.toggle("Log scale", value=False)
st.plotly_chart(charts.growth_chart(lines, symbol, log=log_scale), width="stretch")
basis = f"Total return with dividends reinvested, shown in {md.currency_label}."
if md.fx_mode == "fixed":
    basis += (f" £1 = ${float(md.fx.iloc[0]):.2f} throughout, so exchange-rate moves are excluded and these are "
              f"the stocks' own gains, just counted in pounds.")
elif md.fx_mode == "actual":
    f0, f1 = float(md.fx.loc[active.values.index[0]]), float(md.fx.loc[active.values.index[-1]])
    basis += (f" GBP/USD went from {f0:.2f} to {f1:.2f} over this period, which multiplied sterling results by "
              f"{f0 / f1:.2f}x on top of what the stocks did.")
basis += " Index returns quoted online are usually price-only and in US dollars, so they look lower."
if cfg.leveraged:
    basis += (f" The active strategy is run at {cfg.leverage:g}x leverage ({cfg.leverage_method.lower()}); "
              f"the benchmark lines are unleveraged.")
if inflation_on:
    cum_infl = float(1.0 / defl.iloc[-1] - 1.0)
    basis += (f" **Adjusted for inflation**: every value is in {active.values.index[0].year} money. Consumer prices "
              f"rose {cum_infl:.0%} over the period ({md.cpi_label}), so {symbol}100,000 at the start buys what "
              f"{symbol}{100_000 * (1 + cum_infl):,.0f} buys at the end.")
elif st.session_state.get("inflation_adj"):
    basis += " Inflation adjustment was requested but CPI data is unavailable, so values are nominal."
st.caption(basis)
with st.expander("Why the currency setting changes the numbers"):
    real_fx = FreeDataProvider().get_fx().reindex(active.values.index).ffill().bfill()
    r0, r1 = float(real_fx.iloc[0]), float(real_fx.iloc[-1])
    st.markdown(f"""
US stocks are priced in dollars, so a stock's own return is its **dollar** return. Asking for the result in
pounds means starting with pounds, buying dollars, and converting back at the end, and the exchange rate
on those two days is usually different. On {active.values.index[0].date()} £1 bought ${r0:.2f}; on
{active.values.index[-1].date()} it bought ${r1:.2f}.

| Setting | What it shows |
|---|---|
| **USD** | The stocks as quoted. Same percentages as published S&P 500 total-return figures. |
| **GBP (fixed rate)** | Pounds, but with one exchange rate for the whole period. Same percentages as USD, just a different unit. Like a currency-hedged fund. |
| **GBP (actual rate)** | Pounds converted at each day's real rate. The pound's own rise or fall is part of the result, so this is what an unhedged UK investor actually experienced: over this period the exchange-rate move alone multiplied results by **{r0 / r1:.2f}x**. |

Two things catch people out. First, £100,000 and $100,000 are different amounts of money: in
{active.values.index[0].year} £100,000 was ${100_000 * r0:,.0f}, so a pound investor starts with more dollars
than a dollar investor. Second, the ordering of strategies never changes with the currency, because every
line is converted the same way on the same days; only the absolute level moves.
""")

# ---- tabs ----------------------------------------------------------------------------------
tabs = st.tabs(["Performance", "Drawdown", "Annual returns", "Monthly returns", "Dividends", "Portfolio", "Trades",
                "Costs", "Leverage", "Monte Carlo", "Compare"])

with tabs[0]:
    st.plotly_chart(charts.cumulative_return_chart(lines), width="stretch")
    def bench_metrics(r):
        m = series_metrics(r.values)
        y = annual_dividend_yield(r.dividends, r.values)
        m["Dividends received"] = float(r.dividends.sum())
        m["Average dividend yield"] = float(y.mean()) if len(y) else float("nan")
        return m

    columns = {active.label: m_active}
    if passive is not None:
        columns[passive.label] = bench_metrics(passive)
    columns[universe_res.label] = bench_metrics(universe_res)
    columns[md.benchmark_label] = series_metrics(index_series)
    st.dataframe(metrics_table(columns, symbol), width="stretch", height=560)
    st.caption("Sharpe and Sortino assume a 0% risk-free rate. Turnover = (buys + sells) / 2 / average "
               "portfolio value per year. Benchmarks are cap-weighted, rebalanced quarterly, and pay the same costs.")

with tabs[1]:
    dds = {label: (drawdown_series(s), color) for label, (s, color) in lines.items()}
    st.plotly_chart(charts.drawdown_chart(dds), width="stretch")
    rows = []
    for label, (s, _) in lines.items():
        dd = drawdown_series(s)
        trough = dd.idxmin()
        peak = s.loc[:trough].idxmax()
        after = s.loc[trough:]
        recovered = after[after >= s.loc[peak]]
        rows.append({"Series": label, "Max drawdown": pct(dd.min()), "Peak": str(peak.date()),
                     "Trough": str(trough.date()),
                     "Recovered": str(recovered.index[0].date()) if len(recovered) else "not yet"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tabs[2]:
    ann = {label: (annual_returns(s), color) for label, (s, color) in lines.items()}
    st.plotly_chart(charts.annual_returns_chart(ann), width="stretch")
    table = pd.DataFrame({label: s for label, (s, _) in ann.items()})
    st.dataframe(table.map(lambda v: pct(v)), width="stretch")

with tabs[3]:
    which = st.selectbox("Series", list(lines), index=0, key="heat_series")
    st.plotly_chart(charts.monthly_heatmap(monthly_returns(lines[which][0])), width="stretch")

with tabs[4]:
    div_series = {active.label: (active.dividends, charts.ROLE["active"])}
    if passive is not None:
        div_series[passive.label] = (passive.dividends, charts.ROLE["passive"])
    div_series[universe_res.label] = (universe_res.dividends, charts.ROLE["universe"])
    total_div = float(active.dividends.sum())
    ann_div = annual_dividends(active.dividends)
    ann_yield = annual_dividend_yield(active.dividends, active.values)
    full_years = ann_div.iloc[:-1] if len(ann_div) > 1 else ann_div
    k = st.columns(4)
    k[0].metric("Dividends received", money(total_div, symbol),
                help="All cash dividends paid to the portfolio over the backtest, before reinvestment")
    k[1].metric("Share of final value", pct(total_div / active.final_value, signed=False))
    k[2].metric("Average yield on portfolio", pct(float(ann_yield.mean()) if len(ann_yield) else float("nan"), signed=False),
                help="Dividends received each year divided by that year's average portfolio value")
    k[3].metric("Last full year's income", money(float(full_years.iloc[-1]), symbol))
    st.caption("Dividends are reinvested automatically on the day they are paid (the simulation runs on "
               "total-return prices), so they are already inside the portfolio value. These figures show the "
               f"cash received, in {md.currency_label}.")
    st.plotly_chart(charts.money_bars_chart({lab: (annual_dividends(s), c) for lab, (s, c) in div_series.items()},
                                            symbol, "Dividend income per year"), width="stretch")
    st.plotly_chart(charts.money_lines_chart({lab: (s.cumsum(), c) for lab, (s, c) in div_series.items()},
                                             symbol, "Cumulative dividends received"), width="stretch")
    table = pd.DataFrame({"Dividends received": ann_div.map(lambda v: money(v, symbol)),
                          "Yield on portfolio": ann_yield.reindex(ann_div.index).map(lambda v: pct(v, signed=False))})
    table.index.name = "Year"
    st.dataframe(table, width="stretch")

with tabs[5]:
    st.markdown("**Portfolio composition over time** (weights set at each rebalance)")
    st.plotly_chart(charts.composition_chart(active.weights), width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        if not active.weights.empty:
            last = active.weights.iloc[-1]
            st.markdown(f"**Holdings at last rebalance** ({pd.Timestamp(active.weights.index[-1]).date()})")
            st.plotly_chart(charts.holdings_bar(last), width="stretch")
    with c2:
        if active.pool:
            when = list(active.pool)[-1]
            pool = active.pool[when]
            i = md.index_of(when)
            caps = [md.mcap[i, md.tickers.index(t)] / 1e9 for t in pool]
            st.markdown(f"**Pool on {pd.Timestamp(when).date()}**: {pool_label(cfg.rank_from, cfg.top_n)}, "
                        f"{len(pool)} companies (market cap, US$ bn)")
            st.dataframe(pd.DataFrame({"Rank": range(1, len(pool) + 1), "Ticker": pool,
                                       "Market cap ($bn)": [f"{c:,.0f}" for c in caps]}),
                         width="stretch", hide_index=True, height=min(420, 38 * len(pool) + 40))
    with st.expander("Data coverage (survivorship check)"):
        st.caption("Constituents of the universe at the start of each year and how many of them have price data. "
                   "Missing companies are mostly ones that were later delisted or acquired.")
        cov = md.coverage.copy()
        cov["Coverage"] = cov["Coverage"].map(lambda v: pct(v, signed=False))
        st.dataframe(cov, width="stretch", hide_index=True)
        src = md.shares_source.value_counts()
        st.caption("Share-count sources: " + ", ".join(f"{k}: {v}" for k, v in src.items())
                   + f". Prices downloaded {md.prices_as_of}.")

with tabs[6]:
    trades = active.trades
    if trades.empty:
        st.info("No trades were made.")
    else:
        counts = trades.loc[trades["Ticker"].isin(md.tickers), "Ticker"].value_counts()
        tk = st.selectbox("Ticker", counts.index.tolist(), index=0, key="trade_ticker",
                          help="'PORTFOLIO' rows in the table are cash sweeps: a uniform slice of every holding "
                               "bought or sold to keep the portfolio fully invested.")
        j = md.tickers.index(tk)
        lo, hi = md.index_of(cfg.start), md.index_of(cfg.end)
        price = pd.Series(md.close[lo:hi + 1, j], index=md.dates[lo:hi + 1]).dropna()
        if inflation_on:
            price = deflate_series(price, defl)
        st.plotly_chart(charts.trades_chart(price, trades[trades["Ticker"] == tk], tk, symbol), width="stretch")
        show = trades.copy()
        show["Date"] = pd.to_datetime(show["Date"]).dt.date
        show = show.round({"Price": 4, "Quantity": 4, "Value": 2, "Cost": 2})
        st.dataframe(show, width="stretch", hide_index=True, height=420,
                     column_config={"Price": st.column_config.NumberColumn(format=f"{symbol}%.2f"),
                                    "Quantity": st.column_config.NumberColumn(format="%.2f"),
                                    "Value": st.column_config.NumberColumn(format=f"{symbol}%.0f"),
                                    "Cost": st.column_config.NumberColumn(format=f"{symbol}%.2f")})
        c1, c2 = st.columns([1, 3])
        c1.download_button("Download trades as CSV", data=show.to_csv(index=False).encode("utf-8"),
                           file_name="trades.csv", mime="text/csv", width="stretch")
        c2.caption(f"{len(trades):,} trades, {money(active.costs_paid, symbol)} paid in costs, "
                   f"{money(active.turnover_value, symbol)} traded in total. Prices are split-adjusted closes in "
                   f"{md.currency_label}.")

with tabs[7]:
    tx_share = cfg.tx_cost / cfg.cost_rate if cfg.cost_rate > 0 else 0.0
    total_trading = active.costs_paid
    total_costs = total_trading + active.interest_paid
    yrs = years_spanned(active.values)
    owned_avg = float(stocks_owned(active).mean())
    k = st.columns(4)
    k[0].metric("Total costs", money(total_costs, symbol),
                help="Commission and slippage on every trade, plus loan interest if leveraged")
    k[1].metric("Commission", money(total_trading * tx_share, symbol),
                delta=f"{money(total_trading * (1 - tx_share), symbol)} slippage", delta_color="off")
    k[2].metric("Loan interest", money(active.interest_paid, symbol))
    k[3].metric("Costs per year vs stocks owned",
                f"{total_costs / owned_avg / yrs:.2%}" if owned_avg > 0 else "n/a",
                help="Total costs per year divided by the average value of the stocks held, like a fund's expense ratio")
    with st.spinner("Running the same strategy with zero trading costs ..."):
        free = Backtester(md).run(replace(cfg, tx_cost=0.0, slippage=0.0),
                                  make_strategy(state["strategy_name"], **state["strategy_params"]),
                                  "Same strategy, no trading costs")
        if inflation_on:
            free = deflate_result(free, defl)
    k2 = st.columns(3)
    k2[0].metric("Final value with zero trading costs", money(free.final_value, symbol),
                 delta=f"{money(active.final_value - free.final_value, symbol)} given up to costs")
    k2[1].metric("CAGR given up to trading costs", f"{(cagr(free.values) - cagr(active.values)) * 100:.2f} pp per year")
    k2[2].metric("Turnover per year", pct(m_active["Turnover (per year)"], signed=False),
                 help="(Buys + sells) / 2 as a share of the average portfolio value, per year")
    st.caption(f"Costs are charged at {cfg.tx_cost:.2%} commission plus {cfg.slippage:.2%} slippage on every trade "
               f"(including forced sales and cash sweeps). Benchmarks pay the same rates; the SPY ETF line pays none.")

    cost_runs = [(active.label, active, charts.ROLE["active"])]
    if passive is not None:
        cost_runs.append((passive.label, passive, charts.ROLE["passive"]))
    cost_runs.append((universe_res.label, universe_res, charts.ROLE["universe"]))
    yearly = {lab: annual_costs(r) for lab, r, _ in cost_runs}

    def daily_costs(r) -> pd.Series:
        if len(r.trades):
            s = pd.Series(r.trades["Cost"].to_numpy(), index=pd.to_datetime(r.trades["Date"])).groupby(level=0).sum()
            s = s.reindex(r.values.index, fill_value=0.0)
        else:
            s = pd.Series(0.0, index=r.values.index)
        return s + r.interest

    st.plotly_chart(charts.money_bars_chart({lab: (yearly[lab]["Total costs"], c) for lab, _, c in cost_runs},
                                            symbol, "Costs per year"), width="stretch")
    st.plotly_chart(charts.annual_returns_chart({lab: (yearly[lab]["Cost ratio"], c) for lab, _, c in cost_runs},
                                                y_title="Costs as a share of the stocks owned", tickformat=".2%",
                                                hoverformat=".3%"), width="stretch")
    st.plotly_chart(charts.money_lines_chart({lab: (daily_costs(r).cumsum(), c) for lab, r, c in cost_runs},
                                             symbol, "Cumulative costs"), width="stretch")
    show = yearly[active.label].copy()
    for c in ["Trading costs", "Interest", "Total costs", "Stocks owned (avg)"]:
        show[c] = show[c].map(lambda v: money(v, symbol))
    show["Cost ratio"] = show["Cost ratio"].map(lambda v: f"{v:.3%}" if pd.notna(v) else "n/a")
    show["Trades"] = show["Trades"].astype(int)
    show = show[["Trades", "Trading costs", "Interest", "Total costs", "Stocks owned (avg)", "Cost ratio"]]
    show.index.name = "Year"
    st.markdown("**Active strategy, year by year**")
    st.dataframe(show, width="stretch")

with tabs[8]:
    if not cfg.leveraged:
        st.info("This run uses no leverage. Open **Leverage (optional)** in the sidebar and set a multiple above 1 "
                "to borrow against the portfolio: daily reset (like a leveraged ETF), reset at each rebalance, or "
                "a one-off loan. This tab then shows the interest, margin calls and any wipe-out it produces.")
    else:
        calls = active.margin_calls
        k = st.columns(4)
        k[0].metric("Average leverage", f"{np.nanmean(active.leverage_path.to_numpy()):.2f}x")
        k[1].metric("Interest paid", money(active.interest_paid, symbol))
        k[2].metric("Margin calls", f"{int((calls['Type'] == 'margin call').sum())}")
        k[3].metric("Wiped out", active.wiped_out.date().isoformat() if active.wiped_out is not None else "No")
        st.caption(f"{cfg.leverage:g}x via {cfg.leverage_method.lower()}, borrowing at {cfg.borrow_rate:.1%} a year, "
                   f"maintenance margin {cfg.maintenance_margin:.0%}. Benchmarks are unleveraged.")
        with st.spinner("Running the same strategy without leverage ..."):
            unlev = Backtester(md).run(replace(cfg, leverage=1.0),
                                       make_strategy(state["strategy_name"], **state["strategy_params"]),
                                       "Same strategy, no leverage")
            if inflation_on:
                unlev = deflate_result(unlev, defl)
        series = {active.label: (active.values, charts.ROLE["active"]),
                  "Same strategy, no leverage": (unlev.values, charts.SERIES[6])}
        if passive is not None:
            series[passive.label] = (passive.values, charts.ROLE["passive"])
        series[universe_res.label] = (universe_res.values, charts.ROLE["universe"])
        st.plotly_chart(charts.equity_with_calls_chart(series, calls, symbol, log=log_scale), width="stretch")
        st.plotly_chart(charts.ratio_chart({"Leverage (exposure ÷ equity)": (active.leverage_path, charts.ROLE["active"])},
                                           "Leverage", reference=cfg.leverage), width="stretch")
        if len(calls):
            show = calls.copy()
            show["Date"] = pd.to_datetime(show["Date"]).dt.date
            st.markdown(f"**Forced sales** ({len(calls)})")
            st.dataframe(show, width="stretch", hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format=f"{symbol}%.0f")
                                        for c in ["Equity before", "Exposure before", "Sold", "Equity after"]}
                         | {c: st.column_config.NumberColumn(format="%.2fx") for c in ["Leverage before", "Leverage after"]})
        else:
            st.success("No margin calls in this run.")
        st.caption("A margin call sells positions pro rata until leverage is back to the target (or 10 points above "
                   "the maintenance level if the target is more aggressive than that). Daily reset cannot be "
                   "margin-called, but a one-day fall of more than 1 ÷ leverage wipes the account out. Leveraged "
                   "ETFs also charge fees of around 1% a year; add that to the borrowing rate to mimic them.")

with tabs[9]:
    st.markdown("Run many **random** stock selections from the same pool, with the same number of holdings, "
                "rebalance schedule and costs, to see how the active strategy compares with luck.")
    c1, c2 = st.columns([1, 3])
    n_sims = int(c1.number_input("Number of simulations", 10, 5000, 200, 50))
    if c2.button("Run Monte Carlo", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        mc = Backtester(md).monte_carlo(cfg, n_sims, progress=lambda f, m: bar.progress(f, text=m))
        bar.empty()
        st.session_state["mc"] = mc
    mc = st.session_state.get("mc")
    if mc is not None and inflation_on:   # random outcomes are stored nominal; show them in start-year money too
        mc = mc.copy()
        mc["final_value"] = mc["final_value"] * float(defl.iloc[-1])
        mc["cagr"] = (mc["final_value"] / cfg.capital) ** (1.0 / years_spanned(active.values)) - 1.0
    if mc is not None:
        ref_final = reference.final_value
        ref_cagr = cagr(reference.values)
        uni_final, uni_cagr = universe_res.final_value, cagr(universe_res.values)
        p_beat_ref = float((mc["final_value"] > ref_final).mean())
        p_beat_uni = float((mc["final_value"] > uni_final).mean())
        percentile = float((mc["final_value"] < active.final_value).mean())
        k = st.columns(4)
        k[0].metric("Random beats passive Top-N", pct(p_beat_ref, signed=False),
                    help="Share of random portfolios that finished above the passive Top-N benchmark")
        k[1].metric("Random beats universe", pct(p_beat_uni, signed=False),
                    help="Share of random portfolios that finished above the cap-weighted universe")
        k[2].metric("Active vs random", pct(percentile, signed=False),
                    help="Percentile of the active strategy's final value among the random portfolios")
        k[3].metric("Median random outcome", money(float(mc["final_value"].median()), symbol))
        markers = {"Active strategy": (active.final_value, charts.ROLE["active"]),
                   "Passive Top-N": (ref_final, charts.ROLE["passive"]),
                   "Universe": (uni_final, charts.ROLE["universe"])}
        st.plotly_chart(charts.histogram(mc["final_value"].to_numpy(), f"Final portfolio value ({symbol})", markers, symbol=symbol),
                        width="stretch")
        markers_c = {"Active strategy": (cagr(active.values), charts.ROLE["active"]),
                     "Passive Top-N": (ref_cagr, charts.ROLE["passive"]),
                     "Universe": (uni_cagr, charts.ROLE["universe"])}
        st.plotly_chart(charts.histogram(mc["cagr"].to_numpy(), "CAGR", markers_c, pct=True), width="stretch")
        st.plotly_chart(charts.histogram(mc["max_drawdown"].to_numpy(), "Maximum drawdown",
                                         {"Active strategy": (max_drawdown(active.values), charts.ROLE["active"])}, pct=True),
                        width="stretch")
        q = mc[["final_value", "cagr", "max_drawdown"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        q.index = [f"{int(p * 100)}th percentile" for p in q.index]
        q["final_value"] = q["final_value"].map(lambda v: money(v, symbol))
        q["cagr"] = q["cagr"].map(pct)
        q["max_drawdown"] = q["max_drawdown"].map(pct)
        q.columns = ["Final value", "CAGR", "Max drawdown"]
        st.dataframe(q, width="stretch")

with tabs[10]:
    st.markdown("Collect configurations and compare them side by side: the current run, the current strategy "
                "over different pools, **passive** cap-weighted pools (e.g. the largest company vs the second "
                "largest, or Top 1 vs Top 25, with no active management), and plain market benchmarks.")
    compare: list = st.session_state.setdefault("compare", [])

    def add_to_compare(label: str, values: pd.Series, n_trades: int | None) -> None:
        if any(c["label"] == label for c in compare):
            return
        if len(compare) >= MAX_COMPARE:
            st.warning(f"The comparison holds up to {MAX_COMPARE} series. Clear it to add more.")
            return
        m = series_metrics(values)
        m["Trades"] = n_trades
        compare.append({"label": label, "values": values, "metrics": m})

    def add_result(label: str, result) -> None:
        add_to_compare(label, result.values, result.n_trades)

    st.markdown("**Current run**")
    if st.button("Add current run (active, passive Top-N and universe)"):
        add_result(state["label"], active)
        if passive is not None:
            add_result(passive.label, passive)
        add_result(universe_res.label, universe_res)

    st.markdown("**Pool sweep**: pick Top-N sizes and/or type rank ranges, then run them as the current strategy "
                "or as passive cap-weighted pools")
    sweep_options = [n for n in TOP_N_OPTIONS if n < state["size"]] + ["All"]
    c1, c2 = st.columns([1.1, 1.9])
    sweep = c1.multiselect("Top-N sizes", sweep_options, default=[], key="sweep",
                           help="Ranks 1 to N; 'All' means the whole universe")
    ranges_text = c2.text_input("Rank ranges", "", key="sweep_ranges", placeholder="e.g. 1-1, 2-2, 3-10, 25-50, 100-end",
                                help="Comma-separated. 'a-b' = the a-th to b-th largest company at the time; "
                                     "'a-end' = from rank a down to the smallest; a single number = Top N")
    pools: list[tuple[int, int]] = [(1, ALL if n == "All" else int(n)) for n in sweep]
    parsed, bad = parse_ranges(ranges_text)
    pools = list(dict.fromkeys(pools + parsed))
    if bad:
        st.warning("Could not read: " + ", ".join(bad))
    c3, c4 = st.columns(2)
    run_active_sweep = c3.button("Sweep current strategy", width="stretch", disabled=not pools,
                                 help="Re-run the strategy in the sidebar on each pool")
    run_passive_sweep = c4.button("Sweep passive (cap-weighted)", width="stretch", disabled=not pools,
                                  help="Hold every company in the pool, weighted by market cap, refreshed "
                                       "quarterly, no stock picking")
    if run_active_sweep or run_passive_sweep:
        bt = Backtester(md)
        bar = st.progress(0.0, text="Running sweep ...")
        for k, (lo, hi) in enumerate(pools):
            whole = lo == 1 and hi >= state["size"]
            if run_passive_sweep:
                r = bt.run_passive(cfg, hi, rank_from=lo)
                add_result(r.label, r)
            else:
                strategy = make_strategy(state["strategy_name"], **state["strategy_params"])
                pool_n = ALL if hi >= ALL else hi - lo + 1
                holdings = min(cfg.n_holdings, pool_n)
                r = bt.run(replace(cfg, top_n=ALL if whole else hi, n_holdings=holdings, rank_from=lo), strategy)
                hold_label = "All" if holdings >= state["size"] else holdings
                add_result(f"{strategy.label()} | {state['universe']} {pool_label(lo, hi, whole)} → "
                           f"{hold_label} holdings, {strategy.rebalance_override or cfg.rebalance}", r)
            bar.progress((k + 1) / len(pools), text=f"{pool_label(lo, hi, whole)} done")
        bar.empty()

    st.markdown("**Market benchmarks** (plain buy-and-hold of an index ETF, dividends reinvested, "
                f"in {md.currency_label})")
    c1, c2, c3 = st.columns([2, 1.2, 1])
    picked = c1.multiselect("Markets", list(MARKET_BENCHMARKS), default=[], key="bench_pick")
    extra = c2.text_input("Other tickers", "", key="bench_extra", placeholder="e.g. VOO, BRK-B",
                          help="Any Yahoo ticker quoted in US dollars")
    if c3.button("Add benchmarks", width="stretch", disabled=not (picked or extra.strip())):
        wanted = [(label, MARKET_BENCHMARKS[label]) for label in picked]
        wanted += [(t, t) for t in parse_custom(extra)]
        skipped = []
        with st.spinner("Loading benchmark prices ..."):
            for label, ticker in wanted:
                series, reason = benchmark_values(provider(), md, ticker, active.values.index, cfg.capital)
                if series is None:
                    skipped.append(reason)
                else:
                    add_to_compare(label, series, None)
        if skipped:
            st.warning("Not added: " + "; ".join(skipped))

    if st.button("Clear comparison"):
        compare.clear()

    if compare:
        shown = []   # (label, values, metrics) - deflated on the fly when the inflation switch is on
        for c in compare:
            v, m = c["values"], c["metrics"]
            if inflation_on:
                v = deflate_series(v, deflator(md.cpi, v.index))
                m = series_metrics(v) | {"Trades": c["metrics"]["Trades"]}
            shown.append((c["label"], v, m))
        rows = [{"Configuration": lab, "CAGR": pct(m["CAGR"]), "Final value": money(m["Final value"], symbol),
                 "Total return": pct(m["Total return"]), "Max drawdown": pct(m["Max drawdown"]),
                 "Volatility": pct(m["Annual volatility"], signed=False), "Sharpe": f"{m['Sharpe ratio']:.2f}",
                 "Trades": f"{m['Trades']:,}" if m["Trades"] is not None else "-"}
                for lab, _, m in shown]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if inflation_on:
            st.caption(f"Inflation-adjusted: values in {active.values.index[0].year} money.")
        palette = charts.SERIES
        series = {lab: (v, palette[k % len(palette)]) for k, (lab, v, _) in enumerate(shown)}
        dashes = {c["label"]: "dash" for k, c in enumerate(compare) if k >= len(palette)}
        st.plotly_chart(charts.growth_chart(series, symbol, log=log_scale, height=480, dashes=dashes),
                        width="stretch")
        st.caption("Every line starts from the same capital on the backtest start date. Passive and universe "
                   "portfolios pay the same transaction costs as the active strategy; ETF benchmarks pay none.")
    else:
        st.info("Nothing to compare yet. Add the current run, run a sweep, or add market benchmarks.")
