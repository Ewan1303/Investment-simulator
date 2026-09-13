"""
app.py - Stock Active Management Simulator (Streamlit UI).

    streamlit run app.py

Answers one question: if I actively managed the largest companies in an investment
universe, would I have made more money than simply holding the universe?
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import charts
from analytics import (annual_returns, cagr, drawdown_series, max_drawdown, metrics_table, monthly_returns,
                       result_metrics, series_metrics, sharpe, total_return)
from backtester import ALL, Backtester, BacktestConfig, REBALANCE_OPTIONS
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

    n_options: list = [n for n in TOP_N_OPTIONS if n < size] + ["All"]
    top_choice = st.select_slider("Top N companies (by market cap at the time)", options=n_options,
                                  value=10 if 10 in n_options else n_options[-1])
    top_n = size if top_choice == "All" else int(top_choice)
    st.caption(f"Universe holds {size} companies today.")

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

    h_options = sorted({h for h in HOLDINGS_OPTIONS if h < top_n} | {top_n})
    n_holdings = st.select_slider("Number of holdings", options=h_options,
                                  value=5 if 5 in h_options else h_options[-1])
    buy_hold = strategy_name == "Buy & Hold"
    rebalance = st.selectbox("Rebalance", REBALANCE_OPTIONS, index=5 if buy_hold else 2, disabled=buy_hold)

    currency = st.radio("Currency", ["GBP", "USD"], horizontal=True)
    symbol = "£" if currency == "GBP" else "$"
    capital = float(st.number_input(f"Starting capital ({symbol})", min_value=1_000, value=100_000, step=10_000))
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start date", date(2005, 1, 1), min_value=date(1997, 1, 1), max_value=date.today())
    end_date = c2.date_input("End date", date.today(), min_value=date(1998, 1, 1), max_value=date.today())
    c1, c2 = st.columns(2)
    tx_cost = c1.number_input("Transaction cost (%)", 0.0, 5.0, 0.10, 0.05, format="%.2f") / 100
    slippage = c2.number_input("Slippage (%)", 0.0, 5.0, 0.05, 0.05, format="%.2f") / 100

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
        cfg = BacktestConfig(top_n=top_n, n_holdings=n_holdings, rebalance=rebalance,
                             start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), capital=capital,
                             tx_cost=tx_cost, slippage=slippage, seed=seed)
        top_label = "All" if top_n >= size else f"Top {top_n}"
        label = f"{strategy.label()} | {universe} {top_label} → {n_holdings} holdings, {strategy.rebalance_override or rebalance}"
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
reference = passive or universe_res

if md.warnings:
    st.warning("**Data caveats**\n\n" + "\n\n".join(f"- {w}" for w in md.warnings))

# ---- headline statistics --------------------------------------------------------------
m_active = result_metrics(active)
m_ref = series_metrics(reference.values)
ref_name = "passive Top-N" if passive is not None else "universe"
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
head.subheader("Portfolio value")
log_scale = toggle.toggle("Log scale", value=False)
st.plotly_chart(charts.growth_chart(lines, symbol, log=log_scale), width="stretch")

# ---- tabs ----------------------------------------------------------------------------------
tabs = st.tabs(["Performance", "Drawdown", "Annual returns", "Monthly returns", "Portfolio", "Trades",
                "Monte Carlo", "Compare"])

with tabs[0]:
    st.plotly_chart(charts.cumulative_return_chart(lines), width="stretch")
    columns = {active.label: m_active}
    if passive is not None:
        columns[passive.label] = series_metrics(passive.values)
    columns[universe_res.label] = series_metrics(universe_res.values)
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
        rows.append({"Series": label, "Max drawdown": pct(dd.min()), "Peak": peak.date(), "Trough": trough.date(),
                     "Recovered": recovered.index[0].date() if len(recovered) else "not yet"})
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
            st.markdown(f"**Top {len(pool)} pool on {pd.Timestamp(when).date()}** (market cap, US$ bn)")
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

with tabs[5]:
    trades = active.trades
    if trades.empty:
        st.info("No trades were made.")
    else:
        counts = trades["Ticker"].value_counts()
        tk = st.selectbox("Ticker", counts.index.tolist(), index=0, key="trade_ticker")
        j = md.tickers.index(tk)
        lo, hi = md.index_of(cfg.start), md.index_of(cfg.end)
        price = pd.Series(md.close[lo:hi + 1, j], index=md.dates[lo:hi + 1]).dropna()
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
                   f"{money(active.turnover_value, symbol)} traded in total. Prices are split-adjusted closes in {md.currency}.")

with tabs[6]:
    st.markdown("Run many **random** stock selections from the same Top-N pool, with the same number of holdings, "
                "rebalance schedule and costs, to see how the active strategy compares with luck.")
    c1, c2 = st.columns([1, 3])
    n_sims = int(c1.number_input("Number of simulations", 10, 5000, 200, 50))
    if c2.button("Run Monte Carlo", type="primary"):
        bar = st.progress(0.0, text="Starting ...")
        mc = Backtester(md).monte_carlo(cfg, n_sims, progress=lambda f, m: bar.progress(f, text=m))
        bar.empty()
        st.session_state["mc"] = mc
    mc = st.session_state.get("mc")
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

with tabs[7]:
    st.markdown("Collect configurations and compare them side by side: the current run, the current strategy "
                "over different pool sizes, **passive** cap-weighted Top-N portfolios (e.g. Top 1 vs Top 25 with "
                "no active management), and plain market benchmarks.")
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

    st.markdown("**Top-N sweep**")
    sweep_options = [n for n in TOP_N_OPTIONS if n < state["size"]] + ["All"]
    c1, c2, c3 = st.columns([2, 1, 1])
    sweep = c1.multiselect("Pool sizes (N)", sweep_options, default=[], key="sweep",
                           help="Each N is run separately; 'All' means the whole universe")
    run_active_sweep = c2.button("Sweep current strategy", width="stretch", disabled=not sweep,
                                 help="Re-run the strategy in the sidebar with each pool size")
    run_passive_sweep = c3.button("Sweep passive Top-N", width="stretch", disabled=not sweep,
                                  help="Buy the N largest companies cap-weighted, refreshed quarterly, no stock picking")
    if run_active_sweep or run_passive_sweep:
        bt = Backtester(md)
        bar = st.progress(0.0, text="Running sweep ...")
        for k, n in enumerate(sweep):
            top = state["size"] if n == "All" else int(n)
            if run_passive_sweep:
                r = bt.run_passive(cfg, top)
                add_result(r.label, r)
            else:
                strategy = make_strategy(state["strategy_name"], **state["strategy_params"])
                holdings = min(cfg.n_holdings, top)
                r = bt.run(replace(cfg, top_n=top, n_holdings=holdings), strategy)
                add_result(f"{strategy.label()} | {state['universe']} {'All' if n == 'All' else f'Top {top}'} → "
                           f"{holdings} holdings, {strategy.rebalance_override or cfg.rebalance}", r)
            bar.progress((k + 1) / len(sweep), text=f"Top {n} done")
        bar.empty()

    st.markdown("**Market benchmarks** (plain buy-and-hold of an index ETF, dividends reinvested, "
                f"in {md.currency})")
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
        rows = [{"Configuration": c["label"], "CAGR": pct(c["metrics"]["CAGR"]),
                 "Final value": money(c["metrics"]["Final value"], symbol),
                 "Total return": pct(c["metrics"]["Total return"]), "Max drawdown": pct(c["metrics"]["Max drawdown"]),
                 "Volatility": pct(c["metrics"]["Annual volatility"], signed=False),
                 "Sharpe": f"{c['metrics']['Sharpe ratio']:.2f}",
                 "Trades": f"{c['metrics']['Trades']:,}" if c["metrics"]["Trades"] is not None else "-"}
                for c in compare]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        palette = charts.SERIES
        series = {c["label"]: (c["values"], palette[k % len(palette)]) for k, c in enumerate(compare)}
        dashes = {c["label"]: "dash" for k, c in enumerate(compare) if k >= len(palette)}
        st.plotly_chart(charts.growth_chart(series, symbol, log=log_scale, height=480, dashes=dashes),
                        width="stretch")
        st.caption("Every line starts from the same capital on the backtest start date. Passive and universe "
                   "portfolios pay the same transaction costs as the active strategy; ETF benchmarks pay none.")
    else:
        st.info("Nothing to compare yet. Add the current run, run a sweep, or add market benchmarks.")
