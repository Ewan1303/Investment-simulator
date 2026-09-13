# Stock Active Management Simulator

A small Streamlit app that answers one question:

> If I actively managed the largest companies in an investment universe, would I have made
> more money than simply holding the universe?

Pick a universe (S&P 500, NASDAQ 100, an S&P 500 sector, or your own tickers), keep a **range of
companies by market-cap rank at each point in history** (ranks 1-10 is the Top 10; ranks 25-50 is
the 25th to 50th largest), run a strategy over that pool, and compare it with the passive
cap-weighted version of the same pool, the whole universe and the S&P 500 ETF.

## Install and run

```powershell
cd C:\Users\Ewan1\OneDrive\Documents\_Python\SNP_X
```

```powershell
pip install -r requirements.txt
```

```powershell
python -m streamlit run app.py
```

The browser opens at http://localhost:8501. Python 3.10 or newer is needed.

**First run:** press RUN BACKTEST and wait. The app downloads ~1,000 tickers of price history
from Yahoo Finance (3-5 minutes), quarterly share counts from the SEC (2 minutes) and a
constituent history from GitHub. Everything is cached in `data_cache/`, so every later run
starts in seconds. Use *Data > Refresh price data* in the sidebar to re-download.

## How a backtest works

At every rebalance date the engine:

1. looks up the universe's constituents **on that date** (point-in-time, not today's list),
2. keeps the N largest by market cap **on that date**,
3. hands the strategy only prices up to that date and lets it pick K holdings and weights,
4. fills the trades at the **next** trading day's close, charging transaction cost + slippage
   (the chosen weights are carried to that day's prices, so nothing is traded just because
   prices moved overnight),
5. lets the positions drift until the next rebalance. Dividends are reinvested on the day they
   are paid; the **Dividends** tab shows the cash received per year and the yield on the portfolio.

No look-ahead: strategies receive a price frame cut off at the decision date, membership is
point-in-time, and share counts are only used 60 days after the filing date they refer to.

Benchmarks are run through the same engine with the same costs: the passive Top-N and the whole
universe are cap-weighted and rebalanced quarterly; the S&P 500 line is the SPY ETF's total return.

## Currency: why pounds and dollars give different numbers

US stocks are priced in dollars, so a stock's own return is its dollar return. Showing the result in
pounds means starting with pounds, buying dollars, and converting back at the end. If the exchange
rate moved in between, the pound result differs from the dollar one even though the stocks did
exactly the same thing. The sidebar offers three settings:

| Setting | What it shows |
|---|---|
| **GBP (fixed rate)** (default) | Pounds, with one exchange rate for the whole period. Same percentages as USD, just a different unit, like a currency-hedged fund. Use this to see how much the stocks went up. |
| **GBP (actual rate)** | Pounds converted at each day's real GBP/USD rate. The pound's own rise or fall is part of the result. This is what an unhedged UK investor actually experienced. |
| **USD** | The stocks as quoted. Matches published S&P 500 total-return figures. |

Example: from January 2005 to September 2026 the pound fell from $1.88 to $1.35. SPY returned
+842% in dollars, and the same investment returned +1,225% in pounds at actual rates, because the
dollars came back as 39% more pounds than they cost. Note also that £100,000 and $100,000 are not
the same amount of money: in 2005, £100,000 was about $188,000.

The currency setting never changes which strategy wins. Every line is converted the same way on the
same days, so only the absolute level moves. Published index returns are usually price-only and in
dollars; the app always includes reinvested dividends, which is why its figures look higher.

## Comparing against plain market returns

The **Compare** tab collects series into one table and one chart, all from the same starting capital:

- **Add current run** - the active strategy plus its passive Top-N and whole-universe benchmarks.
- **Pool sweep** - pick Top-N sizes and/or type rank ranges such as `1-1, 2-2, 3-10, 25-50, 100-end`,
  then run them all as the sidebar strategy (**Sweep current strategy**) or with no stock picking at
  all (**Sweep passive**: every company in the pool, cap-weighted, refreshed quarterly). Use it for
  questions like "the largest company vs the second largest" or "Top 1 vs Top 25 vs the whole S&P 500".
- **Market benchmarks** - buy-and-hold of index ETFs with dividends reinvested: S&P 500 (SPY),
  NASDAQ 100 (QQQ), Dow (DIA), Russell 2000 (IWM), developed ex-US (EFA), emerging markets (EEM),
  UK (EWU), world (VT), US bonds (AGG), gold (GLD), or any USD-quoted Yahoo ticker you type in.
  ETFs whose history starts after the backtest start date are skipped with a note.

## Inflation adjustment

Tick **Adjust for inflation** in the sidebar to show every value in start-year money: portfolio
values, dividends, costs, trades, benchmarks, comparisons and Monte Carlo outcomes are all divided
by the rise in consumer prices since the start date (US CPI-U from the BLS for dollars, UK CPI from
the ONS for pounds). The CAGR then becomes a real return, i.e. growth in what the money can buy.
The switch applies instantly to the last run; the engine itself always works in nominal money.

## Costs

The **Costs** tab breaks every charge down: commission, slippage and loan interest, per year and
cumulative, for the active strategy and both benchmarks. Costs are also expressed as a share of the
value of the stocks owned each year (the same idea as a fund's expense ratio), and the same
strategy is re-run with zero trading costs to show exactly how much final value and CAGR the costs
gave up. Small "PORTFOLIO" rows in the trade list are cash sweeps: a uniform slice of every
holding bought or sold so skipped dust-sized trades never leave cash idle.

## Leverage

Open *Leverage (optional)* in the sidebar to run the active strategy with borrowed money. Only the
active strategy is leveraged; the benchmarks stay at 1x, so you can see whether borrowing beat
plain holding. Three methods:

| Method | How it works | Risk |
|---|---|---|
| **Reset at rebalance** | Exposure is set to L x equity at each rebalance and drifts in between | Margin calls between rebalances |
| **Daily reset** | Exposure is reset to L x equity every day, like a leveraged ETF (volatility drag and daily re-hedging costs included) | No margin calls, but a one-day fall of more than 1/L wipes the account out |
| **Initial loan only** | Borrow (L-1) x capital once and never top up; leverage falls as the portfolio grows and rises as it falls | Margin calls |

The loan accrues interest daily at the borrowing rate. For the two margin methods, equity divided
by gross exposure is checked every day against the maintenance margin (25% by default). Below it,
positions are sold pro rata until leverage is back to the target (or 10 points above the
maintenance level if the target is more aggressive), and the sale is logged. If equity reaches
zero the account is wiped out and stays at zero. The **Leverage** tab shows interest paid,
leverage over time, every forced sale, and the same strategy run without leverage for comparison.

## Strategies

| Strategy | What it does |
|---|---|
| Buy & Hold | Buy the K largest on day one, cap-weighted, never trade again |
| Market Cap Weighted | K largest, weighted by market cap, refreshed at each rebalance |
| Equal Weight | K largest, equal weights, rebalanced back to equal |
| Momentum | Rank the pool by trailing 1/3/6/12-month return, hold the top K equally |
| Trend Following | Only hold stocks above their 50/100/200-day moving average, ranked by distance above it |
| Random Selection | K random picks from the pool; the Monte Carlo tab runs hundreds of these |

Add a strategy by writing a small class in `strategies.py` and adding it to `STRATEGIES`.

## Data sources and known limitations

| What | Source | Caveat |
|---|---|---|
| Prices, dividends, splits | Yahoo Finance (yfinance) | Delisted companies are missing (see below) |
| S&P 500 membership | github.com/fja05680/sp500 (point-in-time since 1996) | Renamed tickers are mapped with a small alias table |
| Share counts | SEC XBRL frames API, quarterly from 2009 | Before 2009 the earliest count is back-filled; a few companies use today's count |
| GICS sectors | github.com/datasets/s-and-p-500-companies | Current members only, so sector universes miss former members |
| NASDAQ 100 | Nasdaq API | Current members only, so this universe has survivorship bias |

**Survivorship bias.** Yahoo has no history for most companies that were delisted or acquired
(about a quarter of the names that passed through the S&P 500 since 2005). The point-in-time
membership avoids the usual "today's constituents" bias, but the missing names mean coverage
is ~73% of the index in 2005 rising to ~99% today. The app shows this per year under
*Portfolio > Data coverage*, and prints a warning above the results. Because the experiment
focuses on the largest companies, which are rarely delisted, the effect on Top-N pools is small.
Sanity check: the reconstructed cap-weighted S&P 500 tracks SPY with a 0.99 daily correlation
and annual returns within about one percentage point every year since 2005.

To use a different data source, implement the `DataProvider` interface in `data.py`.

## Project layout

```
app.py          Streamlit UI
backtester.py   simulation engine (rebalance calendar, trades, costs, benchmarks, Monte Carlo)
strategies.py   selection and weighting rules
analytics.py    CAGR, drawdown, Sharpe, Sortino, annual / monthly returns
charts.py       Plotly figures
data.py         DataProvider interface + free Yahoo/SEC/GitHub implementation
data_cache/     downloaded data (created automatically)
```
