# Stock Active Management Simulator

A small Streamlit app that answers one question:

> If I actively managed the largest companies in an investment universe, would I have made
> more money than simply holding the universe?

Pick a universe (S&P 500, NASDAQ 100, an S&P 500 sector, or your own tickers), keep the
**N largest companies by market cap at each point in history**, run a strategy over that pool,
and compare it with the passive cap-weighted Top-N, the whole universe and the S&P 500 ETF.

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
4. fills the trades at the **next** trading day's close, charging transaction cost + slippage,
5. lets the positions drift (dividends reinvested) until the next rebalance.

No look-ahead: strategies receive a price frame cut off at the decision date, membership is
point-in-time, and share counts are only used 60 days after the filing date they refer to.

Benchmarks are run through the same engine with the same costs: the passive Top-N and the whole
universe are cap-weighted and rebalanced quarterly; the S&P 500 line is the SPY ETF's total return.
Prices are converted to GBP with the daily GBP/USD rate (switch to USD in the sidebar).

## Comparing against plain market returns

The **Compare** tab collects series into one table and one chart, all from the same starting capital:

- **Add current run** - the active strategy plus its passive Top-N and whole-universe benchmarks.
- **Sweep current strategy** - the sidebar strategy re-run for several pool sizes (Top 10, Top 50, All ...).
- **Sweep passive Top-N** - no stock picking at all: buy the N largest companies cap-weighted,
  refreshed quarterly. Use it for questions like "Top 1 vs Top 25 vs the whole S&P 500".
- **Market benchmarks** - buy-and-hold of index ETFs with dividends reinvested: S&P 500 (SPY),
  NASDAQ 100 (QQQ), Dow (DIA), Russell 2000 (IWM), developed ex-US (EFA), emerging markets (EEM),
  UK (EWU), world (VT), US bonds (AGG), gold (GLD), or any USD-quoted Yahoo ticker you type in.
  ETFs whose history starts after the backtest start date are skipped with a note.

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
