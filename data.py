"""
data.py - the market-data layer.

Everything the backtester knows about the outside world comes through the
DataProvider interface at the top of this file.  FreeDataProvider is the default
implementation and uses only free sources:

  prices, dividends, splits   Yahoo Finance via yfinance ("Adj Close" = total return)
  S&P 500 membership          point-in-time constituent history (github.com/fja05680/sp500)
  GICS sectors                github.com/datasets/s-and-p-500-companies (current members)
  NASDAQ-100 membership       Nasdaq's public API (current members only)
  shares outstanding          SEC XBRL "frames" API, quarterly from 2009, with a
                              Yahoo fallback (today's share count held constant)

Every download is cached in ./data_cache so later runs are instant.  To use a
different source, subclass DataProvider and pass it to build_market_data().
"""
from __future__ import annotations

import io
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import requests
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# SEC asks for a descriptive User-Agent with a contact address. Override with the
# SEC_USER_AGENT environment variable if you like.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "StockBacktester/1.0 (personal research; contact@example.com)")
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SP500_HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                     "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SP500_SECTORS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NASDAQ100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/shares/{period}.json"

PRICE_HISTORY_START = "1990-01-01"
SEC_FIRST_YEAR = 2009             # XBRL filings start here
SHARES_PUBLICATION_LAG_DAYS = 60  # a share count dated D only becomes "known" on D + 60 days
WARMUP_DAYS = 400                 # history loaded before the start date for momentum / moving averages

# Old ticker -> ticker Yahoo uses today for the SAME company (history lives under the new symbol).
TICKER_ALIASES = {
    "FB": "META", "ANTM": "ELV", "WLP": "ELV", "FI": "FISV", "KFT": "MDLZ", "PCLN": "BKNG",
    "WAG": "WBA", "HRS": "LHX", "UTX": "RTX", "BBT": "TFC", "TMK": "GL", "DPS": "KDP",
    "CTL": "LUMN", "HCN": "WELL", "HCP": "DOC", "PEAK": "DOC", "SQ": "XYZ", "BLL": "BALL",
    "FLT": "CPAY", "PKI": "RVTY", "ABC": "COR", "RE": "EG", "WLTW": "WTW", "LB": "BBWI",
    "SYMC": "GEN", "NLOK": "GEN", "BHGE": "BKR", "DWDP": "DD", "ADS": "BFH", "DISCA": "WBD",
    "TSO": "ANDV", "CBG": "CBRE",
}
# Second share classes of companies already represented by their primary class.
SECONDARY_SHARE_CLASSES = {"GOOG", "FOX", "NWS", "BRK.A", "CMCSK", "DISCK", "UA", "LEN.B", "GOOGM", "GOOGN"}

GICS_SECTORS = ["Information Technology", "Health Care", "Financials", "Consumer Discretionary",
                "Communication Services", "Industrials", "Consumer Staples", "Energy",
                "Utilities", "Real Estate", "Materials"]
UNIVERSES = ["S&P 500", "NASDAQ 100"] + [f"S&P 500: {s}" for s in GICS_SECTORS] + ["Custom"]

BENCHMARK_TICKER = "SPY"
FX_TICKER = "GBPUSD=X"

# Plain "buy the market" alternatives (USD-quoted ETFs with long histories) for the Compare tab.
MARKET_BENCHMARKS = {
    "S&P 500 (SPY)": "SPY",
    "NASDAQ 100 (QQQ)": "QQQ",
    "Dow Jones 30 (DIA)": "DIA",
    "Russell 2000 small caps (IWM)": "IWM",
    "World ex-US developed (EFA)": "EFA",
    "Emerging markets (EEM)": "EEM",
    "UK stocks (EWU)": "EWU",
    "World, all countries (VT)": "VT",
    "US bonds (AGG)": "AGG",
    "Gold (GLD)": "GLD",
}


def yahoo_symbol(ticker: str) -> str:
    """Yahoo writes share classes with a dash: BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def canonical(ticker: str) -> str:
    return TICKER_ALIASES.get(ticker, ticker)


# --------------------------------------------------------------------------------------
# Small cached-download helper
# --------------------------------------------------------------------------------------
def _fetch(url: str, cache_name: str, ttl_days: float, headers: dict | None = None, timeout: int = 60) -> bytes:
    path = CACHE_DIR / cache_name
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_days * 86400:
        return path.read_bytes()
    try:
        r = requests.get(url, headers=headers or {"User-Agent": BROWSER_UA}, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return r.content
    except Exception as exc:  # offline: fall back to a stale cache if one exists
        if path.exists():
            return path.read_bytes()
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


# --------------------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------------------
@dataclass
class PriceData:
    adj_close: pd.DataFrame     # total-return price (splits + dividends), USD
    close: pd.DataFrame         # split-adjusted close, USD (used for "real" trade prices)
    splits: pd.DataFrame        # split ratios on split dates (0 elsewhere)
    failed: list[str]           # tickers Yahoo has no data for (delisted / acquired)
    as_of: str                  # date the cache was downloaded


@dataclass
class ConstituentHistory:
    """Point-in-time membership: rows of (effective date, frozenset of tickers)."""
    rows: list[tuple[pd.Timestamp, frozenset]]
    point_in_time: bool         # False when only today's membership is known
    warnings: list[str] = field(default_factory=list)

    def members_at(self, when: pd.Timestamp) -> frozenset:
        current: frozenset = frozenset()
        for d, members in self.rows:
            if d <= when:
                current = members
            else:
                break
        return current

    def all_tickers(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        out: set[str] = set(self.members_at(start))
        for d, members in self.rows:
            if start <= d <= end:
                out |= members
        return sorted(out)


class DataProvider(ABC):
    """Implement these five methods to plug in any other data source."""

    @abstractmethod
    def get_constituents(self, universe: str, custom_tickers: list[str] | None = None) -> ConstituentHistory: ...

    @abstractmethod
    def get_prices(self, tickers: list[str], progress=None) -> PriceData: ...

    @abstractmethod
    def get_market_caps(self, close_usd: pd.DataFrame, splits: pd.DataFrame, progress=None
                        ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return (market caps in USD aligned to close_usd, per-ticker source label, warnings)."""

    @abstractmethod
    def get_fx(self, pair: str = FX_TICKER) -> pd.Series: ...

    @abstractmethod
    def get_benchmark(self) -> tuple[pd.Series, str]: ...

    def universe_size(self, universe: str, custom_tickers: list[str] | None = None) -> int:
        hist = self.get_constituents(universe, custom_tickers)
        return len(hist.members_at(pd.Timestamp.today()))


# --------------------------------------------------------------------------------------
# Free provider: Yahoo Finance + GitHub constituent history + SEC share counts
# --------------------------------------------------------------------------------------
class FreeDataProvider(DataProvider):
    name = "Yahoo Finance + SEC + GitHub constituent history"

    # ---- constituents ----------------------------------------------------------------
    def _sp500_history(self) -> ConstituentHistory:
        raw = _fetch(SP500_HISTORY_URL, "sp500_history.csv", ttl_days=7)
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"]).sort_values("date")
        rows = []
        for d, tickers in zip(df["date"], df["tickers"]):
            members = {canonical(t.strip()) for t in str(tickers).split(",") if t.strip()}
            rows.append((pd.Timestamp(d), frozenset(members - SECONDARY_SHARE_CLASSES)))
        return ConstituentHistory(rows=rows, point_in_time=True)

    def _sp500_sectors(self) -> pd.Series:
        raw = _fetch(SP500_SECTORS_URL, "sp500_sectors.csv", ttl_days=7)
        df = pd.read_csv(io.BytesIO(raw))
        return pd.Series(df["GICS Sector"].values, index=[canonical(s) for s in df["Symbol"]])

    def _nasdaq100_current(self) -> list[str]:
        headers = {"User-Agent": BROWSER_UA, "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}
        raw = _fetch(NASDAQ100_URL, "nasdaq100.json", ttl_days=7, headers=headers, timeout=30)
        rows = json.loads(raw)["data"]["data"]["rows"]
        tickers = {canonical(r["symbol"].replace("/", ".")) for r in rows}
        return sorted(tickers - SECONDARY_SHARE_CLASSES)

    def get_constituents(self, universe: str, custom_tickers: list[str] | None = None) -> ConstituentHistory:
        if universe == "S&P 500":
            return self._sp500_history()
        if universe.startswith("S&P 500: "):
            sector = universe.split(": ", 1)[1]
            sectors = self._sp500_sectors()
            in_sector = frozenset(sectors.index[sectors == sector])
            hist = self._sp500_history()
            rows = [(d, members & in_sector) for d, members in hist.rows]
            warn = (f"Sector classification is only available for today's S&P 500 members, so the "
                    f"'{sector}' universe contains former index members only if they are still in the "
                    f"index today. Results may contain survivorship bias.")
            return ConstituentHistory(rows=rows, point_in_time=True, warnings=[warn])
        if universe == "NASDAQ 100":
            tickers = self._nasdaq100_current()
            warn = ("Historical constituent data unavailable for the NASDAQ 100: today's members are "
                    "used for the whole period. Results may contain survivorship bias.")
            return ConstituentHistory(rows=[(pd.Timestamp("1990-01-01"), frozenset(tickers))],
                                      point_in_time=False, warnings=[warn])
        if universe == "Custom":
            tickers = sorted({canonical(t.strip().upper()) for t in (custom_tickers or []) if t.strip()})
            if not tickers:
                raise ValueError("Enter at least one ticker for the custom universe.")
            warn = "Custom universe: membership is fixed for the whole period (no historical constituents)."
            return ConstituentHistory(rows=[(pd.Timestamp("1990-01-01"), frozenset(tickers))],
                                      point_in_time=False, warnings=[warn])
        raise ValueError(f"Unknown universe {universe!r}")

    # ---- prices ---------------------------------------------------------------------
    _price_files = {"adj_close": "prices_adj_close.parquet", "close": "prices_close.parquet",
                    "splits": "prices_splits.parquet"}

    def _load_price_cache(self) -> tuple[dict[str, pd.DataFrame], dict]:
        frames = {}
        for key, name in self._price_files.items():
            p = CACHE_DIR / name
            frames[key] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        meta_path = CACHE_DIR / "prices_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"failed": [], "as_of": None}
        return frames, meta

    def _save_price_cache(self, frames: dict[str, pd.DataFrame], meta: dict) -> None:
        for key, name in self._price_files.items():
            frames[key].to_parquet(CACHE_DIR / name)
        (CACHE_DIR / "prices_meta.json").write_text(json.dumps(meta))

    @staticmethod
    def _download_chunk(symbols: list[str]) -> dict[str, pd.DataFrame]:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(symbols, start=PRICE_HISTORY_START, auto_adjust=False, actions=True,
                             group_by="column", threads=True, progress=False)
        if df is None or df.empty:
            return {"adj_close": pd.DataFrame(), "close": pd.DataFrame(), "splits": pd.DataFrame()}
        if not isinstance(df.columns, pd.MultiIndex):        # single ticker -> flat columns
            df.columns = pd.MultiIndex.from_product([df.columns, symbols])
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        out = {}
        for key, col in (("adj_close", "Adj Close"), ("close", "Close"), ("splits", "Stock Splits")):
            part = df[col] if col in df.columns.get_level_values(0) else pd.DataFrame(index=df.index)
            out[key] = part.astype("float64")
        return out

    def get_prices(self, tickers: list[str], progress=None) -> PriceData:
        tickers = list(dict.fromkeys(tickers))
        frames, meta = self._load_price_cache()
        cached = set(frames["adj_close"].columns) | set(meta["failed"])
        missing = [t for t in tickers if t not in cached]
        if missing:
            chunks = [missing[i:i + 100] for i in range(0, len(missing), 100)]
            new = {"adj_close": [], "close": [], "splits": []}
            failed = list(meta["failed"])
            for i, chunk in enumerate(chunks):
                if progress:
                    progress(i / len(chunks), f"Downloading prices, tickers {i * 100 + 1}-"
                                              f"{min((i + 1) * 100, len(missing))} of {len(missing)} ...")
                symbols = [yahoo_symbol(t) for t in chunk]
                for attempt in range(3):
                    try:
                        got = self._download_chunk(symbols)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(5)
                back = {yahoo_symbol(t): t for t in chunk}
                have = got["close"].rename(columns=back).reindex(columns=chunk).notna().any()
                chunk_failed = [t for t in chunk if not bool(have.get(t, False))]
                failed += chunk_failed
                for key in new:
                    part = got[key].rename(columns=back).reindex(columns=chunk)
                    new[key].append(part.drop(columns=chunk_failed))
            for key in new:
                add = pd.concat(new[key], axis=1)
                frames[key] = add.sort_index() if frames[key].empty else pd.concat([frames[key], add], axis=1).sort_index()
                frames[key] = frames[key].loc[:, ~frames[key].columns.duplicated()]
            meta["failed"] = sorted(set(failed))
            meta["as_of"] = meta["as_of"] or str(date.today())
            self._save_price_cache(frames, meta)
            if progress:
                progress(1.0, "Price download complete")
        cols = [t for t in tickers if t in frames["adj_close"].columns]
        failed = [t for t in tickers if t not in cols]
        return PriceData(adj_close=frames["adj_close"][cols], close=frames["close"][cols],
                         splits=frames["splits"].reindex(columns=cols).fillna(0.0),
                         failed=failed, as_of=meta["as_of"] or "?")

    def price_cache_info(self) -> dict:
        _, meta = self._load_price_cache()
        p = CACHE_DIR / self._price_files["adj_close"]
        n = len(pd.read_parquet(p).columns) if p.exists() else 0
        return {"as_of": meta.get("as_of"), "tickers": n, "failed": len(meta.get("failed", []))}

    def clear_price_cache(self) -> None:
        for name in list(self._price_files.values()) + ["prices_meta.json", "yahoo_shares.json"]:
            p = CACHE_DIR / name
            if p.exists():
                p.unlink()

    # ---- shares outstanding / market caps ------------------------------------------------
    _SEC_TAGS = [  # (taxonomy, tag, instant?) in order of preference
        ("dei", "EntityCommonStockSharesOutstanding", True),
        ("us-gaap", "CommonStockSharesOutstanding", True),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", False),
    ]

    @staticmethod
    def _quarters_through_today() -> list[tuple[int, int]]:
        today = date.today()
        last_q = (today.month - 1) // 3          # last COMPLETE calendar quarter
        y, q = (today.year, last_q) if last_q > 0 else (today.year - 1, 4)
        return [(year, qq) for year in range(SEC_FIRST_YEAR, y + 1) for qq in range(1, 5) if (year, qq) <= (y, q)]

    def _sec_share_frames(self, progress=None) -> pd.DataFrame:
        """Quarterly share counts for every SEC filer: columns cik, end, val, priority."""
        path = CACHE_DIR / "sec_shares.parquet"
        meta_path = CACHE_DIR / "sec_shares_meta.json"
        done: set[str] = set()
        table = pd.DataFrame({"cik": pd.Series(dtype="int64"), "end": pd.Series(dtype="datetime64[ns]"),
                              "val": pd.Series(dtype="float64"), "priority": pd.Series(dtype="int64")})
        if path.exists() and meta_path.exists():
            table = pd.read_parquet(path)
            done = set(json.loads(meta_path.read_text())["periods"])
        todo = []
        for (y, q) in self._quarters_through_today():
            for prio, (tax, tag, instant) in enumerate(self._SEC_TAGS):
                period = f"CY{y}Q{q}{'I' if instant else ''}"
                key = f"{tag}:{period}"
                if key not in done:
                    todo.append((key, tax, tag, period, prio))
        if todo:
            parts = [table] if not table.empty else []
            headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
            for i, (key, tax, tag, period, prio) in enumerate(todo):
                if progress and i % 5 == 0:
                    progress(i / len(todo), f"Downloading SEC share counts ({i + 1}/{len(todo)}) ...")
                url = SEC_FRAMES_URL.format(taxonomy=tax, tag=tag, period=period)
                r = requests.get(url, headers=headers, timeout=60)
                if r.status_code == 200:
                    rows = r.json().get("data", [])
                    if rows:
                        df = pd.DataFrame(rows)[["cik", "end", "val"]]
                        df["priority"] = prio
                        parts.append(df)
                elif r.status_code != 404:
                    r.raise_for_status()
                done.add(key)
                time.sleep(0.12)  # SEC fair-use limit is 10 requests/second
            if parts:
                table = pd.concat(parts, ignore_index=True)
            table["cik"] = table["cik"].astype("int64")
            table["end"] = pd.to_datetime(table["end"])
            table["val"] = table["val"].astype("float64")
            table["priority"] = table["priority"].astype("int64")
            table.to_parquet(path)
            meta_path.write_text(json.dumps({"periods": sorted(done)}))
        table["end"] = pd.to_datetime(table["end"])
        return table

    def _sec_ticker_map(self) -> dict[str, int]:
        raw = _fetch(SEC_TICKERS_URL, "sec_tickers.json", ttl_days=30, headers={"User-Agent": SEC_USER_AGENT})
        return {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(raw).values()}

    def _yahoo_current_shares(self, tickers: list[str], progress=None) -> pd.Series:
        """Fallback: today's implied share count (market cap / price) from Yahoo, cached."""
        path = CACHE_DIR / "yahoo_shares.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        todo = [t for t in tickers if t not in cache]
        if todo:
            import logging
            from concurrent.futures import ThreadPoolExecutor
            import yfinance as yf
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)

            def fetch(t: str):
                try:
                    fi = yf.Ticker(yahoo_symbol(t)).fast_info
                    shares = fi.get("shares")
                    if not shares and fi.get("marketCap") and fi.get("lastPrice"):
                        shares = fi["marketCap"] / fi["lastPrice"]
                    return t, (float(shares) if shares else None)
                except Exception:
                    return t, None

            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (t, v) in enumerate(pool.map(fetch, todo)):
                    cache[t] = v
                    if progress and i % 10 == 0:
                        progress(i / len(todo), f"Fetching share counts from Yahoo ({i + 1}/{len(todo)}) ...")
                    if (i + 1) % 50 == 0:
                        path.write_text(json.dumps(cache))
            path.write_text(json.dumps(cache))
        return pd.Series({t: cache.get(t) for t in tickers}, dtype="float64")

    def get_market_caps(self, close_usd: pd.DataFrame, splits: pd.DataFrame, progress=None
                        ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        tickers = list(close_usd.columns)
        dates = close_usd.index
        warns: list[str] = []
        source = pd.Series("none", index=tickers, dtype="object")

        # SEC counts are the shares in issue at the time while Yahoo prices are split-adjusted, so
        # every historical count is restated in today's share terms: multiplied by all splits that
        # happened AFTER the date the count was reported.
        ratios = splits.reindex(columns=tickers).fillna(0.0).replace(0.0, 1.0).sort_index()
        cum = ratios.cumprod()
        total_factor = cum.iloc[-1] if len(cum) else pd.Series(1.0, index=tickers)

        shares = pd.DataFrame(np.nan, index=dates, columns=tickers)
        sec_ok: list[str] = []
        try:
            table = self._sec_share_frames(progress)
            cik_map = self._sec_ticker_map()
            cik_of = {t: cik_map.get(yahoo_symbol(t)) for t in tickers}
            wanted = {c for c in cik_of.values() if c is not None}
            sub = table[table["cik"].isin(wanted) & (table["val"] > 0)].copy()
            sub["quarter"] = sub["end"].dt.to_period("Q")
            # best tag per company-quarter
            sub = sub.sort_values(["cik", "quarter", "priority"]).drop_duplicates(["cik", "quarter"])
            by_cik = {c: g.sort_values("end") for c, g in sub.groupby("cik")}
            for t in tickers:
                g = by_cik.get(cik_of[t])
                if g is None or g.empty:
                    continue
                ends = pd.DatetimeIndex(g["end"].values)
                c = cum[t]
                base = c.reindex(c.index.union(ends)).ffill().reindex(ends).fillna(1.0)
                q = pd.Series(g["val"].to_numpy() * (total_factor[t] / base.to_numpy()), index=ends)
                # reject filing errors (wrong scale, stray zeros): first isolated ones by comparing
                # with neighbours, then runs of them by comparing with the whole series
                med = q.rolling(5, center=True, min_periods=1).median()
                q = q[(q / med).between(0.4, 2.5)]
                if len(q):
                    q = q[(q / q.median()).between(1 / 8, 8)]
                if q.empty:
                    continue
                # a count is only usable once it has been published (filing lag)
                q.index = q.index + pd.Timedelta(days=SHARES_PUBLICATION_LAG_DAYS)
                q = q[~q.index.duplicated(keep="last")]
                daily = q.reindex(dates.union(q.index)).ffill().reindex(dates)
                daily = daily.bfill()  # before the first filing: hold the earliest known count (approximation)
                shares[t] = daily.to_numpy()
                sec_ok.append(t)
            # Cross-check the latest SEC count against Yahoo's current total. Companies with several
            # share classes (BRK.A/BRK.B, GOOGL/GOOG ...) may file one class per line, and a few
            # report in thousands; when the two disagree the SEC series is dropped.
            current = self._yahoo_current_shares(sec_ok, progress)
            for t in list(sec_ok):
                latest, y = shares[t].iloc[-1], current.get(t)
                if y and np.isfinite(y) and y > 0 and np.isfinite(latest) and not (0.5 < latest / y < 2.0):
                    sec_ok.remove(t)
                    shares[t] = np.nan
            source[sec_ok] = "SEC"
        except Exception as exc:
            warns.append(f"SEC share-count download failed ({exc}); using today's share counts for all "
                         f"tickers, so historical market caps ignore buybacks and share issuance.")

        need_fallback = [t for t in tickers if t not in sec_ok]
        if need_fallback:
            current = self._yahoo_current_shares(need_fallback, progress)
            for t in need_fallback:
                v = current.get(t)
                if v is not None and np.isfinite(v) and v > 0:
                    shares[t] = v
                    source[t] = "Yahoo (today's count)"
        n_none = int((source == "none").sum())
        if n_none:
            warns.append(f"{n_none} ticker(s) have no share-count data and are excluded from market-cap ranking: "
                         + ", ".join(source.index[source == "none"][:12]) + (" ..." if n_none > 12 else ""))
        n_fallback = int((source == "Yahoo (today's count)").sum())
        if n_fallback or sec_ok:
            warns.append(f"Historical share counts (SEC filings, 2009 onwards) found for {len(sec_ok)} tickers; "
                         f"{n_fallback} use today's share count held constant, and counts before 2009 are "
                         f"back-filled from the earliest filing. Market-cap ranking is therefore approximate.")
        mcap = close_usd * shares
        return mcap, source, warns

    # ---- FX & benchmark -----------------------------------------------------------------
    def get_fx(self, pair: str = FX_TICKER) -> pd.Series:
        return self.get_prices([pair]).close[pair].dropna()

    def get_benchmark(self) -> tuple[pd.Series, str]:
        s = self.get_prices([BENCHMARK_TICKER]).adj_close[BENCHMARK_TICKER].dropna()
        return s, "S&P 500 ETF (SPY)"


# --------------------------------------------------------------------------------------
# MarketData: everything the engine needs, aligned to one trading calendar
# --------------------------------------------------------------------------------------
@dataclass
class MarketData:
    universe: str
    currency: str                   # "GBP" or "USD"
    dates: pd.DatetimeIndex         # trading days (warm-up start .. end)
    tickers: list[str]
    adj: np.ndarray                 # D x T total-return prices in `currency`, forward-filled after listing
    close: np.ndarray               # D x T split-adjusted close in `currency`, forward-filled
    mcap: np.ndarray                # D x T market cap in USD (NaN when the stock has no price that day)
    member: np.ndarray              # D x T bool: point-in-time universe membership
    benchmark: pd.Series            # index total-return series in `currency`
    benchmark_label: str
    warnings: list[str]
    coverage: pd.DataFrame          # per year: constituents, with price data, coverage %
    shares_source: pd.Series
    prices_as_of: str
    point_in_time: bool
    fx: pd.Series | None = None

    @property
    def symbol(self) -> str:
        return "£" if self.currency == "GBP" else "$"

    def index_of(self, when) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(when)))


def build_market_data(provider: DataProvider, universe: str, start, end, currency: str = "GBP",
                      custom_tickers: list[str] | None = None, progress=None) -> MarketData:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    warm_start = start - pd.Timedelta(days=WARMUP_DAYS)
    warns: list[str] = []

    hist = provider.get_constituents(universe, custom_tickers)
    warns += hist.warnings
    tickers = hist.all_tickers(warm_start, end)
    if not tickers:
        raise ValueError(f"No constituents found for {universe} in this date range.")

    prices = provider.get_prices(tickers + [BENCHMARK_TICKER, FX_TICKER], progress)
    bench, bench_label = provider.get_benchmark()
    calendar = bench.index[(bench.index >= warm_start) & (bench.index <= end)]
    if len(calendar) < 30:
        raise ValueError("Not enough trading days in the selected range.")

    ok = [t for t in tickers if t not in prices.failed]
    failed = [t for t in tickers if t in prices.failed]
    if failed and hist.point_in_time:
        warns.append(f"{len(failed)} of {len(tickers)} companies that were in the universe during this period "
                     f"have no price history on Yahoo (delisted, acquired or renamed). They are excluded, so "
                     f"results contain partial survivorship bias. Examples: {', '.join(failed[:10])}"
                     + (" ..." if len(failed) > 10 else ""))
    elif failed:
        warns.append(f"No price data for: {', '.join(failed[:15])}" + (" ..." if len(failed) > 15 else ""))
    if not ok:
        raise ValueError("None of the universe tickers have price data.")

    adj = prices.adj_close[ok].reindex(calendar)
    close = prices.close[ok].reindex(calendar)

    # market caps use UNFILLED closes so a stock that stops trading drops out of the ranking;
    # the full split history is passed so counts can be restated in today's share terms
    mcap, source, mc_warns = provider.get_market_caps(close, prices.splits[ok], progress)
    warns += mc_warns

    adj_f, close_f = adj.ffill(), close.ffill()
    fx = None
    if currency == "GBP":
        rate = provider.get_fx(FX_TICKER).reindex(calendar).ffill()
        if rate.isna().any():
            first = rate.first_valid_index()
            warns.append(f"GBP/USD exchange rates start {first.date()}; earlier dates use that first rate.")
            rate = rate.bfill()
        adj_f, close_f = adj_f.div(rate, axis=0), close_f.div(rate, axis=0)
        bench = bench.reindex(calendar).ffill().div(rate)
        fx = rate
    else:
        bench = bench.reindex(calendar).ffill()

    # point-in-time membership matrix
    member = np.zeros((len(calendar), len(ok)), dtype=bool)
    col = {t: i for i, t in enumerate(ok)}
    rows = hist.rows
    for k, (d, members) in enumerate(rows):
        nxt = rows[k + 1][0] if k + 1 < len(rows) else calendar[-1] + pd.Timedelta(days=1)
        a, b = calendar.searchsorted(d), calendar.searchsorted(nxt)
        if b <= a:
            continue
        idx = [col[t] for t in members if t in col]
        if idx:
            member[a:b, idx] = True

    # coverage by year: how much of the true universe we can actually see
    cov_rows = []
    for year in range(start.year, end.year + 1):
        d = max(pd.Timestamp(year=year, month=1, day=1), start)
        if d > end:
            break
        members = hist.members_at(d)
        with_data = [t for t in members if t in col]
        cov_rows.append({"Year": year, "Constituents": len(members), "With price data": len(with_data),
                         "Coverage": (len(with_data) / len(members)) if members else np.nan})
    coverage = pd.DataFrame(cov_rows)

    return MarketData(universe=universe, currency=currency, dates=calendar, tickers=ok,
                      adj=adj_f.to_numpy(dtype="float64"), close=close_f.to_numpy(dtype="float64"),
                      mcap=mcap.to_numpy(dtype="float64"), member=member,
                      benchmark=bench, benchmark_label=bench_label, warnings=warns, coverage=coverage,
                      shares_source=source, prices_as_of=prices.as_of, point_in_time=hist.point_in_time, fx=fx)


def benchmark_values(provider: DataProvider, md: MarketData, ticker: str, index: pd.DatetimeIndex,
                     capital: float) -> tuple[pd.Series | None, str | None]:
    """Total-return series of a USD-quoted ETF/stock, converted to the MarketData currency and
    scaled to `capital` on the first day of `index`. Returns (series, None) or (None, reason)."""
    prices = provider.get_prices([ticker])
    if ticker in prices.failed:
        return None, f"{ticker}: no price data on Yahoo"
    s = prices.adj_close[ticker].dropna()
    first = s.index[0]
    if first > index[0]:
        return None, f"{ticker}: history only starts {first.date()}"
    s = s.reindex(md.dates).ffill()
    if md.fx is not None:
        s = s / md.fx
    s = s.reindex(index).ffill()
    return (s / s.iloc[0] * capital), None
