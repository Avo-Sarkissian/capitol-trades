"""
data/fetch.py
Handles all external data fetching: Quiver Quant congressional trades + yfinance prices.
Results are cached as CSVs in data/cache/ to avoid repeated API calls.
"""

import os
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

# Suppress yfinance noise
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()  # loads QUIVER_API_TOKEN from .env file

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_TTL_HOURS = 24  # re-fetch if cache is older than this
QUIVER_ENDPOINT = "https://api.quiverquant.com/beta/live/congresstrading"
QUIVER_TOKEN = os.getenv("QUIVER_API_TOKEN", "")

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_is_fresh(path: Path) -> bool:
    """Return True if the cache file exists and is less than CACHE_TTL_HOURS old."""
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV cache."""
    df.to_csv(path, index=False)
    logger.info(f"Cached {len(df)} rows → {path.name}")


# ── Mock data generator ────────────────────────────────────────────────────────

def _generate_mock_trades() -> pd.DataFrame:
    """
    Generate ~500 realistic mock congressional trade records.
    Used as fallback when the Quiver Quant API is unavailable.
    Schema mirrors the real API response.
    """
    logger.warning("Using MOCK trade data — set QUIVER_API_TOKEN in .env for real data.")

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    politicians = [
        ("Nancy Pelosi", "Democrat", "House"),
        ("Paul Pelosi", "Democrat", "House"),      # spouse trades
        ("Dan Crenshaw", "Republican", "House"),
        ("Marjorie Taylor Greene", "Republican", "House"),
        ("Tommy Tuberville", "Republican", "Senate"),
        ("Mark Kelly", "Democrat", "Senate"),
        ("Josh Gottheimer", "Democrat", "House"),
        ("Virginia Foxx", "Republican", "House"),
        ("Michael McCaul", "Republican", "House"),
        ("Ro Khanna", "Democrat", "House"),
        ("Donald Beyer", "Democrat", "House"),
        ("Shelley Moore Capito", "Republican", "Senate"),
        ("Debbie Stabenow", "Democrat", "Senate"),
        ("Pat Toomey", "Republican", "Senate"),
        ("Richard Burr", "Republican", "Senate"),
        ("Dianne Feinstein", "Democrat", "Senate"),
        ("David Perdue", "Republican", "Senate"),
        ("Kelly Loeffler", "Republican", "Senate"),
        ("Chris Collins", "Republican", "House"),
        ("Tom Reed", "Republican", "House"),
        ("Suzan DelBene", "Democrat", "House"),
        ("Pete Sessions", "Republican", "House"),
        ("Alan Grayson", "Democrat", "House"),
        ("Spencer Bachus", "Republican", "House"),
        ("Brian Higgins", "Democrat", "House"),
        ("Tim Walberg", "Republican", "House"),
        ("John Boehner", "Republican", "House"),
        ("Jim Renacci", "Republican", "House"),
        ("Billy Long", "Republican", "House"),
        ("Diane Black", "Republican", "House"),
        ("Ed Royce", "Republican", "House"),
        ("Steve Womack", "Republican", "House"),
        ("John Hoeven", "Republican", "Senate"),
        ("Ron Wyden", "Democrat", "Senate"),
        ("Jeff Merkley", "Democrat", "Senate"),
        ("Maria Cantwell", "Democrat", "Senate"),
        ("Amy Klobuchar", "Democrat", "Senate"),
        ("Elizabeth Warren", "Democrat", "Senate"),
        ("Bernie Sanders", "Democrat", "Senate"),
        ("Mitt Romney", "Republican", "Senate"),
        ("Lisa Murkowski", "Republican", "Senate"),
        ("Susan Collins", "Republican", "Senate"),
        ("Marco Rubio", "Republican", "Senate"),
        ("Ted Cruz", "Republican", "Senate"),
        ("Rand Paul", "Republican", "Senate"),
        ("John Cornyn", "Republican", "Senate"),
        ("Thom Tillis", "Republican", "Senate"),
        ("Richard Shelby", "Republican", "Senate"),
        ("Patty Murray", "Democrat", "Senate"),
        ("Chuck Schumer", "Democrat", "Senate"),
    ]

    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC",
        "WFC", "GS", "MS", "JNJ", "PFE", "MRNA", "BMY", "ABBV", "LLY", "UNH",
        "CVS", "XOM", "CVX", "COP", "SLB", "HAL", "LMT", "RTX", "BA", "NOC",
        "GD", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC",
        "CRM", "NOW", "ADBE", "ORCL", "SAP", "SNOW", "PLTR", "COIN", "HOOD",
        "V", "MA", "PYPL", "SQ", "AFRM", "DIS", "NFLX", "PARA", "WBD",
        "T", "VZ", "TMUS", "AMT", "CCI", "PG", "KO", "PEP", "MCD", "COST",
        "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "CMG", "YUM", "MKC",
        "CAT", "DE", "HON", "MMM", "EMR", "ETN", "PH", "GE", "ITW",
        "UPS", "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU", "CCL", "RCL", "MAR",
        "SPY", "QQQ", "IWM",
    ]

    amount_ranges = [
        "$1,001 - $15,000",
        "$15,001 - $50,000",
        "$50,001 - $100,000",
        "$100,001 - $250,000",
        "$250,001 - $500,000",
        "$500,001 - $1,000,000",
        "Over $1,000,000",
    ]

    transaction_types = ["Purchase", "Sale (Full)", "Sale (Partial)"]

    # Generate 500 trades over the past 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    n = 500

    trade_dates = [
        start_date + timedelta(days=int(rng.integers(0, 730)))
        for _ in range(n)
    ]

    politician_choices = [politicians[i] for i in rng.integers(0, len(politicians), n)]

    rows = []
    for i in range(n):
        name, party, chamber = politician_choices[i]
        ticker = tickers[int(rng.integers(0, len(tickers)))]
        tx_type = transaction_types[int(rng.integers(0, len(transaction_types)))]
        amount = amount_ranges[int(rng.integers(0, len(amount_ranges)))]
        trade_date = trade_dates[i].strftime("%Y-%m-%d")
        rows.append({
            "Representative": name,
            "TransactionDate": trade_date,
            "Ticker": ticker,
            "Transaction": tx_type,
            "Amount": amount,
            "Party": party,
            "Chamber": chamber,
        })

    df = pd.DataFrame(rows)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = df.sort_values("TransactionDate").reset_index(drop=True)
    return df


# ── Main fetch functions ───────────────────────────────────────────────────────

def fetch_congress_trades() -> pd.DataFrame:
    """
    Fetch congressional stock trading disclosures from Quiver Quantitative API.
    Falls back to mock data if the API token is missing or the request fails.
    Results are cached for CACHE_TTL_HOURS.

    Returns:
        DataFrame with columns: Representative, TransactionDate, Ticker,
        Transaction, Amount, Party, Chamber
    """
    cache_path = CACHE_DIR / "congress_trades.csv"

    # Use cache if fresh
    if _cache_is_fresh(cache_path):
        logger.info("Loading congress trades from cache.")
        df = pd.read_csv(cache_path, parse_dates=["TransactionDate"])
        return df

    # Try the real API
    if QUIVER_TOKEN:
        try:
            headers = {"Authorization": f"Token {QUIVER_TOKEN}"}
            resp = requests.get(QUIVER_ENDPOINT, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)

            # Normalize column names to our expected schema
            col_map = {
                "Name": "Representative",
                "Date": "TransactionDate",
                "Ticker": "Ticker",
                "Transaction": "Transaction",
                "Range": "Amount",
                "Party": "Party",
                "Chamber": "Chamber",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
            df = df.dropna(subset=["TransactionDate", "Ticker"])

            _save_cache(df, cache_path)
            logger.info(f"Fetched {len(df)} real trades from Quiver Quant.")
            return df

        except Exception as e:
            logger.warning(f"Quiver Quant API failed: {e}. Falling back to mock data.")

    # Fall back to mock data
    df = _generate_mock_trades()
    _save_cache(df, cache_path)
    return df


def fetch_stock_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download historical daily closing prices for a list of tickers using yfinance.
    Always includes SPY for benchmark comparison.
    Results are cached per ticker.

    Args:
        tickers:    List of ticker symbols (e.g. ["AAPL", "MSFT"])
        start_date: Start date string "YYYY-MM-DD"
        end_date:   End date string "YYYY-MM-DD"

    Returns:
        DataFrame with date index and one column per ticker (Adj Close prices).
        Missing/delisted tickers are silently skipped.
    """
    # Always include SPY as the benchmark
    tickers = list(set(tickers + ["SPY"]))

    cache_path = CACHE_DIR / "stock_prices.csv"

    # Load existing cache if fresh
    cached_df = None
    if _cache_is_fresh(cache_path):
        cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Drop any rows where the index failed to parse as a date (shows up as NaN float)
        cached_df.index = pd.to_datetime(cached_df.index, errors="coerce")
        cached_df = cached_df[cached_df.index.notna()]
        # Check if all requested tickers are already in the cache
        missing = [t for t in tickers if t not in cached_df.columns]
        if not missing:
            logger.info("Loading stock prices from cache.")
            return cached_df

        # Only fetch the missing tickers
        tickers = missing
        logger.info(f"Fetching {len(tickers)} new tickers not in cache: {tickers}")

    frames = {}
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning(f"yfinance returned no data for {ticker} — skipping.")
                continue
            # Keep only the Close column, rename to ticker symbol
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            frames[ticker] = close
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e} — skipping.")

    if not frames:
        logger.error("No price data could be fetched.")
        return cached_df if cached_df is not None else pd.DataFrame()

    new_df = pd.DataFrame(frames)
    # Ensure new_df index is a clean, unique DatetimeIndex
    new_df.index = pd.to_datetime(new_df.index, errors="coerce")
    new_df = new_df[new_df.index.notna()]
    new_df = new_df[~new_df.index.duplicated(keep="last")]

    # Merge with existing cache if we had one
    if cached_df is not None:
        # Deduplicate cache index too, then combine
        cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
        result = cached_df.combine_first(new_df)
    else:
        result = new_df

    _save_cache(result, cache_path)
    logger.info(f"Stock prices loaded for {list(result.columns)} from {start_date} to {end_date}.")
    return result
