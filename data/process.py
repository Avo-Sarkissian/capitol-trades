"""
data/process.py
Data cleaning, joining, and derived metric calculations.
Depends on data from fetch.py — call fetch functions first, then pass results here.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Amount parsing ─────────────────────────────────────────────────────────────

# Maps known string amount descriptors to midpoint dollar values
_AMOUNT_OVERRIDES: dict[str, float] = {
    "over $1,000,000":  2_500_000.0,
    "over $1000000":    2_500_000.0,
    "$1,000,001+":      2_500_000.0,
}


def parse_trade_amount(amount_str: str) -> float:
    """
    Convert a disclosure amount range string to a numeric midpoint.

    Examples:
        "$1,001 - $15,000"    → 8000.5
        "$50,001 - $100,000"  → 75000.5
        "Over $1,000,000"     → 2500000.0

    Args:
        amount_str: Raw amount string from the API or CSV.

    Returns:
        Float midpoint of the range, or 0.0 if parsing fails.
    """
    if not isinstance(amount_str, str):
        return 0.0

    cleaned = amount_str.strip().lower()

    # Check known special cases first
    for key, val in _AMOUNT_OVERRIDES.items():
        if key in cleaned:
            return val

    # Extract all numbers from the string (handles commas and decimals)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", amount_str)
    nums = [float(n.replace(",", "")) for n in numbers]

    if len(nums) == 2:
        return (nums[0] + nums[1]) / 2
    elif len(nums) == 1:
        return nums[0]
    else:
        logger.warning(f"Could not parse amount: '{amount_str}'")
        return 0.0


# ── Sector mapping ─────────────────────────────────────────────────────────────

def get_sector_mapping(tickers: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame mapping ticker → GICS sector using yfinance .info.
    Results are cached so we don't hammer yfinance on every run.

    Args:
        tickers: List of stock ticker symbols.

    Returns:
        DataFrame with columns: Ticker, Sector
    """
    cache_path = CACHE_DIR / "sector_mapping.csv"

    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        cached_tickers = set(existing["Ticker"].tolist())
        missing = [t for t in tickers if t not in cached_tickers]
        if not missing:
            return existing
        tickers_to_fetch = missing
    else:
        existing = pd.DataFrame()
        tickers_to_fetch = tickers

    rows = []
    for ticker in tickers_to_fetch:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"
        rows.append({"Ticker": ticker, "Sector": sector})
        logger.info(f"  Sector for {ticker}: {sector}")

    new_df = pd.DataFrame(rows)

    if not existing.empty:
        result = pd.concat([existing, new_df], ignore_index=True)
    else:
        result = new_df

    result.to_csv(cache_path, index=False)
    return result


def get_committee_sector_mapping() -> pd.DataFrame:
    """
    Return a hardcoded mapping of congressional committees to GICS sectors.
    This represents which committees have oversight authority over each sector.

    Returns:
        DataFrame with columns: Committee, Sector
    """
    # Each row is one (committee, sector) pair.
    # Committees can map to multiple sectors.
    rows = [
        # Senate committees
        ("Senate Armed Services",           "Industrials"),
        ("Senate Armed Services",           "Information Technology"),
        ("Senate Banking",                  "Financials"),
        ("Senate Banking",                  "Real Estate"),
        ("Senate Commerce",                 "Communication Services"),
        ("Senate Commerce",                 "Consumer Discretionary"),
        ("Senate Energy and Natural Resources", "Energy"),
        ("Senate Energy and Natural Resources", "Utilities"),
        ("Senate Environment and Public Works", "Utilities"),
        ("Senate Environment and Public Works", "Materials"),
        ("Senate Finance",                  "Financials"),
        ("Senate Finance",                  "Health Care"),
        ("Senate Health Education Labor",   "Health Care"),
        ("Senate Intelligence",             "Information Technology"),
        ("Senate Intelligence",             "Communication Services"),
        ("Senate Judiciary",                "Communication Services"),
        ("Senate Judiciary",                "Information Technology"),
        ("Senate Agriculture",              "Consumer Staples"),
        ("Senate Agriculture",              "Materials"),
        # House committees
        ("House Armed Services",            "Industrials"),
        ("House Armed Services",            "Information Technology"),
        ("House Financial Services",        "Financials"),
        ("House Financial Services",        "Real Estate"),
        ("House Energy and Commerce",       "Energy"),
        ("House Energy and Commerce",       "Health Care"),
        ("House Energy and Commerce",       "Communication Services"),
        ("House Intelligence",              "Information Technology"),
        ("House Intelligence",              "Communication Services"),
        ("House Judiciary",                 "Communication Services"),
        ("House Science Space Technology",  "Information Technology"),
        ("House Science Space Technology",  "Industrials"),
        ("House Ways and Means",            "Financials"),
        ("House Ways and Means",            "Health Care"),
        ("House Agriculture",               "Consumer Staples"),
        ("House Agriculture",               "Materials"),
        ("House Transportation",            "Industrials"),
        ("House Transportation",            "Energy"),
        ("House Oversight",                 "Industrials"),
        ("House Oversight",                 "Information Technology"),
    ]
    return pd.DataFrame(rows, columns=["Committee", "Sector"])


# ── Alpha calculation ──────────────────────────────────────────────────────────

def calculate_trade_alpha(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    windows: list[int] = [30, 60, 90],
) -> pd.DataFrame:
    """
    For each congressional trade, compute forward returns at 30/60/90-day windows
    and compare against SPY (S&P 500) over the same window. Alpha = trade return − SPY return.

    Args:
        trades_df:  DataFrame from fetch_congress_trades() with TransactionDate and Ticker columns.
        prices_df:  DataFrame of daily close prices from fetch_stock_prices() — tickers as columns.
        windows:    List of calendar-day windows to calculate returns for.

    Returns:
        trades_df with additional columns:
            return_30d, return_60d, return_90d  (trade's own return)
            spy_30d,    spy_60d,    spy_90d     (SPY return over same window)
            alpha_30d,  alpha_60d,  alpha_90d   (difference)
            AmountMidpoint                       (parsed dollar midpoint)
    """
    if prices_df.empty:
        logger.warning("prices_df is empty — skipping alpha calculation.")
        return trades_df

    df = trades_df.copy()

    # Parse dollar midpoints
    df["AmountMidpoint"] = df["Amount"].apply(parse_trade_amount)

    # Normalize sell transactions to negative for net calculations
    df["IsBuy"] = df["Transaction"].str.lower().str.contains("purchase")

    # Ensure prices index is DatetimeIndex, drop any NaT/NaN index values
    prices_df = prices_df.copy()
    prices_df.index = pd.to_datetime(prices_df.index, errors="coerce")
    prices_df = prices_df[prices_df.index.notna()]  # drop rows with unparseable dates
    price_dates = prices_df.index

    for window in windows:
        returns_col  = f"return_{window}d"
        spy_col      = f"spy_{window}d"
        alpha_col    = f"alpha_{window}d"

        trade_returns = []
        spy_returns   = []

        for _, row in df.iterrows():
            trade_date = pd.to_datetime(row["TransactionDate"])
            ticker     = row["Ticker"]

            # Find the nearest available price date on or after the trade date
            future_dates = price_dates[price_dates >= trade_date]
            if len(future_dates) == 0:
                trade_returns.append(np.nan)
                spy_returns.append(np.nan)
                continue

            start_idx = future_dates[0]
            # Find date ~window calendar days later
            target_date = start_idx + pd.Timedelta(days=window)
            end_candidates = price_dates[price_dates >= target_date]
            if len(end_candidates) == 0:
                trade_returns.append(np.nan)
                spy_returns.append(np.nan)
                continue
            end_idx = end_candidates[0]

            # Calculate return for the stock
            if ticker in prices_df.columns:
                p_start = prices_df.loc[start_idx, ticker]
                p_end   = prices_df.loc[end_idx, ticker]
                if pd.notna(p_start) and pd.notna(p_end) and p_start != 0:
                    t_ret = (p_end - p_start) / p_start
                else:
                    t_ret = np.nan
            else:
                t_ret = np.nan

            # Calculate SPY return over same window
            if "SPY" in prices_df.columns:
                s_start = prices_df.loc[start_idx, "SPY"]
                s_end   = prices_df.loc[end_idx, "SPY"]
                if pd.notna(s_start) and pd.notna(s_end) and s_start != 0:
                    s_ret = (s_end - s_start) / s_start
                else:
                    s_ret = np.nan
            else:
                s_ret = np.nan

            trade_returns.append(t_ret)
            spy_returns.append(s_ret)

        df[returns_col] = trade_returns
        df[spy_col]     = spy_returns
        df[alpha_col]   = df[returns_col] - df[spy_col]

    return df


# ── Convenience filter ─────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    politicians: list[str] | None = None,
    party: str | None = None,
    chamber: str | None = None,
    sectors: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_amount: float | None = None,
) -> pd.DataFrame:
    """
    Apply sidebar filter values to a trades DataFrame.
    All parameters are optional — pass only the ones you want to filter on.

    Args:
        df:          The master trades DataFrame (with AmountMidpoint and Sector columns).
        politicians: List of representative names to include (None = all).
        party:       "Democrat", "Republican", or None for all.
        chamber:     "House", "Senate", or None for all.
        sectors:     List of GICS sector names to include (None = all).
        start_date:  Filter trades on/after this date (string "YYYY-MM-DD").
        end_date:    Filter trades on/before this date (string "YYYY-MM-DD").
        min_amount:  Minimum AmountMidpoint to include.

    Returns:
        Filtered copy of df.
    """
    out = df.copy()

    if politicians:
        out = out[out["Representative"].isin(politicians)]
    if party and party.lower() not in ("both", "all"):
        out = out[out["Party"].str.lower() == party.lower()]
    if chamber and chamber.lower() not in ("both", "all"):
        out = out[out["Chamber"].str.lower() == chamber.lower()]
    if sectors and "Sector" in out.columns:
        out = out[out["Sector"].isin(sectors)]
    if start_date:
        out = out[out["TransactionDate"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["TransactionDate"] <= pd.to_datetime(end_date)]
    if min_amount is not None and "AmountMidpoint" in out.columns:
        out = out[out["AmountMidpoint"] >= min_amount]

    return out.reset_index(drop=True)
