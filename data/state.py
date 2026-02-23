"""
data/state.py
Module-level shared state — holds runtime data (prices_df) so component callbacks
can access it without creating circular imports back to app.py.
Populated once by app.py at startup.
"""

from io import StringIO

import pandas as pd

# These are set by app.py after data loading is complete.
# Components import from here instead of from app.py directly.
prices_df: pd.DataFrame = pd.DataFrame()
trades_df: pd.DataFrame = pd.DataFrame()

# Set to True if mock data was used instead of real API data
using_mock_data: bool = False

# Required columns that must exist after JSON deserialization
_REQUIRED_COLS = {"Representative", "TransactionDate", "Ticker", "Transaction",
                  "Amount", "Party", "AmountMidpoint"}


def deserialize_store(store_data: str | None) -> pd.DataFrame | None:
    """
    Safely deserialize the dcc.Store JSON string into a DataFrame.
    Returns None if store_data is empty/invalid or missing required columns.
    Handles type coercion for TransactionDate and AmountMidpoint.
    """
    if not store_data:
        return None

    try:
        df = pd.read_json(StringIO(store_data), orient="split")
    except (ValueError, KeyError):
        return None

    # Validate required columns exist
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        return None

    # Coerce types
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["AmountMidpoint"] = pd.to_numeric(df["AmountMidpoint"], errors="coerce").fillna(0)

    return df
