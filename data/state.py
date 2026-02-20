"""
data/state.py
Module-level shared state — holds runtime data (prices_df) so component callbacks
can access it without creating circular imports back to app.py.
Populated once by app.py at startup.
"""

import pandas as pd

# These are set by app.py after data loading is complete.
# Components import from here instead of from app.py directly.
prices_df: pd.DataFrame = pd.DataFrame()
trades_df: pd.DataFrame = pd.DataFrame()
