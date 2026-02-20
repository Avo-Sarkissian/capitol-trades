"""
app.py — Capitol Trades Dashboard
Entry point. Initializes Dash, loads data, defines layout, wires callbacks.
Keep this file thin — visualization logic lives in components/.
"""

from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback

# ── Data loading (runs once at startup) ───────────────────────────────────────
from data.fetch import fetch_congress_trades, fetch_stock_prices
from data.process import (
    parse_trade_amount,
    get_sector_mapping,
    calculate_trade_alpha,
)
import data.state as _state  # shared runtime state (avoids circular imports)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("Loading congressional trade data...")
trades_raw = fetch_congress_trades()

# Parse dollar midpoints upfront
trades_raw["AmountMidpoint"] = trades_raw["Amount"].apply(parse_trade_amount)

# Determine date range for price fetching
min_date = trades_raw["TransactionDate"].min()
max_date = datetime.now()
price_start = (min_date - timedelta(days=5)).strftime("%Y-%m-%d")
price_end   = (max_date + timedelta(days=95)).strftime("%Y-%m-%d")

print(f"Fetching prices from {price_start} to {price_end}...")
unique_tickers = trades_raw["Ticker"].dropna().unique().tolist()
prices_df = fetch_stock_prices(unique_tickers, price_start, price_end)

print("Computing sector mapping...")
sector_map = get_sector_mapping(unique_tickers[:30])
ticker_sector = dict(zip(sector_map["Ticker"], sector_map["Sector"]))
trades_raw["Sector"] = trades_raw["Ticker"].map(ticker_sector).fillna("Unknown")

print("Calculating alpha...")
trades_df = calculate_trade_alpha(trades_raw, prices_df, windows=[30, 60, 90])

print(f"Ready — {len(trades_df)} trades loaded.\n")

# ── Populate shared state (BEFORE importing components) ───────────────────────
_state.trades_df = trades_df
_state.prices_df = prices_df

# ── Component callbacks (importing registers them with Dash) ──────────────────
import components.timeline    as _tl   # noqa: F401, E402
import components.heatmap     as _hm   # noqa: F401, E402
import components.scatter     as _sc   # noqa: F401, E402
import components.network     as _nw   # noqa: F401, E402
import components.leaderboard as _lb   # noqa: F401, E402

# ── Derived lists for dropdowns ───────────────────────────────────────────────
ALL_POLITICIANS = sorted(trades_df["Representative"].dropna().unique().tolist())
ALL_SECTORS     = sorted([s for s in trades_df["Sector"].dropna().unique() if s != "Unknown"])

DATE_MIN = trades_df["TransactionDate"].min().strftime("%Y-%m-%d")
DATE_MAX = trades_df["TransactionDate"].max().strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(
    __name__,
    title="Capitol Trades",
    suppress_callback_exceptions=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════


def _filter_block(title: str, children) -> html.Div:
    """Wrap a filter in a labeled section block."""
    return html.Div(className="filter-block", children=[
        html.Div(title, className="filter-title"),
        *([children] if not isinstance(children, list) else children),
    ])


def build_sidebar() -> html.Div:
    """Build the left sidebar with all global filter controls."""
    return html.Div(id="sidebar", children=[

        # ── Politician ────────────────────────────────────────
        _filter_block("POLITICIAN", dcc.Dropdown(
            id="filter-politicians",
            options=[{"label": p, "value": p} for p in ALL_POLITICIANS],
            multi=True,
            placeholder="All politicians",
        )),

        # ── Party (radio pills) ───────────────────────────────
        _filter_block("PARTY", dcc.RadioItems(
            id="filter-party",
            options=["Both", "Democrat", "Republican"],
            value="Both",
            className="radio-pill",
            inline=True,
        )),

        # ── Chamber (radio pills) ─────────────────────────────
        _filter_block("CHAMBER", dcc.RadioItems(
            id="filter-chamber",
            options=["Both", "House", "Senate"],
            value="Both",
            className="radio-pill",
            inline=True,
        )),

        # ── Sector ────────────────────────────────────────────
        _filter_block("SECTOR", dcc.Dropdown(
            id="filter-sectors",
            options=[{"label": s, "value": s} for s in ALL_SECTORS],
            multi=True,
            placeholder="All sectors",
        )),

        # ── Date range ─────────────────────────────────────────
        _filter_block("DATE RANGE", [
            html.Div(className="date-row", children=[
                dcc.DatePickerSingle(
                    id="filter-start-date",
                    date=DATE_MIN,
                    min_date_allowed=DATE_MIN,
                    max_date_allowed=DATE_MAX,
                    display_format="MM/DD/YY",
                    className="dark-datepicker",
                ),
                html.Span("→", className="date-arrow"),
                dcc.DatePickerSingle(
                    id="filter-end-date",
                    date=DATE_MAX,
                    min_date_allowed=DATE_MIN,
                    max_date_allowed=DATE_MAX,
                    display_format="MM/DD/YY",
                    className="dark-datepicker",
                ),
            ]),
        ]),

        # ── Min size ──────────────────────────────────────────
        _filter_block("MIN TRADE SIZE", dcc.Dropdown(
            id="filter-min-amount",
            options=[
                {"label": "Any size", "value": 0},
                {"label": "$15K+",    "value": 15_000},
                {"label": "$50K+",    "value": 50_000},
                {"label": "$100K+",   "value": 100_000},
                {"label": "$250K+",   "value": 250_000},
                {"label": "$500K+",   "value": 500_000},
                {"label": "$1M+",     "value": 1_000_000},
            ],
            value=0,
            clearable=False,
        )),

        # ── Stats ─────────────────────────────────────────────
        html.Div(id="sidebar-stats", className="sidebar-stats"),
    ])


def build_tabs() -> dcc.Tabs:
    """Build the tab strip."""
    return dcc.Tabs(
        id="main-tabs",
        value="tab-timeline",
        className="custom-tabs",
        children=[
            dcc.Tab(label="Trade Timeline",    value="tab-timeline",    className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Sector Heatmap",    value="tab-heatmap",     className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Alpha Scatter",     value="tab-scatter",     className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Committee Network", value="tab-network",     className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Leaderboard",       value="tab-leaderboard", className="custom-tab", selected_className="custom-tab--selected"),
        ],
    )


app.layout = html.Div(id="app-wrapper", children=[

    # ── Header ────────────────────────────────────────────────────
    html.Div(id="header", children=[
        html.Div(id="header-left", children=[
            html.Span("CAPITOL TRADES", id="header-logo"),
            html.Span("Congressional Stock Trading Dashboard", id="header-subtitle"),
        ]),
        html.Div(id="header-right", children=[
            html.Span(className="live-dot"),
            html.Span("LIVE", id="header-badge"),
        ]),
    ]),

    # ── Body ──────────────────────────────────────────────────────
    html.Div(id="body-layout", children=[
        build_sidebar(),
        html.Div(id="main-content", children=[
            build_tabs(),
            html.Div(id="tab-content", className="tab-content"),
        ]),
    ]),

    # ── Footer ────────────────────────────────────────────────────
    html.Div(id="footer", children=[
        html.Span("Quiver Quantitative · Yahoo Finance"),
        html.Span("Not financial advice"),
    ]),

    # ── Data store ────────────────────────────────────────────────
    dcc.Store(id="store-filtered-trades"),
])

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("store-filtered-trades", "data"),
    Output("sidebar-stats", "children"),
    Input("filter-politicians",  "value"),
    Input("filter-party",        "value"),
    Input("filter-chamber",      "value"),
    Input("filter-sectors",      "value"),
    Input("filter-start-date",   "date"),
    Input("filter-end-date",     "date"),
    Input("filter-min-amount",   "value"),
)
def update_store(politicians, party, chamber, sectors, start_date, end_date, min_amount):
    """Apply sidebar filters → push filtered trades into dcc.Store."""
    from data.process import apply_filters

    filtered = apply_filters(
        trades_df,
        politicians=politicians or None,
        party=party if party != "Both" else None,
        chamber=chamber if chamber != "Both" else None,
        sectors=sectors or None,
        start_date=start_date,
        end_date=end_date,
        min_amount=float(min_amount) if min_amount else 0,
    )

    n = len(filtered)
    n_pol = filtered["Representative"].nunique()
    vol   = filtered["AmountMidpoint"].sum()

    stats = html.Div(className="stats-grid", children=[
        html.Div(className="stat-chip", children=[
            html.Span(f"{n:,}",              className="stat-val"),
            html.Span("trades",              className="stat-lbl"),
        ]),
        html.Div(className="stat-chip", children=[
            html.Span(str(n_pol),            className="stat-val"),
            html.Span("politicians",         className="stat-lbl"),
        ]),
        html.Div(className="stat-chip", children=[
            html.Span(f"${vol/1e6:.1f}M",   className="stat-val"),
            html.Span("est. volume",         className="stat-lbl"),
        ]),
    ])

    out = filtered.copy()
    out["TransactionDate"] = out["TransactionDate"].astype(str)
    return out.to_json(date_format="iso", orient="split"), stats


@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("store-filtered-trades", "data"),
)
def render_tab(tab_value: str, store_data: str):
    """Route tab selection to the appropriate visualization component."""
    from components.timeline    import build_timeline_tab
    from components.heatmap     import build_heatmap_tab
    from components.scatter     import build_scatter_tab
    from components.network     import build_network_tab
    from components.leaderboard import build_leaderboard_tab

    if store_data:
        df = pd.read_json(StringIO(store_data), orient="split")
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    else:
        df = trades_df.copy()

    dispatch = {
        "tab-timeline":    build_timeline_tab,
        "tab-heatmap":     build_heatmap_tab,
        "tab-scatter":     build_scatter_tab,
        "tab-network":     build_network_tab,
        "tab-leaderboard": build_leaderboard_tab,
    }

    build_fn = dispatch.get(tab_value)
    if build_fn:
        return build_fn(df, prices_df)
    return html.Div("Unknown tab")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
