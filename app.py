"""
app.py — Capitol Trades Dashboard
Entry point. Initializes Dash, loads data, defines layout, wires callbacks.
Keep this file thin — visualization logic lives in components/.
"""

from datetime import datetime, timedelta

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback

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
# Add 90 extra days after last trade for alpha window calculations
price_start = (min_date - timedelta(days=5)).strftime("%Y-%m-%d")
price_end   = (max_date + timedelta(days=95)).strftime("%Y-%m-%d")

print(f"Fetching prices from {price_start} to {price_end}...")
unique_tickers = trades_raw["Ticker"].dropna().unique().tolist()
prices_df = fetch_stock_prices(unique_tickers, price_start, price_end)

print("Computing sector mapping...")
sector_map = get_sector_mapping(unique_tickers[:30])  # limit to 30 on first run to stay fast
ticker_sector = dict(zip(sector_map["Ticker"], sector_map["Sector"]))
trades_raw["Sector"] = trades_raw["Ticker"].map(ticker_sector).fillna("Unknown")

print("Calculating alpha...")
trades_df = calculate_trade_alpha(trades_raw, prices_df, windows=[30, 60, 90])

print(f"Ready — {len(trades_df)} trades loaded.\n")

# ── Populate shared state (must happen BEFORE importing components) ────────────
_state.trades_df = trades_df
_state.prices_df = prices_df

# ── Component callbacks (importing registers them with Dash) ───────────────────
# These must be imported AFTER data is loaded so state is available.
import components.timeline    as _tl   # noqa: F401, E402
import components.heatmap     as _hm   # noqa: F401, E402
import components.scatter     as _sc   # noqa: F401, E402
import components.network     as _nw   # noqa: F401, E402
import components.leaderboard as _lb   # noqa: F401, E402

# ── Derived lists for dropdowns ────────────────────────────────────────────────
ALL_POLITICIANS = sorted(trades_df["Representative"].dropna().unique().tolist())
ALL_SECTORS     = sorted([s for s in trades_df["Sector"].dropna().unique() if s != "Unknown"])

# Date bounds for the date inputs (pre-formatted as YYYY-MM-DD strings)
DATE_MIN = trades_df["TransactionDate"].min().strftime("%Y-%m-%d")
DATE_MAX = trades_df["TransactionDate"].max().strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(
    __name__,
    title="Capitol Trades",
    suppress_callback_exceptions=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def build_sidebar() -> html.Div:
    """Build the left sidebar with all global filter controls."""
    return html.Div(
        id="sidebar",
        children=[

            # ── Politician ────────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Politician", className="sidebar-section-title"),
                dcc.Dropdown(
                    id="filter-politicians",
                    options=[{"label": p, "value": p} for p in ALL_POLITICIANS],
                    multi=True,
                    placeholder="All politicians",
                    optionHeight=30,
                ),
            ]),

            # ── Party ─────────────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Party", className="sidebar-section-title"),
                dcc.RadioItems(
                    id="filter-party",
                    options=[
                        {"label": "Both",       "value": "Both"},
                        {"label": "Democrat",   "value": "Democrat"},
                        {"label": "Republican", "value": "Republican"},
                    ],
                    value="Both",
                    className="radio-pill",
                    inputStyle={"display": "none"},
                ),
            ]),

            # ── Chamber ───────────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Chamber", className="sidebar-section-title"),
                dcc.RadioItems(
                    id="filter-chamber",
                    options=[
                        {"label": "Both",   "value": "Both"},
                        {"label": "House",  "value": "House"},
                        {"label": "Senate", "value": "Senate"},
                    ],
                    value="Both",
                    className="radio-pill",
                    inputStyle={"display": "none"},
                ),
            ]),

            # ── Sector ────────────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Sector", className="sidebar-section-title"),
                dcc.Dropdown(
                    id="filter-sectors",
                    options=[{"label": s, "value": s} for s in ALL_SECTORS],
                    multi=True,
                    placeholder="All sectors",
                    optionHeight=30,
                ),
            ]),

            # ── Date Range ────────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Date Range", className="sidebar-section-title"),
                html.Div(className="date-row", children=[
                    dcc.Input(
                        id="filter-start-date",
                        type="date",
                        value=DATE_MIN,
                        min=DATE_MIN,
                        max=DATE_MAX,
                    ),
                    html.Span("→"),
                    dcc.Input(
                        id="filter-end-date",
                        type="date",
                        value=DATE_MAX,
                        min=DATE_MIN,
                        max=DATE_MAX,
                    ),
                ]),
            ]),

            # ── Min Trade Size ────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div("Min Trade Size", className="sidebar-section-title"),
                dcc.Dropdown(
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
                    optionHeight=30,
                ),
            ]),

            # ── Stats strip ───────────────────────────────────────
            html.Div(className="sidebar-section", children=[
                html.Div(id="sidebar-stats"),
            ]),
        ],
    )


def build_tabs() -> dcc.Tabs:
    """Build the main tab strip."""
    return dcc.Tabs(
        id="main-tabs",
        value="tab-timeline",
        className="custom-tabs",
        children=[
            dcc.Tab(label="Trade Timeline",    value="tab-timeline"),
            dcc.Tab(label="Sector Heatmap",    value="tab-heatmap"),
            dcc.Tab(label="Alpha Scatter",     value="tab-scatter"),
            dcc.Tab(label="Committee Network", value="tab-network"),
            dcc.Tab(label="Leaderboard",       value="tab-leaderboard"),
        ],
    )


app.layout = html.Div(
    id="app-wrapper",
    children=[

        # ── Header ────────────────────────────────────────────────
        html.Div(
            id="header",
            children=[
                html.Span("CAPITOL TRADES", id="header-logo"),
                html.Span(
                    "Congressional Stock Trading Intelligence Dashboard",
                    id="header-subtitle",
                ),
                html.Span("LIVE DATA", id="header-badge"),
            ],
        ),

        # ── Body ──────────────────────────────────────────────────
        html.Div(
            id="body-layout",
            children=[
                build_sidebar(),

                html.Div(
                    id="main-content",
                    children=[
                        build_tabs(),
                        dcc.Loading(
                            id="tab-loading",
                            type="circle",
                            color="#d4a843",
                            children=html.Div(id="tab-content", className="tab-content"),
                        ),
                    ],
                ),
            ],
        ),

        # ── Footer ────────────────────────────────────────────────
        html.Div(
            id="footer",
            children=[
                html.Span("Data: Quiver Quantitative (congressional disclosures) + Yahoo Finance (market prices)"),
                html.Span("Disclosure lag may affect alpha calculations · Not financial advice"),
            ],
        ),

        # ── Global data store ─────────────────────────────────────
        dcc.Store(id="store-filtered-trades"),
    ],
)

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
    Input("filter-start-date",   "value"),
    Input("filter-end-date",     "value"),
    Input("filter-min-amount",   "value"),
)
def update_store(politicians, party, chamber, sectors, start_date, end_date, min_amount):
    """
    Apply sidebar filters to the master trades DataFrame and push the result
    into dcc.Store so all visualisation callbacks can read from it.
    """
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

    n_trades      = len(filtered)
    n_politicians = filtered["Representative"].nunique()
    total_vol     = filtered["AmountMidpoint"].sum()

    # Three tidy stat chips in the sidebar
    stats = [
        html.Div(className="stat-chip", children=[
            html.Span("Trades",     className="stat-chip-label"),
            html.Span(f"{n_trades:,}", className="stat-chip-value"),
        ]),
        html.Div(className="stat-chip", children=[
            html.Span("Politicians",      className="stat-chip-label"),
            html.Span(str(n_politicians), className="stat-chip-value"),
        ]),
        html.Div(className="stat-chip", children=[
            html.Span("Est. Volume",             className="stat-chip-label"),
            html.Span(f"${total_vol/1e6:.1f}M",  className="stat-chip-value"),
        ]),
    ]

    out = filtered.copy()
    out["TransactionDate"] = out["TransactionDate"].astype(str)
    return out.to_json(date_format="iso", orient="split"), stats


@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    State("store-filtered-trades", "data"),
)
def render_tab(tab_value: str, store_data: str):
    """Route tab selection to the appropriate visualization component."""
    from components.timeline    import build_timeline_tab
    from components.heatmap     import build_heatmap_tab
    from components.scatter     import build_scatter_tab
    from components.network     import build_network_tab
    from components.leaderboard import build_leaderboard_tab

    if store_data:
        df = pd.read_json(store_data, orient="split")
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
    return html.Div("Unknown tab", style={"color": "red"})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
