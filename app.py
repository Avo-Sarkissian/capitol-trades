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
ALL_SECTORS     = sorted(trades_df["Sector"].dropna().unique().tolist())

# ══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(
    __name__,
    title="Capitol Trades",
    # Suppress callback exceptions for components that load dynamically
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
            html.Div("Filters", className="sidebar-section-title"),

            # ── Politician selector ────────────────────────────
            html.Label("Politician(s)", className="filter-label"),
            dcc.Dropdown(
                id="filter-politicians",
                options=[{"label": p, "value": p} for p in ALL_POLITICIANS],
                multi=True,
                placeholder="All politicians",
                style={"marginBottom": "10px"},
            ),

            # ── Party ─────────────────────────────────────────
            html.Label("Party", className="filter-label"),
            dcc.Dropdown(
                id="filter-party",
                options=[
                    {"label": "Both", "value": "Both"},
                    {"label": "Democrat", "value": "Democrat"},
                    {"label": "Republican", "value": "Republican"},
                ],
                value="Both",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            # ── Chamber ───────────────────────────────────────
            html.Label("Chamber", className="filter-label"),
            dcc.Dropdown(
                id="filter-chamber",
                options=[
                    {"label": "Both", "value": "Both"},
                    {"label": "House", "value": "House"},
                    {"label": "Senate", "value": "Senate"},
                ],
                value="Both",
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            # ── Sector ────────────────────────────────────────
            html.Label("Sector(s)", className="filter-label"),
            dcc.Dropdown(
                id="filter-sectors",
                options=[{"label": s, "value": s} for s in ALL_SECTORS],
                multi=True,
                placeholder="All sectors",
                style={"marginBottom": "10px"},
            ),

            # ── Date range ────────────────────────────────────
            html.Label("Date Range", className="filter-label"),
            dcc.DatePickerRange(
                id="filter-dates",
                min_date_allowed=trades_df["TransactionDate"].min().date(),
                max_date_allowed=trades_df["TransactionDate"].max().date(),
                start_date=trades_df["TransactionDate"].min().date(),
                end_date=trades_df["TransactionDate"].max().date(),
                display_format="MM/DD/YY",
                style={"marginBottom": "10px", "width": "100%"},
            ),

            # ── Min trade size ────────────────────────────────
            html.Label("Min Trade Size", className="filter-label"),
            dcc.Dropdown(
                id="filter-min-amount",
                options=[
                    {"label": "Any size",       "value": 0},
                    {"label": "$1K+",           "value": 1_000},
                    {"label": "$15K+",          "value": 15_000},
                    {"label": "$50K+",          "value": 50_000},
                    {"label": "$100K+",         "value": 100_000},
                    {"label": "$250K+",         "value": 250_000},
                    {"label": "$500K+",         "value": 500_000},
                    {"label": "$1M+",           "value": 1_000_000},
                ],
                value=0,
                clearable=False,
                style={"marginBottom": "10px"},
            ),

            # ── Stats mini-panel ──────────────────────────────
            html.Hr(style={"borderColor": "#1e3358", "margin": "8px 0"}),
            html.Div(id="sidebar-stats", style={"fontSize": "11px", "color": "#7a90b0"}),
        ],
    )


def build_tabs() -> dcc.Tabs:
    """Build the main tab strip and content panels."""
    tab_style = {"padding": "8px 14px"}

    return dcc.Tabs(
        id="main-tabs",
        value="tab-timeline",
        className="custom-tabs",
        children=[
            dcc.Tab(
                label="Trade Timeline",
                value="tab-timeline",
                style=tab_style,
                selected_style=tab_style,
            ),
            dcc.Tab(
                label="Sector Heatmap",
                value="tab-heatmap",
                style=tab_style,
                selected_style=tab_style,
            ),
            dcc.Tab(
                label="Alpha Scatter",
                value="tab-scatter",
                style=tab_style,
                selected_style=tab_style,
            ),
            dcc.Tab(
                label="Committee Network",
                value="tab-network",
                style=tab_style,
                selected_style=tab_style,
            ),
            dcc.Tab(
                label="Leaderboard",
                value="tab-leaderboard",
                style=tab_style,
                selected_style=tab_style,
            ),
        ],
    )


app.layout = html.Div(
    id="app-wrapper",
    children=[

        # ── Header ────────────────────────────────────────────
        html.Div(
            id="header",
            children=[
                html.Span("CAPITOL TRADES", id="header-logo"),
                html.Span(
                    "Congressional Stock Trading Intelligence Dashboard",
                    id="header-subtitle",
                ),
                html.Span("LIVE DATA ●", id="header-badge"),
            ],
        ),

        # ── Body (sidebar + tabs) ──────────────────────────────
        html.Div(
            id="body-layout",
            children=[
                build_sidebar(),

                html.Div(
                    id="main-content",
                    children=[
                        build_tabs(),
                        # Tab content rendered dynamically by callback below
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

        # ── Footer ────────────────────────────────────────────
        html.Div(
            id="footer",
            children=[
                html.Span("Data: Quiver Quantitative (congressional disclosures) + Yahoo Finance (prices)"),
                html.Span("Disclosure lag may affect alpha calculations · Not financial advice"),
            ],
        ),

        # ── Global data store (holds filtered trades as JSON) ──
        dcc.Store(id="store-filtered-trades"),
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("store-filtered-trades", "data"),
    Output("sidebar-stats", "children"),
    Input("filter-politicians", "value"),
    Input("filter-party", "value"),
    Input("filter-chamber", "value"),
    Input("filter-sectors", "value"),
    Input("filter-dates", "start_date"),
    Input("filter-dates", "end_date"),
    Input("filter-min-amount", "value"),
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

    # Sidebar mini-stats
    n_trades     = len(filtered)
    n_politicians = filtered["Representative"].nunique()
    total_vol    = filtered["AmountMidpoint"].sum()

    stats = [
        html.Div(f"{n_trades:,} trades", style={"fontWeight": "bold", "color": "#d4a843"}),
        html.Div(f"{n_politicians} politicians"),
        html.Div(f"Est. vol: ${total_vol/1e6:.1f}M"),
    ]

    # Serialize dates for JSON store
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

    # Parse stored JSON back to DataFrame
    if store_data:
        df = pd.read_json(store_data, orient="split")
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    else:
        df = trades_df.copy()
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

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
