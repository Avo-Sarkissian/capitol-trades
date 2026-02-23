"""
components/timeline.py
Tab 1: Trade Timeline — stock price line chart with congressional buy/sell markers overlaid.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from components.constants import (
    CHART_BG, PAPER_BG, GRID_COLOR, TEXT_COLOR, GOLD,
    BUY_COLOR, SELL_COLOR, CHART_FONT, empty_fig,
)

# Marker symbol size is scaled by trade amount — this sets the min/max px
MARKER_MIN_PX = 8
MARKER_MAX_PX = 28


def _scale_marker(amount: float, min_a: float, max_a: float) -> float:
    """Map a dollar amount to a marker pixel size between MARKER_MIN/MAX_PX."""
    if max_a == min_a or max_a == 0:
        return MARKER_MIN_PX
    scaled = (amount - min_a) / (max_a - min_a)
    return MARKER_MIN_PX + scaled * (MARKER_MAX_PX - MARKER_MIN_PX)


def build_timeline_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Trade Timeline tab layout.
    Renders an initial figure for the top politician and a politician dropdown
    to switch between subjects.
    """
    if trades_df.empty:
        return html.Div(
            "No trades match current filters.",
            style={"color": "#7a90b0", "padding": "40px", "textAlign": "center"},
        )

    top_politicians = (
        trades_df.groupby("Representative")["AmountMidpoint"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    if not top_politicians:
        return html.Div(
            "No politicians found in current filters.",
            style={"color": "#7a90b0", "padding": "40px", "textAlign": "center"},
        )

    default_pol = top_politicians[0]

    # Pre-populate ticker options for the default politician so the chart
    # renders immediately without waiting for a callback round-trip.
    default_tickers = sorted(
        trades_df[trades_df["Representative"] == default_pol]["Ticker"]
        .dropna().unique().tolist()
    )
    default_ticker = default_tickers[0] if default_tickers else None

    pol_options    = [{"label": p, "value": p} for p in top_politicians]
    ticker_options = [{"label": t, "value": t} for t in default_tickers]

    return html.Div([
        # ── Controls bar ──────────────────────────────────────
        html.Div(
            className="chart-controls",
            children=[
                html.Label("Politician", className="filter-label"),
                dcc.Dropdown(
                    id="timeline-politician-select",
                    options=pol_options,
                    value=default_pol,
                    clearable=False,
                    optionHeight=30,
                    style={"width": "260px"},
                ),
                html.Label("Ticker", className="filter-label", style={"marginLeft": "4px"}),
                dcc.Dropdown(
                    id="timeline-ticker-select",
                    options=ticker_options,
                    value=default_ticker,
                    clearable=False,
                    optionHeight=30,
                    style={"width": "130px"},
                ),
            ],
        ),

        # ── Chart ──────────────────────────────────────────────
        dcc.Loading(
            type="circle",
            color=GOLD,
            children=dcc.Graph(
                id="timeline-chart",
                config={"displayModeBar": True, "scrollZoom": True},
                style={"height": "calc(100vh - 220px)"},
            ),
        ),
    ])


def make_timeline_figure(
    politician: str,
    ticker: str,
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> go.Figure:
    """
    Build the price + trade marker figure for one politician / ticker pair.
    """
    # Filter to this politician's trades for this ticker
    mask = (trades_df["Representative"] == politician) & (trades_df["Ticker"] == ticker)
    pol_trades = trades_df[mask].copy()

    # Get price series
    prices_df.index = pd.to_datetime(prices_df.index)
    price_series = prices_df.get(ticker, pd.Series(dtype=float))

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"{ticker} Price  ·  {politician}'s Trades",
            "Trade Frequency (weekly count)",
        ],
    )

    # ── Price line ─────────────────────────────────────────────
    if not price_series.empty:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                name=ticker,
                line={"color": "#60a5fa", "width": 1.5},
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ── Buy / sell markers ─────────────────────────────────────
    if not pol_trades.empty:
        amount_min = pol_trades["AmountMidpoint"].min()
        amount_max = pol_trades["AmountMidpoint"].max()

        for _, row in pol_trades.iterrows():
            trade_date = pd.to_datetime(row["TransactionDate"])
            is_buy     = "purchase" in str(row["Transaction"]).lower()
            color      = BUY_COLOR if is_buy else SELL_COLOR
            symbol     = "triangle-up" if is_buy else "triangle-down"
            size       = _scale_marker(row["AmountMidpoint"], amount_min, amount_max)

            # Find price at trade date (nearest available, clamped to valid range)
            if not price_series.empty:
                idx = price_series.index.searchsorted(trade_date, side="left")
                idx = max(0, min(idx, len(price_series) - 1))
                y_val = price_series.iloc[idx]
                # If we landed before the trade date and there's a next point, use it
                if idx < len(price_series) - 1 and price_series.index[idx] < trade_date:
                    idx += 1
                y_val = float(price_series.iloc[idx])
                if pd.isna(y_val):
                    y_val = 0
            else:
                y_val = 0

            fig.add_trace(
                go.Scatter(
                    x=[trade_date],
                    y=[y_val],
                    mode="markers",
                    marker={
                        "symbol": symbol,
                        "size": size,
                        "color": color,
                        "line": {"color": "white", "width": 1},
                    },
                    name=row["Transaction"],
                    hovertemplate=(
                        f"<b>{politician}</b><br>"
                        f"Ticker: {ticker}<br>"
                        f"Date: {trade_date.strftime('%b %d, %Y')}<br>"
                        f"Type: {row['Transaction']}<br>"
                        f"Amount: {row['Amount']}<br>"
                        f"Est. Value: ${row['AmountMidpoint']:,.0f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── Weekly trade count bar ─────────────────────────────────
    if not pol_trades.empty:
        weekly = (
            pol_trades.set_index("TransactionDate")
            .resample("W")["Ticker"]
            .count()
            .reset_index()
        )
        weekly.columns = ["Week", "Count"]

        fig.add_trace(
            go.Bar(
                x=weekly["Week"],
                y=weekly["Count"],
                marker_color=GOLD,
                name="Trade count",
                hovertemplate="%{x|Week of %b %d}<br>%{y} trade(s)<extra></extra>",
            ),
            row=2, col=1,
        )

    # ── Manual buy/sell legend entries ────────────────────────
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker={"symbol": "triangle-up", "size": 10, "color": BUY_COLOR},
                   name="Buy"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker={"symbol": "triangle-down", "size": 10, "color": SELL_COLOR},
                   name="Sell"),
        row=1, col=1,
    )

    # ── Styling ────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="closest",
        showlegend=True,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickfont={"size": 10},
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        showgrid=True,
        zeroline=False,
        tickfont={"size": 10},
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="# Trades",   row=2, col=1)

    return fig


# ── Callbacks ──────────────────────────────────────────────────────────────────
from dash import callback, Input, Output  # noqa: E402


@callback(
    Output("timeline-ticker-select", "options"),
    Output("timeline-ticker-select", "value"),
    Input("timeline-politician-select", "value"),
    Input("store-filtered-trades", "data"),
)
def update_ticker_options(politician: str, store_data: str):
    """Populate the ticker dropdown with tickers traded by the selected politician."""
    import data.state as _state

    if not politician:
        return [], None

    df = _state.deserialize_store(store_data)
    if df is None:
        df = _state.trades_df if not _state.trades_df.empty else None
    if df is None:
        return [], None

    tickers = sorted(
        df[df["Representative"] == politician]["Ticker"]
        .dropna().unique().tolist()
    )
    options = [{"label": t, "value": t} for t in tickers]
    default = tickers[0] if tickers else None
    return options, default


@callback(
    Output("timeline-chart", "figure"),
    Input("timeline-politician-select", "value"),
    Input("timeline-ticker-select", "value"),
    Input("store-filtered-trades", "data"),
)
def update_timeline(politician: str, ticker: str, store_data: str):
    """Re-render the timeline chart when politician or ticker changes."""
    import data.state as _state

    if not politician or not ticker:
        return empty_fig("Select a politician and ticker to view trades.")

    df = _state.deserialize_store(store_data)
    if df is None:
        df = _state.trades_df.copy() if not _state.trades_df.empty else None
    if df is None:
        return empty_fig("No data available.")

    return make_timeline_figure(politician, ticker, df, _state.prices_df)
