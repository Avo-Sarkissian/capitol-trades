"""
components/scatter.py
Tab 3: Alpha Scatter Plot — each dot is a congressional trade.
X-axis = SPY return, Y-axis = trade return. Dots above the diagonal beat the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State

CHART_BG   = "#080e1a"
PAPER_BG   = "#080e1a"
GRID_COLOR = "#1c2e4a"
TEXT_COLOR = "#dce6f5"
GOLD       = "#d4a843"
DEM_BLUE   = "#3b82f6"
REP_RED    = "#ef4444"

# Maps dot size to trade amount (marker area ∝ amount)
MARKER_MIN_PX = 5
MARKER_MAX_PX = 20


def build_scatter_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Alpha Scatter tab layout with a window selector.

    Args:
        trades_df:  Filtered trades DataFrame (must include alpha columns).
        prices_df:  Unused — alpha already computed in trades_df.

    Returns:
        html.Div with window dropdown and graph.
    """
    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center", "marginBottom": "12px"},
            children=[
                html.Label("Return window:", style={"color": "#a0b4cc", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="scatter-window-select",
                    options=[
                        {"label": "30-day", "value": 30},
                        {"label": "60-day", "value": 60},
                        {"label": "90-day", "value": 90},
                    ],
                    value=60,
                    clearable=False,
                    style={"width": "140px", "fontSize": "12px"},
                ),
                html.Div(id="scatter-stats-bar", style={"marginLeft": "auto", "color": "#a0b4cc", "fontSize": "11px"}),
            ],
        ),
        dcc.Loading(
            type="circle",
            color=GOLD,
            children=dcc.Graph(
                id="scatter-chart",
                config={"displayModeBar": True, "scrollZoom": True},
                style={"height": "calc(100vh - 210px)"},
                figure=make_scatter_figure(trades_df, 60),
            ),
        ),
    ])


def make_scatter_figure(trades_df: pd.DataFrame, window: int = 60) -> go.Figure:
    """
    Build the scatter plot for a given return window.

    Args:
        trades_df: Filtered trades DataFrame with return_{window}d and spy_{window}d columns.
        window:    Integer day window (30, 60, or 90).

    Returns:
        Plotly Figure.
    """
    ret_col = f"return_{window}d"
    spy_col = f"spy_{window}d"

    if trades_df.empty or ret_col not in trades_df.columns:
        return _empty_fig(f"Alpha data not available. Run with full price history to compute {window}d returns.")

    df = trades_df.dropna(subset=[ret_col, spy_col]).copy()

    if df.empty:
        return _empty_fig("Not enough data to compute alpha for current filters.")

    # Scale marker sizes
    amounts = df["AmountMidpoint"].clip(lower=1)
    log_amounts = np.log1p(amounts)
    min_log, max_log = log_amounts.min(), log_amounts.max()
    if max_log > min_log:
        sizes = MARKER_MIN_PX + (log_amounts - min_log) / (max_log - min_log) * (MARKER_MAX_PX - MARKER_MIN_PX)
    else:
        sizes = pd.Series([MARKER_MIN_PX] * len(df), index=df.index)

    fig = go.Figure()

    # ── 45° reference line (beating market = above this line) ─────
    val_range = [
        min(df[ret_col].min(), df[spy_col].min()) - 0.05,
        max(df[ret_col].max(), df[spy_col].max()) + 0.05,
    ]
    fig.add_trace(go.Scatter(
        x=val_range, y=val_range,
        mode="lines",
        line={"color": "#7a90b0", "width": 1, "dash": "dash"},
        name="Market parity",
        hoverinfo="skip",
    ))

    # ── One trace per party for proper legend ──────────────────────
    for party, color in [("Democrat", DEM_BLUE), ("Republican", REP_RED)]:
        mask = df["Party"].str.lower() == party.lower()
        sub  = df[mask]
        if sub.empty:
            continue

        sub_sizes = sizes[mask]

        fig.add_trace(go.Scatter(
            x=sub[spy_col] * 100,     # convert to percentage
            y=sub[ret_col] * 100,
            mode="markers",
            name=party,
            marker={
                "color": color,
                "size": sub_sizes.tolist(),
                "opacity": 0.75,
                "line": {"color": "rgba(255,255,255,0.3)", "width": 0.5},
            },
            customdata=sub[[
                "Representative", "Ticker", "TransactionDate",
                "Amount", "AmountMidpoint",
            ]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Ticker: %{customdata[1]}<br>"
                "Date: %{customdata[2]}<br>"
                "Amount: %{customdata[3]}<br>"
                f"Trade return ({window}d): %{{y:.2f}}%<br>"
                f"SPY return ({window}d): %{{x:.2f}}%<br>"
                "Alpha: %{y:.2f}% vs %{x:.2f}%"
                "<extra></extra>"
            ),
        ))

    # ── Shaded "above market" region ──────────────────────────────
    fig.add_hrect(
        y0=val_range[0] * 100,
        y1=val_range[1] * 100,
        fillcolor="rgba(34,197,94,0.03)",
        layer="below",
        line_width=0,
    )

    # ── Layout ────────────────────────────────────────────────────
    # Compute aggregate stats for annotation
    beats_market = (df[ret_col] > df[spy_col]).sum()
    pct_beat     = beats_market / len(df) * 100
    avg_alpha    = (df[ret_col] - df[spy_col]).mean() * 100

    fig.add_annotation(
        text=(
            f"Trades beating market: <b>{beats_market}/{len(df)} ({pct_beat:.1f}%)</b><br>"
            f"Avg alpha: <b>{avg_alpha:+.2f}%</b>"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        showarrow=False,
        font={"color": GOLD, "size": 11, "family": "IBM Plex Mono, monospace"},
        bgcolor="rgba(10,22,40,0.85)",
        bordercolor=GOLD,
        borderwidth=1,
        borderpad=8,
    )

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono, monospace", "size": 11},
        margin={"l": 70, "r": 20, "t": 30, "b": 60},
        xaxis={
            "title": f"S&P 500 (SPY) {window}-Day Return (%)",
            "gridcolor": GRID_COLOR,
            "zeroline": True,
            "zerolinecolor": "#2e4570",
            "zerolinewidth": 1,
        },
        yaxis={
            "title": f"Trade {window}-Day Return (%)",
            "gridcolor": GRID_COLOR,
            "zeroline": True,
            "zerolinecolor": "#2e4570",
            "zerolinewidth": 1,
        },
        legend={
            "bgcolor": "rgba(15,31,61,0.9)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        hovermode="closest",
    )

    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR},
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 13, "color": "#7a90b0"},
        }],
    )
    return fig


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("scatter-chart", "figure"),
    Output("scatter-stats-bar", "children"),
    Input("scatter-window-select", "value"),
    Input("store-filtered-trades", "data"),
)
def update_scatter(window: int, store_data: str):
    """Re-render scatter when window or filters change."""
    if not store_data:
        return _empty_fig("No data."), ""

    df = pd.read_json(store_data, orient="split")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["AmountMidpoint"]  = pd.to_numeric(df["AmountMidpoint"], errors="coerce").fillna(0)

    ret_col = f"return_{window}d"
    spy_col = f"spy_{window}d"
    valid   = df.dropna(subset=[ret_col, spy_col]) if ret_col in df.columns else pd.DataFrame()

    stats_text = ""
    if not valid.empty:
        pct = (valid[ret_col] > valid[spy_col]).mean() * 100
        stats_text = f"{len(valid):,} trades with alpha data · {pct:.1f}% beat SPY"

    return make_scatter_figure(df, window), stats_text
