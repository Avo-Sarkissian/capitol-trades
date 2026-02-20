"""
components/leaderboard.py
Tab 5: Leaderboard — top traders by volume and by alpha, plus summary stats panel.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State

CHART_BG   = "#06090f"
PAPER_BG   = "#06090f"
GRID_COLOR = "#1e2a36"
TEXT_COLOR = "#e2e8f0"
GOLD       = "#f0c040"
DEM_BLUE   = "#3b82f6"
REP_RED    = "#ef4444"

TOP_N = 20  # number of politicians shown in each ranking


def build_leaderboard_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Leaderboard tab: volume ranking, alpha ranking, and a summary stats row.

    Args:
        trades_df: Filtered trades DataFrame.
        prices_df: Unused — kept for consistent call signature.

    Returns:
        html.Div with stats row + two ranking charts.
    """
    summary = _build_summary_stats(trades_df)
    vol_fig  = make_volume_leaderboard(trades_df)
    alpha_fig = make_alpha_leaderboard(trades_df)

    return html.Div([
        # ── Summary stat cards ─────────────────────────────────
        html.Div(
            id="leaderboard-stats-row",
            style={
                "display": "flex",
                "gap": "12px",
                "marginBottom": "16px",
                "flexWrap": "wrap",
            },
            children=summary,
        ),

        # ── Two charts side by side ────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "12px", "height": "calc(100vh - 260px)"},
            children=[
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Div("Top Traders by Estimated Volume",
                                 style={"color": GOLD, "fontSize": "11px",
                                        "fontFamily": "IBM Plex Mono, monospace",
                                        "marginBottom": "6px"}),
                        dcc.Loading(
                            type="circle", color=GOLD,
                            children=dcc.Graph(
                                id="leaderboard-vol-chart",
                                config={"displayModeBar": False},
                                style={"height": "100%"},
                                figure=vol_fig,
                            ),
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Div(
                            "Top Traders by 60-Day Alpha (min 10 trades)",
                            style={"color": GOLD, "fontSize": "11px",
                                   "fontFamily": "IBM Plex Mono, monospace",
                                   "marginBottom": "6px"},
                        ),
                        dcc.Loading(
                            type="circle", color=GOLD,
                            children=dcc.Graph(
                                id="leaderboard-alpha-chart",
                                config={"displayModeBar": False},
                                style={"height": "100%"},
                                figure=alpha_fig,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ])


def _build_summary_stats(trades_df: pd.DataFrame) -> list:
    """Build the row of stat cards at the top of the leaderboard."""
    if trades_df.empty:
        return [html.Div("No data", style={"color": "#7a90b0"})]

    n_trades    = len(trades_df)
    total_vol   = trades_df["AmountMidpoint"].sum()
    top_pol     = trades_df.groupby("Representative")["AmountMidpoint"].sum().idxmax()
    top_sector  = (
        trades_df[trades_df["Sector"] != "Unknown"]
        .groupby("Sector")["AmountMidpoint"].sum().idxmax()
        if "Sector" in trades_df.columns and not trades_df[trades_df["Sector"] != "Unknown"].empty
        else "—"
    )

    # % beats market (60d)
    if "alpha_60d" in trades_df.columns:
        valid = trades_df.dropna(subset=["alpha_60d"])
        pct_beat = f"{(valid['alpha_60d'] > 0).mean() * 100:.1f}%" if not valid.empty else "—"
    else:
        pct_beat = "—"

    def card(label: str, value: str, sub: str = "") -> html.Div:
        return html.Div(className="stat-card", style={"minWidth": "160px", "flex": "1"}, children=[
            html.Div(label, className="stat-card-label"),
            html.Div(value, className="stat-card-value"),
            html.Div(sub,   className="stat-card-sub"),
        ])

    return [
        card("Total Trades",         f"{n_trades:,}"),
        card("Est. Total Volume",    f"${total_vol/1e6:.1f}M"),
        card("% Beating S&P (60d)",  pct_beat),
        card("Most Traded Sector",   top_sector),
        card("Most Active Trader",   top_pol.split()[-1], top_pol),
    ]


def make_volume_leaderboard(trades_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart — top politicians by total estimated trade volume.

    Args:
        trades_df: Filtered trades DataFrame.

    Returns:
        Plotly Figure.
    """
    if trades_df.empty:
        return _empty_fig("No data.")

    agg = (
        trades_df.groupby(["Representative", "Party"])
        .agg(total_vol=("AmountMidpoint", "sum"), n_trades=("Ticker", "count"))
        .reset_index()
        .sort_values("total_vol", ascending=False)
        .head(TOP_N)
    )

    # Sort ascending so highest appears at top of horizontal bar chart
    agg = agg.sort_values("total_vol", ascending=True)

    colors = [DEM_BLUE if p == "Democrat" else REP_RED for p in agg["Party"]]

    fig = go.Figure(go.Bar(
        x=agg["total_vol"] / 1_000_000,
        y=agg["Representative"],
        orientation="h",
        marker_color=colors,
        customdata=agg[["Party", "n_trades"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Party: %{customdata[0]}<br>"
            "Est. volume: $%{x:.2f}M<br>"
            "Total trades: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(**_chart_layout("Est. Volume ($M)", ""))
    return fig


def make_alpha_leaderboard(trades_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart — top politicians by average 60-day alpha (min 10 trades).

    Args:
        trades_df: Filtered trades DataFrame.

    Returns:
        Plotly Figure.
    """
    if trades_df.empty or "alpha_60d" not in trades_df.columns:
        return _empty_fig("Alpha data not available for current filters.")

    valid = trades_df.dropna(subset=["alpha_60d"])
    if valid.empty:
        return _empty_fig("Not enough data to compute alpha rankings.")

    agg = (
        valid.groupby(["Representative", "Party"])
        .agg(avg_alpha=("alpha_60d", "mean"), n_trades=("alpha_60d", "count"))
        .reset_index()
    )
    # Require minimum 10 trades for statistical validity
    agg = agg[agg["n_trades"] >= 10]

    if agg.empty:
        return _empty_fig("Not enough trades per politician to rank by alpha.\n(Need ≥10 trades with alpha data.)")

    agg = agg.sort_values("avg_alpha", ascending=False).head(TOP_N)
    agg = agg.sort_values("avg_alpha", ascending=True)  # flip for horizontal chart

    colors = []
    for _, row in agg.iterrows():
        if row["avg_alpha"] > 0:
            c = DEM_BLUE if row["Party"] == "Democrat" else REP_RED
        else:
            c = "#374151"  # gray for negative alpha
        colors.append(c)

    fig = go.Figure(go.Bar(
        x=agg["avg_alpha"] * 100,
        y=agg["Representative"],
        orientation="h",
        marker_color=colors,
        customdata=agg[["Party", "n_trades"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Party: %{customdata[0]}<br>"
            "Avg 60d alpha: %{x:.2f}%<br>"
            "Trades with alpha data: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color="#7a90b0", line_width=1)

    fig.update_layout(**_chart_layout("Avg 60-Day Alpha (%)", ""))
    return fig


def _chart_layout(x_title: str, y_title: str) -> dict:
    """Shared layout dict for both leaderboard charts."""
    return {
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor":  CHART_BG,
        "font": {"color": TEXT_COLOR, "family": "IBM Plex Mono, monospace", "size": 10},
        "margin": {"l": 160, "r": 20, "t": 10, "b": 50},
        "xaxis": {
            "title": x_title,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
        "yaxis": {
            "title": y_title,
            "gridcolor": GRID_COLOR,
            "tickfont": {"size": 10},
        },
        "hovermode": "closest",
    }


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font={"color": TEXT_COLOR},
        annotations=[{
            "text": message.replace("\n", "<br>"),
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 12, "color": "#7a90b0"},
        }],
    )
    return fig


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("leaderboard-vol-chart",   "figure"),
    Output("leaderboard-alpha-chart", "figure"),
    Output("leaderboard-stats-row",   "children"),
    Input("store-filtered-trades", "data"),
)
def update_leaderboard(store_data: str):
    """Refresh both ranking charts and stats cards when filters change."""
    if not store_data:
        empty = _empty_fig("No data.")
        return empty, empty, []

    df = pd.read_json(store_data, orient="split")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["AmountMidpoint"]  = pd.to_numeric(df["AmountMidpoint"], errors="coerce").fillna(0)

    return (
        make_volume_leaderboard(df),
        make_alpha_leaderboard(df),
        _build_summary_stats(df),
    )
