"""
components/leaderboard.py
Tab 5: Leaderboard — top traders by volume and by alpha, plus summary stats panel.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output

from components.constants import (
    CHART_BG, PAPER_BG, GRID_COLOR, TEXT_COLOR, GOLD,
    DEM_BLUE, REP_RED, CHART_FONT, empty_fig,
)

TOP_N = 20  # number of politicians shown in each ranking


def build_leaderboard_tab(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> html.Div:
    """
    Build the Leaderboard tab: volume ranking, alpha ranking, and a summary stats row.
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
                                        "fontFamily": "JetBrains Mono, monospace",
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
                                   "fontFamily": "JetBrains Mono, monospace",
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

    # Safe idxmax — guard against empty groupby results
    vol_by_pol = trades_df.groupby("Representative")["AmountMidpoint"].sum()
    top_pol = vol_by_pol.idxmax() if not vol_by_pol.empty else "—"

    # Safe sector lookup
    known_sectors = trades_df[trades_df.get("Sector", pd.Series(dtype=str)) != "Unknown"] if "Sector" in trades_df.columns else pd.DataFrame()
    if not known_sectors.empty:
        sector_vol = known_sectors.groupby("Sector")["AmountMidpoint"].sum()
        top_sector = sector_vol.idxmax() if not sector_vol.empty else "—"
    else:
        top_sector = "—"

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

    top_pol_short = top_pol.split()[-1] if top_pol != "—" else "—"
    top_pol_full = top_pol if top_pol != "—" else ""

    return [
        card("Total Trades",         f"{n_trades:,}"),
        card("Est. Total Volume",    f"${total_vol/1e6:.1f}M"),
        card("% Beating S&P (60d)",  pct_beat),
        card("Most Traded Sector",   top_sector),
        card("Most Active Trader",   top_pol_short, top_pol_full),
    ]


def make_volume_leaderboard(trades_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart — top politicians by total estimated trade volume."""
    if trades_df.empty:
        return empty_fig("No data.")

    required = {"Representative", "Party", "AmountMidpoint", "Ticker"}
    if not required.issubset(trades_df.columns):
        return empty_fig("Missing required columns for volume leaderboard.")

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
    """Horizontal bar chart — top politicians by average 60-day alpha (min 10 trades)."""
    if trades_df.empty or "alpha_60d" not in trades_df.columns:
        return empty_fig("Alpha data not available for current filters.")

    valid = trades_df.dropna(subset=["alpha_60d"])
    if valid.empty:
        return empty_fig("Not enough data to compute alpha rankings.")

    agg = (
        valid.groupby(["Representative", "Party"])
        .agg(avg_alpha=("alpha_60d", "mean"), n_trades=("alpha_60d", "count"))
        .reset_index()
    )
    # Require minimum 10 trades for statistical validity
    agg = agg[agg["n_trades"] >= 10]

    if agg.empty:
        return empty_fig("Not enough trades per politician to rank by alpha.\n(Need ≥10 trades with alpha data.)")

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
        "font": CHART_FONT,
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


# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(
    Output("leaderboard-vol-chart",   "figure"),
    Output("leaderboard-alpha-chart", "figure"),
    Output("leaderboard-stats-row",   "children"),
    Input("store-filtered-trades", "data"),
)
def update_leaderboard(store_data: str):
    """Refresh both ranking charts and stats cards when filters change."""
    import data.state as _state

    df = _state.deserialize_store(store_data)
    if df is None:
        e = empty_fig("No data.")
        return e, e, []

    return (
        make_volume_leaderboard(df),
        make_alpha_leaderboard(df),
        _build_summary_stats(df),
    )
